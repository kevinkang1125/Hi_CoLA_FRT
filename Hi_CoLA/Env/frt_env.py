import numpy as np
import gym
from gym import spaces
import heapq
from dataclasses import dataclass

@dataclass
class Event:
    """Event in the simulation"""
    time: float
    event_type: str  # 'arrival', 'vehicle_dispatch'
    vehicle_id: int = None
    node: int = None
    
    def __lt__(self, other):
        return self.time < other.time

class FRT_Env(gym.Env):
    """
    Simple Event-Based Route Control - Step-Based Episodes
    
    SIMPLE DESIGN:
    - Vehicles appear at node 1 every dispatch_interval
    - Decisions at ALL nodes 1 to n_fixed_stops-1
    - Node n_fixed_stops is exit (vehicle disappears)
    - Each node can have a flexible alternative
    - Episode = max_steps decisions
    - Reward = passengers served between decisions
    
    Network: 1 → 2 → 3 → ... → n_fixed_stops (exit)
             ↓   ↓   ↓
             Flex stops (optional detours)
    
    STOCHASTICITY FLAGS:
    - stochastic_travel_time: Enable/disable random travel times
    - stochastic_cancellation: Enable/disable passenger cancellations at flexible stops
    """
    
    def __init__(self,
                 n_fixed_stops: int = 7,
                 max_steps: int = 50,
                 dispatch_interval: float = 12.0,
                 vehicle_capacity: int = 10000,
                 fixed_stop_arrival_rate: float = 6.67,
                 flexible_stop_arrival_rate: float = 8,#10
                 max_waiting_time: float = 15.0,
                 travel_time_fixed: float = 4.0,
                 travel_time_detour: float = 3.0,
                 flexible_cancel_prob: float = 0.3,
                 stochastic_travel_time: bool = True,
                 travel_time_std: float = 2.0,
                 stochastic_cancellation: bool = True):
        
        super().__init__()
        
        self.n_fixed_stops = n_fixed_stops
        self.n_flexible_stops = n_fixed_stops - 1  # Each fixed has flexible except exit
        self.n_nodes = n_fixed_stops + self.n_flexible_stops
        
        # Episode control
        self.max_steps = max_steps
        self.max_time = 10000.0  # Safety limit
        
        # Parameters
        self.dispatch_interval = dispatch_interval
        self.vehicle_capacity = vehicle_capacity
        self.fixed_stop_arrival_rate = fixed_stop_arrival_rate
        self.flexible_stop_arrival_rate = flexible_stop_arrival_rate
        self.max_waiting_time = max_waiting_time
        self.travel_time_fixed = travel_time_fixed
        self.travel_time_detour = travel_time_detour
        self.flexible_cancel_prob = flexible_cancel_prob
        
        # Stochasticity flags
        self.stochastic_travel_time = stochastic_travel_time
        self.travel_time_std = travel_time_std
        self.stochastic_cancellation = stochastic_cancellation
        
        # Network mapping: fixed node i → flexible node (n_fixed_stops + i)
        self.fixed_to_flexible = {i: n_fixed_stops + i for i in range(1, n_fixed_stops)}
        self.flexible_to_fixed = {n_fixed_stops + i: i for i in range(1, n_fixed_stops)}
        
        # Action space: one binary decision per node (except exit)
        self.action_space = spaces.MultiBinary(n_fixed_stops - 1)
        
        # Build edge list for compact representation
        self._build_edge_list()
        
        # Observation space (compact)
        # Demand: n_nodes
        # Vehicles at nodes: n_nodes
        # Vehicles on edges: len(edge_list)
        # Travel times: (n_fixed_stops - 1) fixed edges + 1 flexible edge time (2x since round trip)
        obs_dim = (
            self.n_nodes +           # Demand
            self.n_nodes +           # Vehicles at nodes
            len(self.edge_list) +    # Vehicles on edges (compact)
            (n_fixed_stops - 1) +    # Travel times on fixed edges
            1                        # Travel time on flexible edges (2x detour time)
        )
        self.observation_space = spaces.Box(0, np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Build travel time matrix (mean values)
        self._build_network()
        
        self.reset()
    
    def _build_edge_list(self):
        """Build ordered edge list: fixed edges first, then flexible edges
        
        Fixed edges (node 1 to n-1): (1→2), (2→3), ..., (n-1→n)
        Flexible edges: (1→flex1, flex1→2), (2→flex2, flex2→3), ...
        """
        self.edge_list = []
        self.edge_to_idx = {}
        
        # Fixed edges (consecutive): 1→2, 2→3, ..., (n-1)→n
        for i in range(1, self.n_fixed_stops):
            edge = (i, i + 1)
            self.edge_to_idx[edge] = len(self.edge_list)
            self.edge_list.append(edge)
        
        # Flexible edges (detours): fixed→flex, flex→next_fixed
        for fixed in range(1, self.n_fixed_stops):
            flex = self.fixed_to_flexible.get(fixed)
            if flex:
                # fixed → flexible
                edge1 = (fixed, flex)
                self.edge_to_idx[edge1] = len(self.edge_list)
                self.edge_list.append(edge1)
                
                # flexible → next fixed
                next_fixed = fixed + 1
                if next_fixed <= self.n_fixed_stops:
                    edge2 = (flex, next_fixed)
                    self.edge_to_idx[edge2] = len(self.edge_list)
                    self.edge_list.append(edge2)
    
    def _build_network(self):
        """Build travel time matrix (mean values)"""
        n = self.n_nodes
        self.travel_times = np.full((n, n), np.inf)
        
        # Consecutive fixed stops
        for i in range(1, self.n_fixed_stops):
            self.travel_times[i-1, i] = self.travel_time_fixed
        
        # Detours: fixed to flexible
        for fixed, flex in self.fixed_to_flexible.items():
            self.travel_times[fixed-1, flex-1] = self.travel_time_detour
        
        # Return: flexible to next fixed
        for flex, fixed in self.flexible_to_fixed.items():
            next_fixed = fixed + 1
            if next_fixed <= self.n_fixed_stops:
                self.travel_times[flex-1, next_fixed-1] = self.travel_time_detour
    
    def _sample_travel_time(self, from_node: int, to_node: int) -> float:
        """Sample travel time with optional stochasticity
        
        Mean travel time equals the value in travel_times matrix.
        When stochastic_travel_time=True, samples from Gamma(shape, scale)
        parameterized to have mean=travel_times[i,j] and std=travel_time_std.
        """
        mean_time = self.travel_times[from_node - 1, to_node - 1]
        
        if not self.stochastic_travel_time or mean_time == np.inf:
            return mean_time
        
        # Use gamma distribution to ensure positive values
        # Parameterize by mean and std
        # For gamma: mean = shape * scale, var = shape * scale^2
        # So: shape = (mean / std)^2, scale = std^2 / mean
        shape = (mean_time / self.travel_time_std) ** 2
        scale = (self.travel_time_std ** 2) / mean_time
        
        sampled_time = np.random.gamma(shape, scale)
        sampled_time = min(sampled_time, 2.0)  # Cap travel time at 2.0 minutes (upper bound, as used in the paper)
        # lo = mean_time - 1.0
        # hi = mean_time + 1.0
        # sampled_time = np.random.uniform(lo, hi)
        return sampled_time
    
    def reset(self):
        """Reset environment"""
        self.current_time = 0.0
        self.current_step = 0
        
        # Vehicles (now tracking edges)
        self.vehicles = {}
        self.next_vehicle_id = 0
        
        # Events
        self.event_queue = []
        self._schedule_dispatches()
        
        # Demand
        self.demand_queues = {i: [] for i in range(1, self.n_nodes + 1)}
        self._generate_demand()
        
        # Stats
        self.total_served = 0
        self.total_abandoned = 0
        self.total_cancelled = 0
        self.reward_accumulator = 0
        
        # Current decision
        self.current_vehicle_id = None
        self.current_node = None
        
        # Process to first decision
        self._process_to_decision()
        
        return self._get_obs()
    
    def _schedule_dispatches(self):
        """Schedule vehicle arrivals at node 1"""
        t = 0.0
        while t < self.max_time:
            heapq.heappush(self.event_queue, Event(t, 'vehicle_dispatch'))
            t += self.dispatch_interval
    
    def _generate_demand(self):
        """Generate passenger arrivals"""
        # Fixed stops
        for node in range(1, self.n_fixed_stops + 1):
            t = 0.0
            while t < self.max_time:
                t += np.random.exponential(self.fixed_stop_arrival_rate)
                if t < self.max_time:
                    self.demand_queues[node].append(t)
        # print(self.demand_queues)
        
        # Flexible stops
        for node in range(self.n_fixed_stops + 1, self.n_nodes + 1):
            t = 0.0
            while t < self.max_time:
                t += np.random.exponential(self.flexible_stop_arrival_rate)
                if t < self.max_time:
                    self.demand_queues[node].append(t)
    
    def _update_demand(self):
        """Remove abandoned passengers"""
        for node in range(1, self.n_nodes + 1):
            before = len(self.demand_queues[node])
            self.demand_queues[node] = [
                t for t in self.demand_queues[node]
                if self.current_time - t <= self.max_waiting_time
            ]
            after = len(self.demand_queues[node])
            self.total_abandoned += (before - after)
    
    def _serve_passengers(self, node: int, capacity: int) -> int:
        """Serve passengers at node"""
        available = [t for t in self.demand_queues[node] if t <= self.current_time]
        if not available:
            return 0
        
        n_to_serve = min(len(available), capacity)
        
        # Cancellation at flexible stops (if enabled)
        n_cancelled = 0
        if node > self.n_fixed_stops and self.stochastic_cancellation:
            n_cancelled = np.random.binomial(n_to_serve, self.flexible_cancel_prob)
            n_to_serve -= n_cancelled
            self.total_cancelled += n_cancelled
            
            # Remove cancelled passengers
            for _ in range(n_cancelled):
                if available:
                    self.demand_queues[node].remove(available.pop(0))
        
        # Serve passengers
        for i in range(n_to_serve):
            if available:
                self.demand_queues[node].remove(available[i])
        
        self.total_served += n_to_serve
        self.reward_accumulator += n_to_serve
        return n_to_serve
    
    def _dispatch_vehicle(self):
        """Create vehicle at node 1"""
        vehicle = {
            'id': self.next_vehicle_id,
            'node': 1,
            'capacity': self.vehicle_capacity,
            'on_edge': False,
            'from_node': None,
            'to_node': None
        }
        self.vehicles[self.next_vehicle_id] = vehicle
        self.next_vehicle_id += 1
        
        # Serve passengers immediately
        served = self._serve_passengers(1, vehicle['capacity'])
        vehicle['capacity'] -= served
        
        # This vehicle needs decision at node 1
        self.current_vehicle_id = vehicle['id']
        self.current_node = 1
    
    def _process_to_decision(self):
        """Process events until next decision needed"""
        self.current_vehicle_id = None
        self.current_node = None
        
        while self.event_queue and self.current_step < self.max_steps:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            
            if self.current_time >= self.max_time:
                return True
            
            self._update_demand()
            
            if event.event_type == 'vehicle_dispatch':
                self._dispatch_vehicle()
                return False  # Need decision
            
            elif event.event_type == 'arrival':
                if event.vehicle_id not in self.vehicles:
                    continue
                
                vehicle = self.vehicles[event.vehicle_id]
                vehicle['node'] = event.node
                vehicle['on_edge'] = False
                vehicle['from_node'] = None
                vehicle['to_node'] = None
                
                # Serve passengers
                served = self._serve_passengers(event.node, vehicle['capacity'])
                vehicle['capacity'] -= served
                
                # Exit?
                if event.node == self.n_fixed_stops:
                    del self.vehicles[event.vehicle_id]
                    continue
                
                # Fixed stop? Need decision
                if event.node <= self.n_fixed_stops:
                    self.current_vehicle_id = event.vehicle_id
                    self.current_node = event.node
                    return False
                
                # Flexible stop? Auto-route to next fixed
                if event.node in self.flexible_to_fixed:
                    next_node = self.flexible_to_fixed[event.node] + 1
                    if next_node <= self.n_fixed_stops:
                        travel_time = self._sample_travel_time(event.node, next_node)
                        vehicle['on_edge'] = True
                        vehicle['from_node'] = event.node
                        vehicle['to_node'] = next_node
                        heapq.heappush(self.event_queue,
                                      Event(self.current_time + travel_time, 'arrival',
                                           vehicle['id'], next_node))
        
        return True  # Done
    
    def _get_obs(self):
        """Get compact observation
        
        Structure:
        1. Demand at each node (n_nodes)
        2. Vehicles at each node (n_nodes)
        3. Vehicles on each edge (len(edge_list)) - ordered as edge_list
        4. Travel times on fixed edges (n_fixed_stops - 1)
        5. Travel time on flexible edges (1) - represented as 2x detour time
        
        Edge ordering:
        - Fixed edges first: (1→2), (2→3), ..., (n-1→n)
        - Flexible edges: (1→flex1), (flex1→2), (2→flex2), (flex2→3), ...
        """
        # 1. Demand at each node
        demand = np.zeros(self.n_nodes)
        for node in range(1, self.n_nodes + 1):
            demand[node - 1] = sum(1 for t in self.demand_queues[node] 
                                  if t <= self.current_time)
        
        # 2. Vehicles at nodes
        vehicles_nodes = np.zeros(self.n_nodes)
        for v in self.vehicles.values():
            if not v['on_edge']:
                vehicles_nodes[v['node'] - 1] += 1
        
        # 3. Vehicles on edges (compact representation)
        vehicles_edges = np.zeros(len(self.edge_list))
        for v in self.vehicles.values():
            if v['on_edge'] and v['from_node'] is not None and v['to_node'] is not None:
                edge = (v['from_node'], v['to_node'])
                if edge in self.edge_to_idx:
                    idx = self.edge_to_idx[edge]
                    vehicles_edges[idx] += 1
        
        # 4. Travel times on fixed edges only (consecutive edges)
        travel_times_fixed = np.zeros(self.n_fixed_stops - 1)
        for i in range(self.n_fixed_stops - 1):
            travel_times_fixed[i] = self.travel_times[i, i + 1]
        
        # 5. Travel time on flexible edges (2x detour time for round trip)
        travel_time_flexible = np.array([2 * self.travel_time_detour])
        
        obs = np.concatenate([
            demand,
            vehicles_nodes,
            vehicles_edges,
            travel_times_fixed,
            travel_time_flexible
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action):
        """Execute action"""
        # Route current vehicle
        if self.current_vehicle_id in self.vehicles:
            vehicle = self.vehicles[self.current_vehicle_id]
            node = vehicle['node']
            
            # Decide where to go
            action_idx = node - 1
            if action_idx < len(action) and action[action_idx] == 1:
                # Try detour
                flex = self.fixed_to_flexible.get(node)
                if flex:
                    demand = sum(1 for t in self.demand_queues[flex] if t <= self.current_time)
                    next_node = flex if demand > 0 else node + 1
                else:
                    next_node = node + 1
            else:
                # Go direct
                next_node = node + 1
            
            # Schedule travel with stochastic time
            if next_node <= self.n_nodes:
                travel_time = self._sample_travel_time(node, next_node)
                if travel_time < np.inf:
                    vehicle['on_edge'] = True
                    vehicle['from_node'] = node
                    vehicle['to_node'] = next_node
                    heapq.heappush(self.event_queue,
                                  Event(self.current_time + travel_time, 'arrival',
                                       vehicle['id'], next_node))
        
        # Get reward
        reward = self.reward_accumulator
        self.reward_accumulator = 0
        
        # Increment step
        self.current_step += 1
        
        # Process to next decision
        done = self._process_to_decision()
        
        # Info
        hours = max(self.current_time / 60.0, 0.01)
        info = {
            'step': self.current_step,
            'time': self.current_time,
            'passengers_served': self.total_served,
            'passengers_abandoned': self.total_abandoned,
            'passengers_cancelled': self.total_cancelled,
            'passengers_per_hour': self.total_served / hours,
            'stochastic_travel': self.stochastic_travel_time,
            'stochastic_cancellation': self.stochastic_cancellation,
            'obs_dim': len(self._get_obs()),
            'n_edges': len(self.edge_list)
        }
        
        return self._get_obs(), reward, done, info

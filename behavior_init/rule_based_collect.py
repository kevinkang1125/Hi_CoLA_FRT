"""
Collect episodes with the rule-based dispatching heuristic and build the
imitation-learning dataset (observation -> rule detour probabilities).

Outputs (written next to this script): imitation_dataset.npz,
rule_based_trajectories.parquet, collection_summary.csv.
Then run train_imitation.py to fit the initial behavior policy.
"""
import os
import sys
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import pandas as pd


def rule_based_policy(obs, env):
    """
    Rule-based policy for route control
    
    Logic:
    - If demand at next flexible stop > 0.5 * demand at next fixed stop:
        Output 0.7 probability to detour
    - Otherwise:
        Output 0.2 probability to detour
    
    Args:
        obs: observation vector
        env: environment instance
        current_node: current vehicle node
    
    Returns:
        action_probs: array of probabilities for each action
    """
    # Parse observation to get demand
    # Observation structure: [demand, vehicles_nodes, vehicles_edges, travel_times]
    n_nodes = env.n_nodes
    demand = obs[:n_nodes]
    
    # Initialize action probabilities (one per fixed stop except exit)
    action_probs = np.zeros(env.action_space.n)
    
    # Apply rule to each decision point
    for fixed_node in range(1, env.n_fixed_stops):
        action_idx = fixed_node - 1
        
        # Get next fixed and flexible nodes
        next_fixed = fixed_node + 1
        flexible_node = env.fixed_to_flexible.get(fixed_node)
        
        if flexible_node:
            # Get demands (convert to 0-indexed)
            demand_flexible = demand[flexible_node - 1]
            demand_next_fixed = demand[next_fixed - 1] if next_fixed <= env.n_fixed_stops else 0
            
            # Apply rule
            if demand_flexible > 0.5 * demand_next_fixed:
                action_probs[action_idx] = 0.7
            else:
                action_probs[action_idx] = 0.2
        else:
            action_probs[action_idx] = 0.2
    
    return action_probs




def simulate_trajectory(seed, env_config, gamma):
    """
    Simulate one episode trajectory using rule-based policy
    
    Returns:
        trajectory dict with observations and rule-based outputs
    """
    np.random.seed(seed)
    
    # Create environment
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Hi_CoLA.Env.frt_env import FRT_Env
    env = FRT_Env(**env_config)
    
    # Trajectory data
    trajectory = {
        "observations": [],
        "rule_outputs": [],  # Rule-based policy outputs (target for MLP)
        "actions": [],       # Actual sampled actions
        "rewards": [],
        "return": 0.0
    }
    
    obs = env.reset()
    done = False
    discounted_return = 0.0
    discount = 1.0
    
    while not done:
        # Store observation
        trajectory["observations"].append(obs.copy())
        
        # Get rule-based policy output
        action_probs = rule_based_policy(obs, env)
        trajectory["rule_outputs"].append(action_probs.copy())
        
        # Sample binary action from rule-based probabilities
        action = np.random.binomial(1, action_probs)
        trajectory["actions"].append(action.copy())
        
        # Step
        obs, reward, done, info = env.step(action)
        trajectory["rewards"].append(reward)
        
        # Accumulate return
        discounted_return += discount * reward
        discount *= gamma
    
    trajectory["return"] = discounted_return
    trajectory["episode_length"] = len(trajectory["rewards"])
    trajectory["simulation_time"] = info.get('time', 0)
    trajectory["passengers_per_hour"] = info.get('passengers_per_hour', 0)
    trajectory["total_served"] = info.get('passengers_served', 0)
    
    return trajectory, discounted_return


def collect_trajectories(env_config, n_trajectories, gamma=0.99, 
                         batch_size=100, n_workers=4):
    """
    Collect trajectories using rule-based policy in parallel
    
    Args:
        env_config: dict with environment parameters
        n_trajectories: number of trajectories to collect
        gamma: discount factor
        batch_size: trajectories per batch
        n_workers: number of parallel processes
    
    Returns:
        DataFrame with trajectories, list of returns
    """
    simulate_fn = partial(
        simulate_trajectory,
        env_config=env_config,
        gamma=gamma
    )
    
    all_rows = []
    all_returns = []
    
    print(f"Collecting {n_trajectories} trajectories with rule-based policy...")
    print(f"Rule: detour_prob = 0.7 if demand_flex > 0.5*demand_fixed, else 0.2")
    print("=" * 70)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for batch_start in tqdm(range(0, n_trajectories, batch_size), 
                               desc="Collecting trajectories"):
            batch_end = min(batch_start + batch_size, n_trajectories)
            seeds = range(batch_start, batch_end)
            
            results = list(executor.map(simulate_fn, seeds))
            
            for traj_id, (traj, ret) in enumerate(results):
                row = {
                    'trajectory_id': batch_start + traj_id,
                    'return': traj['return'],
                    'episode_length': traj['episode_length'],
                    'simulation_time': traj['simulation_time'],
                    'passengers_per_hour': traj['passengers_per_hour'],
                    'total_served': traj['total_served'],
                    'observations': traj['observations'],
                    'rule_outputs': traj['rule_outputs'],  # Target for supervised learning
                    'actions': traj['actions'],
                    'rewards': traj['rewards']
                }
                all_rows.append(row)
                all_returns.append(ret)
    
    df = pd.DataFrame(all_rows)
    return df, all_returns


def prepare_imitation_dataset(df):
    """
    Convert trajectory DataFrame to (observation, target) pairs for MLP training
    
    Args:
        df: DataFrame with trajectories
    
    Returns:
        X: np.array of observations [N_total_steps, obs_dim]
        y: np.array of rule outputs [N_total_steps, action_dim]
    """
    all_observations = []
    all_targets = []
    
    for _, row in df.iterrows():
        observations = row['observations']
        rule_outputs = row['rule_outputs']
        
        for obs, target in zip(observations, rule_outputs):
            all_observations.append(obs)
            all_targets.append(target)
    
    X = np.array(all_observations)
    y = np.array(all_targets)
    
    return X, y


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    # Environment config
    env_config = {
        'n_fixed_stops': 5,
        'max_steps': 1024,
        'dispatch_interval': 12.0,
        'vehicle_capacity': 1000,
        'stochastic_travel_time': True,
        'travel_time_std': 2.0,
        'stochastic_cancellation': True
    }
    
    # Collect trajectories
    print("\nCollecting trajectories with rule-based policy...")
    df, returns = collect_trajectories(
        env_config=env_config,
        n_trajectories=20,
        gamma=0.99, 
        batch_size=100,
        n_workers=8
    )
    
    # Statistics
    print("\n" + "=" * 70)
    print("TRAJECTORY COLLECTION RESULTS")
    print("=" * 70)
    print(f"✓ Collected {len(df)} trajectories")
    print(f"\nPerformance Metrics:")
    print(f"  Avg return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Avg episode length: {df['episode_length'].mean():.1f} steps")
    print(f"  Avg simulation time: {df['simulation_time'].mean():.1f} minutes")
    print(f"  Avg passengers/hour: {df['passengers_per_hour'].mean():.1f} ± {df['passengers_per_hour'].std():.1f}")
    print(f"  Avg total served: {df['total_served'].mean():.1f} passengers")
    
    # Prepare imitation learning dataset
    X, y = prepare_imitation_dataset(df)
    
    print("\n" + "=" * 70)
    print("IMITATION LEARNING DATASET")
    print("=" * 70)
    print(f"Total training samples: {len(X)}")
    print(f"Observation shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nSample observation (first 10 dims): {X[0][:10]}")
    print(f"Sample target: {y[0]}")
    np.savez_compressed(
        os.path.join(out_dir, "imitation_dataset.npz"),
        observations=X,
        targets=y
    )
    # Analyze rule outputs
    print("\n" + "=" * 70)
    print("RULE OUTPUT STATISTICS")
    print("=" * 70)
    high_prob = (y == 0.7).sum()
    low_prob = (y == 0.2).sum()
    total_decisions = y.size
    print(f"High detour probability (0.7): {high_prob / total_decisions * 100:.1f}%")
    print(f"Low detour probability (0.2): {low_prob / total_decisions * 100:.1f}%")
    
    # Save data
    print("\n" + "=" * 70)
    print("SAVING DATA")
    print("=" * 70)
    
    # Save full trajectories
    df.to_parquet(os.path.join(out_dir, "rule_based_trajectories.parquet"))
    print("✓ Saved trajectories to: rule_based_trajectories.parquet")
    
    print("✓ Saved imitation dataset to: imitation_dataset.npz")
    
    # Save summary statistics
    summary = {
        'n_trajectories': len(df),
        'total_steps': len(X),
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_passengers_per_hour': df['passengers_per_hour'].mean(),
        'std_passengers_per_hour': df['passengers_per_hour'].std(),
        'obs_dim': X.shape[1],
        'action_dim': y.shape[1]
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "collection_summary.csv"), index=False)
    print("✓ Saved summary to: collection_summary.csv")
    
    print("\n" + "=" * 70)
    print("READY FOR IMITATION LEARNING!")
    print("=" * 70)
    print("\nTo train your MLP:")
    print("1. Load data: data = np.load('imitation_dataset.npz')")
    print("2. Get X, y: X = data['observations'], y = data['targets']")
    print("3. Train MLP to minimize: loss = MSE(MLP(X), y)")
    print("   or use BCE if treating as independent Bernoulli probabilities")
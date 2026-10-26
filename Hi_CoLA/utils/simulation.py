"""Rollout utilities: simulate long FRT episodes with a behavior policy and
split them into non-overlapping 60-minute trajectories."""
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pandas as pd
import torch


def simulate_long_episode(sim_num, env, gamma, pi, min_steps=900, warmup_steps=100):
    """
    Simulate one long episode to extract multiple short trajectories
    
    Args:
        sim_num: simulation number for seeding (MUST BE FIRST for executor.map)
        env: environment instance
        gamma: discount factor
        pi: policy function
        min_steps: minimum number of steps to simulate (after warmup)
        warmup_steps: number of steps to run before collecting data
    
    Returns:
        dict with full episode data (only after warmup period)
    """
    # Episode data
    episode_data = {
        "observations": [],
        "actions": [],           # Action probabilities from policy
        "actions_deter": [],     # Sampled deterministic actions
        "rewards": [],
        "timestamps": []         # Track simulation time
    }
    
    obs = env.reset()
    done = False
    step_count = 0
    
    # Warmup phase - run simulation but don't collect data
    while not done and step_count < warmup_steps:
        obs = torch.tensor(obs, dtype=torch.float32)
        action_probs = pi(obs)
        action_probs = np.array(action_probs.squeeze(0).detach().tolist())
        action = np.random.binomial(1, action_probs)
        obs, reward, done, info = env.step(action)
        step_count += 1
    
    # If episode ended during warmup, return empty data
    if done:
        episode_data["total_steps"] = 0
        return episode_data
    
    # Collection phase - now start collecting data
    collection_steps = 0
    while not done and collection_steps < min_steps:  # Allow for longer episodes
        episode_data["observations"].append(obs.tolist() if isinstance(obs, np.ndarray) else obs)
        episode_data["timestamps"].append(env.current_time)  # Track simulation time
        obs = torch.tensor(obs, dtype=torch.float32)
        
        # Get rule-based policy output (action probabilities)
        action_probs = pi(obs)
        action_probs = np.array(action_probs.squeeze(0).detach().tolist())
        episode_data["actions"].append(action_probs.tolist() if isinstance(action_probs, np.ndarray) else action_probs)
        
        # Sample binary action from rule-based probabilities
        action = np.random.binomial(1, action_probs)
        episode_data["actions_deter"].append(action.tolist() if isinstance(action, np.ndarray) else action)
        
        # Step
        obs, reward, done, info = env.step(action)
        episode_data["rewards"].append(reward)
        
        collection_steps += 1
    
    episode_data["total_steps"] = collection_steps
    # print(episode_data["timestamps"][-1])
    return episode_data


def extract_trajectories_from_episode(episode_data, episode_num, traj_time_threshold, gamma, trajectory_counter):
    """
    Extract non-overlapping sequential trajectories from a single episode based on time threshold
    
    Args:
        episode_data: dict with observations, actions, rewards, timestamps
        episode_num: episode identifier
        traj_time_threshold: time threshold in minutes (e.g., 60 for 1 hour)
        gamma: discount factor
        trajectory_counter: starting counter for trajectory IDs
    
    Returns:
        list of trajectory rows (one per trajectory)
    """
    total_steps = episode_data["total_steps"]
    rows = []
    returns = []
    
    if total_steps == 0:
        return rows, returns, trajectory_counter
    
    # Extract non-overlapping trajectories based on time
    start_idx = 0
    
    while start_idx < total_steps:
        # Get starting time for this trajectory
        start_time = episode_data["timestamps"][start_idx]
        if episode_data["timestamps"][total_steps-1] - start_time <= traj_time_threshold:
            break
        # Find the end index where time exceeds threshold
        end_idx = start_idx + 1
        while end_idx < total_steps:
            elapsed_time = episode_data["timestamps"][end_idx] - start_time
            if elapsed_time >= traj_time_threshold:
                # Include this step to ensure we cross the threshold
                end_idx += 1
                break
            end_idx += 1
        
        # If we couldn't find enough steps to cross threshold, skip remaining steps
        if end_idx - start_idx < 2:  # Need at least 2 steps for a trajectory
            break
        
        traj_length = end_idx - start_idx
        
        # Calculate discounted return for this trajectory
        discounted_return = 0.0
        discount = 1.0
        for i in range(start_idx, end_idx):
            discounted_return += discount * episode_data["rewards"][i]
            discount *= gamma
        
        # Create row with independent columns
        row = {
            "trajectory_id": trajectory_counter,
            "episode_id": episode_num,
            "return": discounted_return,
            "traj_length": traj_length,
            "time_duration": episode_data["timestamps"][end_idx-1] - start_time
        }
        
        # Add observations, rule outputs (target), actions, rewards as separate columns
        for t in range(traj_length):
            idx = start_idx + t
            
            obs = episode_data["observations"][idx]
            action_prob = episode_data["actions"][idx]           # Policy output (probabilities)
            action_deter = episode_data["actions_deter"][idx]    # Sampled action
            reward = episode_data["rewards"][idx]
            timestamp = episode_data["timestamps"][idx]
            
            # Independent columns for each timestep
            row[f"obs_{t}"] = obs
            row[f"action_{t}"] = action_prob         # Target for supervised learning
            row[f"action_deter_{t}"] = action_deter  # Actual sampled action
            row[f"reward_{t}"] = reward
            row[f"timestamp_{t}"] = timestamp
        
        rows.append(row)
        returns.append(discounted_return)
        trajectory_counter += 1
        
        # Move to next trajectory
        start_idx = end_idx
    
    return rows, returns, trajectory_counter


def collect_trajectories(env, n_episodes, pi, gamma=0.99, 
                         traj_time_threshold=60, min_steps=800, warmup_steps=100, batch_size=100, n_workers=4):
    """
    Collect time-based trajectories with independent state/action/reward columns
    
    Args:
        env: environment instance
        n_episodes: number of long episodes to simulate
        pi: policy function
        gamma: discount factor
        traj_time_threshold: time threshold in minutes for each trajectory (e.g., 60 for 1 hour)
        min_steps: minimum steps per episode
        batch_size: episodes per batch for parallel processing
        n_workers: number of parallel processes
    
    Returns:
        DataFrame with independent columns, list of returns
    """
    all_rows = []
    all_returns = []
    trajectory_counter = 0
    
    print(f"Collecting trajectories from {n_episodes} episodes...")
    print(f"Time threshold: {traj_time_threshold} minutes, Min steps per episode: {min_steps}")
    print("=" * 70)
    
    # Fixed: partial now binds env, gamma, pi, min_steps as keyword args
    # executor.map will pass episode_nums as first positional arg (sim_num)
    simulate_fn = partial(
        simulate_long_episode,
        env=env,
        gamma=gamma,
        pi=pi,
        min_steps=min_steps,
        warmup_steps=warmup_steps
    )
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for batch_start in tqdm(range(0, n_episodes, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, n_episodes)
            episode_nums = range(batch_start, batch_end)
            
            # Simulate episodes in parallel
            episode_results = list(executor.map(simulate_fn, episode_nums))
            
            # Extract trajectories from each episode
            for episode_num, episode_data in enumerate(episode_results, start=batch_start):
                rows, returns, trajectory_counter = extract_trajectories_from_episode(
                    episode_data, 
                    episode_num, 
                    traj_time_threshold, 
                    gamma,
                    trajectory_counter
                )
                all_rows.extend(rows)
                all_returns.extend(returns)
    
    df = pd.DataFrame(all_rows)
    
    return df, all_returns


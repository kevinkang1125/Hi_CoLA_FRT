"""Hi-CoLA utilities.

- Policy utilities: KL between policies, parameter perturbation, KL binning.
- Rollout: simulate long FRT episodes with a behavior policy and split them
  into non-overlapping 60-minute trajectories.
- High-confidence off-policy evaluation (Thomas et al., 2015): per-decision
  importance sampling + concentration-inequality lower bound on the normalized
  return of each evaluation policy.
"""
import copy
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from threading import Lock



# ===========================================================================
# Policy utilities
# ===========================================================================

def kl_between_policies(pi_perturbed, pi, states, eps=1e-8):
    """
    Calculate KL divergence between two policies for MultiBinary action space.
    Each of 4 outputs represents P(deviate) for independent binary decisions.
    
    Returns average KL per dimension (not sum) to keep values in reasonable range.
    """
    with torch.no_grad():
        probs_pi = pi(states)  # [batch, 4] - probabilities for each dimension
        probs_pert = pi_perturbed(states)  # [batch, 4]
        
        # Clamp probabilities for numerical stability
        p = torch.clamp(probs_pi, eps, 1 - eps)
        q = torch.clamp(probs_pert, eps, 1 - eps)
        
        # KL divergence for Bernoulli: KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
        kl_per_dim = (
            p * torch.log(p / q) + 
            (1 - p) * torch.log((1 - p) / (1 - q))
        )  # [batch, 4]
        
        # CRITICAL: Average across dimensions, not sum!
        # This keeps KL values in reasonable range (0-1 typically)
        kl_avg_dims = kl_per_dim.mean(dim=-1)  # [batch] - average KL per sample
        kl = kl_avg_dims.mean()  # Scalar - average across batch
        
    return kl


def perturb(pi, c=0.3, generator=None):
    """
    Perturb the policy parameters with Gaussian noise where sigma is proportional to each weight's magnitude.

    Args:
        pi (Policy): the original policy network
        c (float): proportional constant to scale sigma = c * |weight|
        generator (torch.Generator, optional): RNG for the noise (global RNG if None)

    Returns:
        pi_perturbed (Policy): the perturbed policy
    """
    # deepcopy rather than Behavioural_v2(): constructing a network would draw
    # its (discarded) initial weights from the global RNG
    pi_perturbed = copy.deepcopy(pi)

    with torch.no_grad():
        for param in pi_perturbed.parameters():
            if param.requires_grad:
                sigma = torch.sqrt(c * param.abs() + 1e-8)
                noise = torch.randn(param.shape, generator=generator, dtype=param.dtype) * sigma
                param.add_(noise)

    return pi_perturbed

def get_flat_params(model):
    # Flatten the Behavioural model's parameters into a vector
    return torch.cat([p.view(-1) for p in model.parameters()])

def get_kl_bin(kl_score):
    bins = np.arange(0, 0.15, 0.05)
    for start in bins:
        end = round(start + 0.05, 2)
        if start <= kl_score < end:
            return f"{start:.2f}-{end:.2f}"
    return None

# ===========================================================================
# Rollout
# ===========================================================================

def simulate_long_episode(sim_num, env, gamma, pi, min_steps=900, warmup_steps=100, seed=None):
    """
    Simulate one long episode to extract multiple short trajectories
    
    Args:
        sim_num: simulation number for seeding (MUST BE FIRST for executor.map)
        env: environment instance
        gamma: discount factor
        pi: policy function
        min_steps: minimum number of steps to simulate (after warmup)
        warmup_steps: number of steps to run before collecting data
        seed: base seed; the episode is seeded with seed + sim_num (unseeded if None)
    
    Returns:
        dict with full episode data (only after warmup period)
    """
    if seed is not None:
        np.random.seed(seed + sim_num)
        torch.manual_seed(seed + sim_num)

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
                         traj_time_threshold=60, min_steps=800, warmup_steps=100, batch_size=100, n_workers=4,
                         seed=None):
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
        seed: base seed; episode i is seeded with seed + i (unseeded if None)
    
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
        warmup_steps=warmup_steps,
        seed=seed
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

# ===========================================================================
# Thomas high-confidence lower bound
# ===========================================================================

# Global GPU lock for thread safety
gpu_lock = Lock()

# ---------------------------
# Core utilities
# ---------------------------

def _pairwise_variance_terms(y):
    """
    Compute A = n*sum(y^2) - (sum(y))^2  (>= 0 up to FP noise)
    and return (A_clipped, n, S1, S2).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    S1 = y.sum()
    S2 = np.dot(y, y)
    A = n * S2 - S1 * S1
    if A < 0:  # numerical guard
        A = 0.0
    return A, n, S1, S2

def _objective_F_on_pre(c, X_pre_sorted, P1, P2, n_pre, n_post, delta):
    """
    Evaluate the selection objective F(c) on D_pre:
      F(c) = mean(Y) 
             - 7 c ln(2/delta) / (3 (n_post - 1))
             - sqrt( [ln(2/delta)/n_post] * [ 2/(n_pre(n_pre-1)) * (n_pre*S2 - S1^2) ] )
    where Y_i = min{X_i, c}.
    """
    if c < 1.0:
        c = 1.0

    # find k = count of X_pre_sorted <= c
    # X_pre_sorted is ascending
    k = np.searchsorted(X_pre_sorted, c, side='right')

    # S1 = sum(min(X_i, c)), S2 = sum(min(X_i,c)^2)
    S1 = P1[k] + (n_pre - k) * c
    S2 = P2[k] + (n_pre - k) * (c * c)

    ln = np.log(2.0 / delta)

    # mean(Y) on pre
    mean_pre = S1 / n_pre

    # linear penalty term in c (depends on n_post)
    if n_post <= 1:
        # if post is too small, penalize heavily to avoid invalid selection
        term_lin = np.inf
    else:
        term_lin = 7.0 * c * ln / (3.0 * (n_post - 1))

    # pairwise-variance term on pre (uses n_post in front per paper's selection recipe)
    if n_pre <= 1:
        term_var = np.inf
    else:
        A = n_pre * S2 - S1 * S1  # >= 0
        # \hat{\sigma}^2_pairwise(pre) = 2/(n_pre(n_pre-1)) * (n_pre*S2 - S1^2)
        sigma2_pair_pre = (2.0 / (n_pre * (n_pre - 1.0))) * A
        term_var = np.sqrt((ln / n_post) * sigma2_pair_pre) if n_post > 0 else np.inf

    return mean_pre - term_lin - term_var


def _select_c_by_kinks(Dpre, n_post, delta):
    """
    Deterministic, exact (up to ties) selection of c by scanning the breakpoints at {1} ∪ {X_i>=1}.
    """
    Dpre = np.asarray(Dpre, dtype=float)
    if Dpre.size < 2 or n_post < 2:
        # degenerate; safest fallback is c=1
        return 1.0

    # candidates are the kinks: c in {1} ∪ unique{X_i : X_i >= 1}
    X = Dpre
    X_pos = X[X >= 1.0]
    candidates = np.unique(np.concatenate(([1.0], X_pos)))  # sorted by construction if X_pos sorted

    # sort Dpre once and build prefix sums for O(1) evals at each candidate
    X_sorted = np.sort(X)
    P1 = np.concatenate(([0.0], np.cumsum(X_sorted)))
    P2 = np.concatenate(([0.0], np.cumsum(X_sorted * X_sorted)))
    n_pre = X_sorted.size

    best_c = 1.0
    best_val = -np.inf

    for c in candidates:
        val = _objective_F_on_pre(c, X_sorted, P1, P2, n_pre, n_post, delta)
        if val > best_val:
            best_val = val
            best_c = c

    return float(best_c)


def _final_hcope_lower_bound(full_sample, c, delta):
    """
    Theorem 1 lower bound on the full sample with chosen c:
      LB = mean(Y) 
           - 7 c ln(2/delta) / (3 (n - 1))
           - sqrt( (ln(2/delta)/n) * \hat{\sigma}^2_pairwise )
    where \hat{\sigma}^2_pairwise = (2/(n(n-1))) * (n*S2 - S1^2) with Y_i = min{X_i, c}.
    """
    X = np.asarray(full_sample, dtype=float)
    n = X.size
    if n <= 1:
        return float(np.mean(np.minimum(X, c)))

    Y = np.minimum(X, c)
    A, nY, S1, S2 = _pairwise_variance_terms(Y)   # nY == n
    ln = np.log(2.0 / delta)

    meanY = S1 / n
    term_lin = 7.0 * c * ln / (3.0 * max(n - 1.0, 1.0))
    sigma2_pair = (2.0 / (n * (n - 1.0))) * A if n > 1 else 0.0
    term_var = np.sqrt((ln / n) * sigma2_pair) if n > 1 else 0.0

    return float(meanY - term_lin - term_var)

# ---------------------------
# Public APIs
# ---------------------------

def high_confidence_cal_kink(sample, confidence_level=0.9, c_ratio=5, seed=None):
    """
    Backward-compatible with your original signature, but uses the kink-search for c.
    - D_pre size = max(len(sample)//c_ratio, 2), D_post = rest (need >=2).
    - Returns (best_c, lower_bound_on_full_sample)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    sample = np.asarray(sample, dtype=float)
    m = sample.size
    if m < 4:
        # tiny sample: safest is c=1 and trivial bound
        c = 1.0
        lb = _final_hcope_lower_bound(sample, c, 1 - confidence_level)
        return c, lb

    delta = 1.0 - confidence_level

    n_pre = max(m // c_ratio, 2)
    n_pre = min(n_pre, m - 2)  # leave at least 2 for post
    idx = rng.permutation(m)
    Dpre = sample[idx[:n_pre]]
    Dpost = sample[idx[n_pre:]]  # not directly used in final bound, only its size for selection

    c = _select_c_by_kinks(Dpre, n_post=Dpost.size, delta=delta)
    lb = _final_hcope_lower_bound(sample, c, delta)
    return c, lb

def batch_importance_sample(traj_df, eval_policies, behave_pi, up_b, low_b, 
                            device="cuda"):
    """
    Fully vectorized importance sampling for multi-binary action spaces
    Now handles variable-length trajectories and only uses activated actions
    Activation is determined by obs[10:14] (1=activated, 0=not activated)
    """
    with gpu_lock:
        behave_pi.to(device)
        behave_pi.eval()
        
        # Prepare data - now handling variable trajectory lengths
        all_obs = []
        all_actions = []
        all_masks = []  # NEW: Store activation masks
        trajectory_lengths = []
        return_values = traj_df["return"].to_numpy()

        for _, row in traj_df.iterrows():
            traj_len = int(row["traj_length"])  # Get actual trajectory length
            trajectory_lengths.append(traj_len)
            
            for i in range(traj_len):
                obs_key = f"obs_{i}"
                action_key = f"action_deter_{i}"
                
                # Check if column exists (defensive programming)
                if obs_key in row.index and action_key in row.index:
                    obs = row[obs_key]
                    # print(obs)
                    if isinstance(obs, (float, int, np.floating, np.integer)):
                        if np.isnan(obs):
                            break
                    all_obs.append(obs)
                    all_actions.append(row[action_key])
                    # import time
                    # time.sleep(2)
                    # Extract activation mask from obs[9:13]
                    activation_mask = obs[9:13]  # 1=activated, 0=not activated
                    all_masks.append(activation_mask)
                else:
                    break  # Stop if we run out of timesteps

        obs_tensor = torch.tensor(all_obs, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32, device=device)
        masks_tensor = torch.tensor(all_masks, dtype=torch.float32, device=device)  # [batch, 4]
        
        # Get behavioral policy Bernoulli probabilities
        with torch.no_grad():
            behave_probs = behave_pi(obs_tensor)  # [batch, action_dim]
        
        # Vectorized probability calculation - ONLY for activated actions
        action_probs = torch.where(
            actions_tensor == 1,
            behave_probs,
            1 - behave_probs
        )  # [batch, action_dim]
        
        # Apply mask: only consider activated actions (mask=1)
        # For non-activated actions (mask=0), set probability to 1 (neutral in product)
        masked_action_probs = torch.where(
            masks_tensor == 1,
            action_probs,
            torch.ones_like(action_probs)
        )
        
        base_action_probs = masked_action_probs.prod(dim=1).cpu()  # Product across activated dimensions
        
        # Process evaluation policies
        batch_results = []
        batch_size = min(8, len(eval_policies))
        
        for i in range(0, len(eval_policies), batch_size):
            batch_policies = eval_policies[i:i+batch_size]
            batch_iws = []
            
            for eval_pi in batch_policies:
                eval_pi.to(device)
                eval_pi.eval()
                
                with torch.no_grad():
                    eval_probs = eval_pi(obs_tensor)
                
                # Vectorized probability calculation for eval policy
                eval_action_probs = torch.where(
                    actions_tensor == 1,
                    eval_probs,
                    1 - eval_probs
                )  # [batch, action_dim]
                
                # Apply mask: only consider activated actions
                masked_eval_action_probs = torch.where(
                    masks_tensor == 1,
                    eval_action_probs,
                    torch.ones_like(eval_action_probs)
                )
                
                eval_action_probs_masked = masked_eval_action_probs.prod(dim=1).cpu()
                
                # Importance weights - now only based on activated actions
                importance_weights = np.array((eval_action_probs_masked / base_action_probs).tolist())
                
                # Trajectory-level weights - handle variable lengths
                traj_weights = []
                start_idx = 0
                for traj_len in trajectory_lengths:
                    end_idx = start_idx + traj_len
                    traj_iw = importance_weights[start_idx:end_idx].prod()
                    traj_weights.append(traj_iw)
                    start_idx = end_idx
                
                traj_weights = np.array(traj_weights)
                
                # Weighted returns
                norm_returns = (return_values - low_b) / (up_b - low_b)
                weighted_returns = norm_returns * traj_weights
                
                batch_iws.append(weighted_returns.tolist())
                eval_pi.cpu()
            
            batch_results.extend(batch_iws)
            
            if device == "cuda":
                torch.cuda.empty_cache()
        
        behave_pi.cpu()
        
    return batch_results


def batch_thomas_hc(behavior_pi, eval_policies, traj_list, return_list, epochs, 
                    confidence_level=0.9, num_workers=16, device="cuda",c_ratio = 5):
    """
    Optimized batch processing of Thomas HC for multiple evaluation policies
    Now handles variable-length trajectories
    """
    # Pre-process trajectory data once
    max_return, min_return = max(return_list), min(return_list)
    
    # Sample trajectory IDs once (all policies use same sample)
    sampled_ids = np.random.choice(len(traj_list), size=epochs, replace=False)
    sampled_df = traj_list[traj_list["trajectory_id"].isin(sampled_ids)]
    
    # Batch compute importance samples (GPU-intensive part)
    all_importance_samples = batch_importance_sample(
        sampled_df, eval_policies, behavior_pi, max_return, min_return, device=device
    )
    
    # Parallel computation of confidence bounds (CPU-intensive part)
    def compute_single_bound(args):
        importance_samples, seed = args
        return high_confidence_cal_kink(importance_samples, confidence_level, c_ratio,seed=seed)
    
    # Create arguments with different seeds for each worker
    args_list = [(samples, i * 1000) for i, samples in enumerate(all_importance_samples)]
    
    # Use ThreadPoolExecutor for CPU-bound confidence calculation
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(compute_single_bound, args_list),
            total=len(args_list),
            desc="Computing confidence bounds"
        ))
    
    # Extract just the lower bounds
    lower_bounds = [result[1] for result in results]
    c_s = [result[0] for result in results]
    # can add c_s in future
    return lower_bounds,c_s

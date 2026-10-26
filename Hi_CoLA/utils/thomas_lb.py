"""High-confidence off-policy evaluation (Thomas et al., 2015): per-decision
importance sampling + concentration-inequality lower bound on the normalized
return of each evaluation policy."""
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
from threading import Lock

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

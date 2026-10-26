"""
Hi-CoLA (Experiment 1) on the Flexible Route Transit (FRT) environment.

Each behavioral loop (b-loop):
  1. Collect trajectories with the current behavior policy.
  2. Sample perturbed policies within a KL ball, stratified into KL bins.
  3. Label each perturbed policy with its high-confidence Thomas lower bound.
  4. Train the Hi-CoLA surrogate network: policy parameters -> lower bound.
  5. Improve the policy by gradient ascent on the surrogate, with Thomas-LB
     early stopping and a KL trust region; the result is the next behavior policy.

The initial behavior policy is the imitation of a rule-based dispatching
heuristic (pretrained/Initial_Behave_policy.pth, see behavior_init/).
"""
import torch
import torch.multiprocessing as mp
import numpy as np
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import argparse
import time
import copy
import os
import pandas as pd

from Hi_CoLA.Env.frt_env import FRT_Env
from Hi_CoLA.policy import Behavioural_v2
from Hi_CoLA.hi_cola_net import Hi_CoLA_Net
import Hi_CoLA.hi_cola_net as hc
from Hi_CoLA.utils.simulation import collect_trajectories
from Hi_CoLA.utils.utils import kl_between_policies, perturb, get_flat_params, get_kl_bin
from Hi_CoLA.utils.thomas_lb import batch_thomas_hc

os.environ['TORCH_MULTIPROCESSING_SHARING_STRATEGY'] = 'file_system'


def collect_trajectory_chunk_global(args):
    """Global function that can be pickled for multiprocessing"""
    env, pi_state_dict, chunk_size, gamma, seed = args
        # Create new environment and policy for each process
    pi = Behavioural_v2()
    pi.load_state_dict(pi_state_dict)
    
    # Set different seed for each process  
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    return collect_trajectories(env=env, n_episodes=chunk_size, gamma=gamma, pi=pi)
        

class OptimizedTrainer:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def parallel_trajectory_collection(self, env_name, behavior_pi, traj_num, gamma, num_processes=4):
        """Step 1: collect trajectories with the current behavior policy in parallel"""
        chunk_size = traj_num // num_processes
        
        # Prepare arguments for each process
        pi_state_dict = behavior_pi.state_dict()
        args_list = [
            (env_name, pi_state_dict, chunk_size, gamma, i*1000)
            for i in range(num_processes)
        ]
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(collect_trajectory_chunk_global, args_list))
        
        # Combine results
        combined_traj = {}
        combined_returns = []
        
        for traj_list, return_list in results:
            combined_returns.extend(return_list)
            for key in traj_list.keys():
                if key not in combined_traj:
                    combined_traj[key] = traj_list[key]
                else:
                    combined_traj[key] = pd.concat([combined_traj[key], traj_list[key]], ignore_index=True)
        
        return combined_traj, combined_returns
    
    def parallel_perturbation_generation(self, behavior_pi, states, sigma, kl_threshold, target_count, bin_counts):
        """Step 2: generate perturbed policies, stratified into KL bins around the behavior policy"""
        perturbs = []
        manager = Manager()
        shared_bin_counts = manager.dict(bin_counts)
        result_queue = manager.Queue()
        
        def generate_perturb_batch(batch_size=50):
            batch_perturbs = []
            for _ in range(batch_size):
                pi_perturbed = perturb(pi=copy.deepcopy(behavior_pi), c=sigma)
                kl_score = kl_between_policies(pi_perturbed, behavior_pi, states).item()
                # print(kl_score)
                
                if kl_score <= kl_threshold:
                    kl_range = get_kl_bin(kl_score)
                    if shared_bin_counts[kl_range] < target_count:
                        batch_perturbs.append((pi_perturbed, kl_range))
            
            return batch_perturbs
        
        # Use ThreadPoolExecutor for CPU-bound perturbation generation
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            while not all(count >= target_count for count in shared_bin_counts.values()):
                # Submit batch jobs
                future = executor.submit(generate_perturb_batch)
                futures.append(future)
                
                # Process completed futures
                completed_futures = [f for f in futures if f.done()]
                for future in completed_futures:
                    batch_results = future.result()
                    for pi_perturbed, kl_range in batch_results:
                        if shared_bin_counts[kl_range] < target_count:
                            shared_bin_counts[kl_range] += 1
                            perturbs.append(pi_perturbed)
                    
                    #print(dict(shared_bin_counts))
                    futures.remove(future)
                
                # Limit concurrent futures
                if len(futures) > self.num_workers * 2:
                    # Wait for at least one to complete
                    next(as_completed(futures))
        return perturbs, dict(shared_bin_counts)
    
    def parallel_thomas_calculation(self, behavior_pi, perturbs, traj_list, return_list, epochs, confidence_level):
        """Step 3: label each policy with its high-confidence Thomas lower bound"""
        if isinstance(traj_list, dict):
            print("Converting traj_list from dict to DataFrame...")
            traj_list = pd.DataFrame(traj_list)
        thomas_lbs,_ = batch_thomas_hc(
                behavior_pi=behavior_pi, 
                eval_policies=perturbs,  # ALL policies at once
                traj_list=traj_list, 
                return_list=return_list,
                epochs=epochs, 
                confidence_level=confidence_level,
                num_workers=self.num_workers,
                device=self.device
            )
        
        return thomas_lbs
    
    def training_loop(self, behavior_pi, hi_cola, env_name, traj_num, gamma, 
                                states, b_loop, traj_list, return_list,threshold=1, max_steps=200, lr=0.0001, patience=5, save_dir="./results/exp1"):
        """
        Policy improvement: gradient ascent on the Hi-CoLA surrogate lower bound,
        with the true Thomas LB checked at every step for early stopping
        (stop after `patience` consecutive drops) and a KL trust region.

        Args:
            patience: Number of consecutive Thomas LB drops before stopping
        """
        optimizer = torch.optim.Adam(behavior_pi.parameters(), lr=lr)
        
        # Move models to GPU if available
        behavior_pi = behavior_pi.to(self.device)
        hi_cola = hi_cola.to(self.device)
        test_states = states.to(self.device)
        
        # Initialize tracking variables
        return_trace = []
        model_trace = []
        thomas_lb_trace = []
        
        start_pi = copy.deepcopy(behavior_pi)
        # Best policy tracking
        best_thomas_lb = float('-inf')
        best_pi = None
        best_step = 0
        consecutive_drops = 0
        
        for step in range(max_steps):
            
            # Forward pass
            input_vector = get_flat_params(behavior_pi).unsqueeze(0)
            # Get Hi-CoLA prediction for tracking
            hi_cola_lb = hi_cola(input_vector)
            return_trace.append(hi_cola_lb.item())
            model_trace.append(copy.deepcopy(behavior_pi).cpu())
            
            # Check actual Thomas lowerbound periodically
            # Calculate actual Thomas lowerbound
            cpu_pi = copy.deepcopy(behavior_pi).cpu()
            thomas_lb = self.parallel_thomas_calculation(
                copy.deepcopy(start_pi).cpu(),
                [cpu_pi],
                traj_list, 
                return_list,
                epochs=int(0.8 * traj_num),
                confidence_level=0.9
            )[0]
            
            thomas_lb_trace.append(thomas_lb)
            print(f"  Thomas LB: {thomas_lb:.4f} (best: {best_thomas_lb:.4f})")
            
            # Check if this is the best policy so far
            if thomas_lb > best_thomas_lb:
                best_thomas_lb = thomas_lb
                best_pi = copy.deepcopy(behavior_pi)
                best_step = step
                consecutive_drops = 0  # Reset counter when we improve
                print(f"  New best policy! Thomas LB: {best_thomas_lb:.4f}")
            else:
                # Thomas LB dropped
                consecutive_drops += 1
                print(f"  Thomas LB dropped. Consecutive drops: {consecutive_drops}/{patience}")
                
                if consecutive_drops >= patience:
                    print(f"Early stopping: Thomas LB dropped {patience} times consecutively")
                    print(f"Returning best policy from step {best_step} with Thomas LB: {best_thomas_lb:.4f}")
                    break
            # Compute loss and backprop
            optimizer.zero_grad()
            loss = -hi_cola_lb.squeeze()
            loss.backward()
            optimizer.step()
            
            # Check KL divergence constraint
            kl_div = kl_between_policies(behavior_pi, start_pi, test_states).item()
            if kl_div > 0.1:
                print(f"KL divergence {kl_div:.4f} > 0.1, stopping")
                break
        # Return the best policy found
        if best_pi is not None:
            output_pi = best_pi
            print(f"Returning best policy from step {best_step} with Thomas LB: {best_thomas_lb:.4f}")
        else:
            output_pi = copy.deepcopy(behavior_pi)
            print("No Thomas checks performed, returning current policy")
        
        # Save Thomas LB history

        np.savetxt(f"{save_dir}/thomas_lb_trace_{b_loop}.txt", 
                thomas_lb_trace, delimiter=",")
        test_num = 1000
        cpu_pi = copy.deepcopy(best_pi).cpu()
        _, return_list_step = self.parallel_trajectory_collection(
                    env_name, cpu_pi, test_num, gamma, num_processes=2
                )
        np.savetxt(f"{save_dir}/raw_reward_{b_loop}_output_policy.txt", 
                        return_list_step, delimiter=",")
        
        return output_pi.cpu(), return_trace, model_trace

def parse_args():
    parser = argparse.ArgumentParser(description="Hi-CoLA Experiment 1 on the FRT environment")
    parser.add_argument("--save_dir", type=str, default="./results/exp1")
    parser.add_argument("--init_policy", type=str, default="./pretrained/Initial_Behave_policy.pth",
                        help="initial (rule-based imitation) behavior policy")
    parser.add_argument("--b_loops", type=int, default=60, help="number of behavioral loops")
    parser.add_argument("--traj_num", type=int, default=5000, help="episodes simulated per b-loop")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--sigma", type=float, default=0.03, help="perturbation scale")
    parser.add_argument("--kl", type=float, default=0.15, help="max KL of perturbed policies (bins of 0.05)")
    parser.add_argument("--target_count", type=int, default=400, help="perturbed policies per KL bin")
    parser.add_argument("--hc_lr", type=float, default=4e-5, help="Hi-CoLA surrogate learning rate")
    parser.add_argument("--hc_epochs", type=int, default=2000, help="Hi-CoLA surrogate training epochs")
    parser.add_argument("--pi_lr", type=float, default=2e-5, help="policy improvement learning rate")
    parser.add_argument("--max_steps", type=int, default=1500, help="max policy improvement steps")
    parser.add_argument("--patience", type=int, default=20, help="Thomas LB early-stopping patience")
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))
    return parser.parse_args()


def main(args):
    # Configuration
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    gamma = args.gamma
    sigma = args.sigma
    kl = args.kl
    target_count = args.target_count
    num_workers = args.num_workers

    # Initialize trainer and models
    trainer = OptimizedTrainer(num_workers=num_workers)

    behavior_pi = Behavioural_v2()
    behavior_pi.load_state_dict(torch.load(args.init_policy, map_location='cpu'))

    env_config = {
        'n_fixed_stops': 5,
        'max_steps': 1024,
        'dispatch_interval': 12.0,
        'vehicle_capacity': 1000,
        'stochastic_travel_time': True,
        'travel_time_std': 2.0,
        'stochastic_cancellation': True,
        'flexible_stop_arrival_rate': 8,
        'flexible_cancel_prob': 0.2
    }
    env = FRT_Env(**env_config)

    # Training loop variables
    result_rec = []
    epoch_record = []
    step_rec = []
    mr_ls = []
    for b_loop in range(args.b_loops):
        traj_num = args.traj_num
        start_time = time.time()
        print(f"Starting behavioral loop {b_loop + 1}")

        # Step 1: parallel trajectory collection
        print("Collecting trajectories...")
        traj_list, return_list = trainer.parallel_trajectory_collection(
            env, behavior_pi, traj_num, gamma, num_processes=num_workers//2
        )
        traj_num = len(return_list)
        print(f"Collected {traj_num} trajectories")

        # States used to measure KL between policies
        states = torch.tensor(
            np.stack(traj_list["obs_0"].iloc[:60000].values),
            dtype=torch.float32
        )
        bin_counts = {f"{round(start,2):.2f}-{round(start+0.05,2):.2f}": 0
                     for start in np.arange(0, kl, 0.05)}

        # Step 2: KL-stratified perturbed policies
        print("Generating perturbed policies...")
        perturbs, final_bin_counts = trainer.parallel_perturbation_generation(
            behavior_pi, states, sigma, kl, target_count, bin_counts
        )
        print(f"Generated {len(perturbs)} perturbed policies")
        print("Final bin counts:", final_bin_counts)

        # Step 3: Thomas lower-bound labels
        print("Calculating Thomas lower bounds...")
        thomas_lbs = trainer.parallel_thomas_calculation(
            behavior_pi, perturbs, traj_list, return_list,
            epochs=int(0.8 * traj_num), confidence_level=0.9
        )

        # Step 4: train the Hi-CoLA surrogate
        print("Training Hi-CoLA model...")
        sample_params = []
        for param in behavior_pi.parameters():
            sample_params.append(np.array(param.detach().cpu().tolist()).flatten())
        input_dim = len(np.concatenate(sample_params))
        hi_cola, _, _, scaler, mr, label_up = hc.train(
            features=perturbs,
            labels=thomas_lbs,
            model=Hi_CoLA_Net(input_dim=input_dim),
            lr=args.hc_lr,
            epochs=args.hc_epochs,
            verbose=False,
            device=str(trainer.device),
        )
        mr_ls.append(mr)

        # Step 5: policy improvement on the surrogate
        print("Training policy...")
        output_pi, return_trace, model_trace = trainer.training_loop(behavior_pi, hi_cola, env, traj_num, gamma,
                                states, b_loop, traj_list, return_list, max_steps=args.max_steps,
                                lr=args.pi_lr, patience=args.patience, save_dir=save_dir)
        # Update records
        step_rec.append(len(return_trace))
        epoch_record.append(return_trace[-1] if return_trace else 0)
        behavior_pi = output_pi
        result_rec.extend(return_trace)

        # Save intermediate results
        np.savetxt(f"{save_dir}/Confidence_Lowerbound_Predict_{b_loop}.txt",
                return_trace, delimiter=",")
        np.savetxt(f"{save_dir}/Epoch_Record.txt",
                epoch_record, delimiter=",")
        np.savetxt(f"{save_dir}/Epoch_Length_Record.txt",
                step_rec, delimiter=",")
        torch.save(behavior_pi.state_dict(),
                    f"{save_dir}/Optimized_Behavioural_model_{b_loop}.pth")

        elapsed_time = time.time() - start_time
        print(f"Behavioral loop {b_loop + 1} completed in {elapsed_time:.2f} seconds")
        print(f"Confidence lowerbound: {epoch_record[-1]:.4f}")
        # Columns: (validation R2, training R2) of the Hi-CoLA surrogate
        np.savetxt(f"{save_dir}/Hi_cola_perform.txt", mr_ls, delimiter=",")

    # Final save
    torch.save(behavior_pi.state_dict(),
            f"{save_dir}/Optimized_Behavioural_model.pth")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(result_rec, label='Confidence Lowerbound')
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Confidence Lowerbound", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_progress.png", dpi=300)

    print("Training completed! Final results saved.")
    print(f"Total steps: {len(result_rec)}")
    print(f"Final confidence lowerbound: {result_rec[-1]:.4f}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    torch.set_num_threads(min(8, os.cpu_count()))
    main(parse_args())

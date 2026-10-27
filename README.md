# Hi-CoLA for Flexible Route Transit (FRT)

Official code for **Hi-CoLA** on the Flexible Route Transit (FRT) dispatching environment.

Hi-CoLA is a safe offline policy improvement method. It learns a surrogate network that maps a policy's parameters to its **high-confidence lower bound** on return. The behavior policy is then improved by gradient ascent on that surrogate, staying inside a KL trust region. In this experiment the initial behavior policy is an imitation of a **rule-based** dispatching heuristic.

## Algorithm

Each behavioral loop (`b_loop`) in `main.py`:

1. **Collect** trajectories with the current behavior policy (`Hi_CoLA/utils/utils.py`). Long episodes are split into non-overlapping 60-minute trajectories.
2. **Perturb**: sample policies around the behavior policy with parameter noise σ = √(c·|w|), stratified into KL bins [0, 0.05), [0.05, 0.10), [0.10, 0.15) (`Hi_CoLA/utils/utils.py`).
3. **Label** each perturbed policy with its high-confidence Thomas lower bound. This uses per-decision importance sampling plus a concentration inequality, with confidence 0.9 (`Hi_CoLA/utils/utils.py`).
4. **Fit Hi-CoLA**, a surrogate network from flattened policy parameters (6,724-d) to the lower bound (`Hi_CoLA/utils/hi_cola_net.py`).
5. **Improve** the policy by maximizing the surrogate with Adam. The true Thomas bound is checked at every step for early stopping, and the KL from the start policy is capped. The best policy becomes the next behavior policy.

## Repository layout

```
main.py              Hi-CoLA Experiment 1 entry point (argparse)
Hi_CoLA/
  Env/
    frt_env.py       FRT environment (gym): fixed route with optional detours to flexible stops
  utils/
    utils.py         KL/perturbation/KL binning, parallel trajectory collection,
                     importance sampling + high-confidence Thomas lower bound
    policy.py        Behavior policy network (35-d obs -> 4 detour probabilities)
    hi_cola_net.py   Hi-CoLA surrogate network and its training routine
    Behavioral_policy.pth   rule-based initial behavior policy used in the paper
behavior_init/       how the initial policy is produced
  rule_based_collect.py       roll out the rule-based heuristic and build imitation_dataset.npz
  train_imitation.py          fit the policy network to the heuristic (MSE imitation)
  imitation_dataset.npz       the exact dataset the shipped policy was trained on
```

## Requirements

Tested with Python 3.11, PyTorch 2.2.2 (CUDA 11.8) and gym 0.20.0, which uses the old 4-tuple `step` API.

```bash
pip install -r requirements.txt
```

You need a CUDA GPU. The Hi-CoLA surrogate has about 160M parameters, so plan for roughly 4 GB or more of GPU memory.

## Usage

Run Experiment 1 with the paper's settings:

```bash
python main.py --save_dir ./results/exp1
```

| Argument | Default | Meaning |
|---|---|---|
| `--b_loops` | 60 | number of behavioral loops |
| `--traj_num` | 5000 | episodes simulated per loop |
| `--sigma` | 0.03 | perturbation scale `c` |
| `--kl` | 0.15 | max KL of perturbed policies (bins of width 0.05) |
| `--target_count` | 400 | perturbed policies per KL bin |
| `--hc_lr`, `--hc_epochs` | 4e-5, 2000 | Hi-CoLA surrogate training |
| `--pi_lr`, `--max_steps`, `--patience` | 2e-5, 1500, 20 | policy improvement |
| `--num_workers` | min(8, #CPU) | parallel workers |
| `--init_policy` | `Hi_CoLA/utils/Behavioral_policy.pth` | initial behavior policy |

For a quick smoke test, which takes about 2 minutes:

```bash
python main.py --b_loops 1 --traj_num 40 --target_count 5 --hc_epochs 102 --max_steps 3 --patience 2 --save_dir ./results/smoke
```

### Outputs (in `--save_dir`)

| File | Contents |
|---|---|
| `raw_reward_{b}_output_policy.txt` | returns of the improved policy over 1000 fresh test trajectories |
| `thomas_lb_trace_{b}.txt` | Thomas lower bound at each policy-improvement step |
| `Confidence_Lowerbound_Predict_{b}.txt` | Hi-CoLA predicted bound at each step |
| `Hi_cola_perform.txt` | (validation R², training R²) of the surrogate for each loop |
| `Epoch_Record.txt`, `Epoch_Length_Record.txt` | final predicted bound and number of steps for each loop |
| `Optimized_Behavioural_model_{b}.pth`, `Optimized_Behavioural_model.pth` | policy after each loop, and the final policy |
| `training_progress.png` | predicted bound over all steps |

### Re-creating the initial (rule-based) behavior policy

The heuristic works per fixed stop. If the demand at its flexible stop is more than 0.5 × the demand at the next fixed stop, it detours with probability 0.7; otherwise the probability is 0.2.

```bash
python behavior_init/rule_based_collect.py   # overwrites behavior_init/imitation_dataset.npz
python behavior_init/train_imitation.py      # writes behavior_init/Behavioral_policy.pth
```

`Hi_CoLA/utils/Behavioral_policy.pth` reaches an MSE of 1.25e-6 against the rule targets in the shipped `imitation_dataset.npz`. The shipped dataset was collected with an earlier version of the environment: uniform travel-time noise and the default cancel probability of 0.3. Re-collecting with the current `Hi_CoLA/Env/frt_env.py` will therefore give a slightly different dataset.

## Implementation notes

This code is the version that produced the paper's results and is released unchanged. A few details are worth knowing if you reuse it:

- `Hi_CoLA/Env/frt_env.py:178` applies `min(sampled_time, 2.0)` to stochastic travel times, so it acts as a cap, although the comment says it is a minimum.
- No global random seed is set, so runs are not bit-for-bit reproducible.
- Trajectory IDs restart at 0 in each collection worker (`Hi_CoLA/utils/utils.py`), and `batch_thomas_hc` samples by ID (`Hi_CoLA/utils/utils.py:611`). As a result, the Thomas-bound sample can include several trajectories with the same ID.
- The importance-sampling mask uses `obs[9:13] == 1`, the vehicle-at-fixed-stop indicators (`Hi_CoLA/utils/utils.py:508`).
- The policy-improvement KL trust region is fixed at 0.1 (`main.py:226`). `get_kl_bin` assumes `--kl 0.15` (`Hi_CoLA/utils/utils.py:82`).
- Hi-CoLA's `RobustScaler` is fitted, but it is not applied when the surrogate is queried during policy improvement. The surrogate's input `LayerNorm` handles scaling. The early-stopping "best" state is saved with `state_dict().copy()` (`Hi_CoLA/utils/hi_cola_net.py:212`), which is a shallow copy.

## Citation

```bibtex
@article{hicola_frt,
  title  = {TODO},
  author = {TODO},
  year   = {TODO}
}
```

## License

MIT. See [LICENSE](LICENSE).

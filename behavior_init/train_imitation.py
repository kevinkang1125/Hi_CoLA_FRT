"""
Fit the initial behavior policy (35 -> 64 -> 64 -> 4, sigmoid; same layout as
policy.Behavioural_v2) to the rule-based heuristic by MSE imitation on
imitation_dataset.npz (produced by rule_based_collect.py).

The output is the rule-based initial behavior policy used by main.py.
By default it is written to behavior_init/Behavioral_policy.pth so the
shipped Hi_CoLA/utils/Behavioral_policy.pth is not overwritten.
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import itertools


class ImitationDataset(Dataset):
    """Dataset for imitation learning"""
    def __init__(self, observations, targets):
        self.observations = torch.FloatTensor(observations)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.targets[idx]


class Behavioural(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for obs, targets in train_loader:
        obs, targets = obs.to(device), targets.to(device)
        
        # Forward pass
        predictions = model(obs)
        loss = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(obs)
    
    return total_loss / len(train_loader.dataset)


def evaluate(model, loader, criterion, device):
    """Evaluate on a dataset"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for obs, targets in loader:
            obs, targets = obs.to(device), targets.to(device)
            predictions = model(obs)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * len(obs)
    
    return total_loss / len(loader.dataset)


def train_mlp(X_train, y_train, learning_rate=1e-3, batch_size=256, 
              n_epochs=100, device='cpu', verbose=True):
    """
    Simple training without test tracking (for quick experiments)
    
    Returns:
        trained model, training history
    """
    obs_dim = X_train.shape[1]
    action_dim = y_train.shape[1]
    
    # Create dataset
    train_dataset = ImitationDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = Behavioural(obs_dim, action_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    history = {'train_loss': []}
    
    pbar = tqdm(range(n_epochs), desc="Training", disable=not verbose)
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        
        if verbose and epoch % 10 == 0:
            pbar.set_postfix({'train_loss': f'{train_loss:.6f}'})
    
    return model, history


def train_with_test_tracking(X_train, y_train, X_test, y_test, 
                             learning_rate=1e-3, batch_size=256, 
                             n_epochs=100, device='cpu', verbose=True):
    """
    Train with both train and test loss tracking
    
    Returns:
        trained model, training history
    """
    obs_dim = X_train.shape[1]
    action_dim = y_train.shape[1]
    
    # Create datasets
    train_dataset = ImitationDataset(X_train, y_train)
    test_dataset = ImitationDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Create model
    model = Behavioural(obs_dim, action_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    history = {'train_loss': [], 'test_loss': []}
    
    pbar = tqdm(range(n_epochs), desc="Training", disable=not verbose)
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        
        if verbose and epoch % 10 == 0:
            pbar.set_postfix({
                'train_loss': f'{train_loss:.6f}',
                'test_loss': f'{test_loss:.6f}'
            })
    
    return model, history


def hyperparameter_search(X_train, y_train, X_test, y_test, device='cpu', n_epochs_grid=(100, 500)):
    """
    Simple grid search - train on train set, evaluate on test set
    
    Returns:
        DataFrame with results, best config
    """
    # Hyperparameter grid
    hp_grid = {
        'learning_rate': [1e-3],
        'batch_size': [128],
        'n_epochs': list(n_epochs_grid)
    }
    
    # Generate all combinations
    keys = list(hp_grid.keys())
    values = list(hp_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"Testing {len(combinations)} hyperparameter combinations...")
    print("=" * 70)
    
    results = []
    
    for combo in tqdm(combinations, desc="Hyperparameter search"):
        config = dict(zip(keys, combo))
        
        try:
            # Train model
            model, history = train_with_test_tracking(
                X_train, y_train, X_test, y_test,
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                n_epochs=config['n_epochs'],
                device=device,
                verbose=False
            )
            
            result = {
                **config,
                'final_test_loss': history['test_loss'][-1],
                'final_train_loss': history['train_loss'][-1],
            }
            results.append(result)
        
        except Exception as e:
            print(f"Failed with config {config}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('final_test_loss')
    
    return results_df, dict(results_df.iloc[0])


def plot_training_curves(history, save_path='training_curves.png'):
    """Plot training and test loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['test_loss'], label='Test Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved training curves to {save_path}")


def evaluate_policy_on_env(model, env_config, n_episodes=100, device='cpu'):
    """
    Evaluate trained policy in the environment
    
    Returns:
        dict with performance metrics
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Hi_CoLA.Env.frt_env import FRT_Env
    
    model.eval()
    returns = []
    passengers_per_hour = []
    
    for seed in tqdm(range(n_episodes), desc="Evaluating policy"):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        env = FRT_Env(**env_config)
        obs = env.reset()
        done = False
        episode_return = 0.0
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                action_probs = model(obs_tensor).cpu().numpy()
            
            # Sample action
            action = np.random.binomial(1, action_probs)
            obs, reward, done, info = env.step(action)
            episode_return += reward
        
        returns.append(episode_return)
        passengers_per_hour.append(info['passengers_per_hour'])
    
    results = {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_passengers_per_hour': np.mean(passengers_per_hour),
        'std_passengers_per_hour': np.std(passengers_per_hour)
    }
    
    return results


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Imitate the rule-based policy")
    parser.add_argument("--data", type=str, default=os.path.join(here, "imitation_dataset.npz"))
    parser.add_argument("--out", type=str, default=os.path.join(here, "Behavioral_policy.pth"))
    parser.add_argument("--epochs", type=str, default="100,500", help="comma-separated epoch grid")
    parser.add_argument("--eval_episodes", type=int, default=100)
    args = parser.parse_args()
    os.chdir(here)

    print("=" * 70)
    print("MLP IMITATION LEARNING TRAINING")
    print("=" * 70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading imitation dataset...")
    data = np.load(args.data)
    X = data['observations']
    y = data['targets']
    print(f"✓ Loaded {len(X)} samples")
    print(f"  Observation dim: {X.shape[1]}")
    print(f"  Action dim: {y.shape[1]}")
    
    # Split data (90% train, 10% test)
    print("\nSplitting data (90% train, 10% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Hyperparameter search
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH")
    print("=" * 70)
    results_df, best_config = hyperparameter_search(
        X_train, y_train, X_test, y_test, device=device,
        n_epochs_grid=[int(e) for e in args.epochs.split(',')]
    )
    
    # Save results
    results_df.to_csv('hyperparameter_results.csv', index=False)
    print("\n✓ Saved hyperparameter results to hyperparameter_results.csv")
    
    # Show top configurations
    print("\n" + "=" * 70)
    print("TOP HYPERPARAMETER CONFIGURATIONS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Best config
    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS")
    print("=" * 70)
    for key, value in best_config.items():
        if key not in ['final_test_loss', 'final_train_loss']:
            print(f"  {key}: {value}")
    print(f"  Final train loss: {best_config['final_train_loss']:.6f}")
    print(f"  Final test loss: {best_config['final_test_loss']:.6f}")
    
    # Train final model with best config
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)
    final_model, history = train_with_test_tracking(
        X_train, y_train, X_test, y_test,
        learning_rate=best_config['learning_rate'],
        batch_size=int(best_config['batch_size']),
        n_epochs=int(best_config['n_epochs']),
        device=device,
        verbose=True
    )
    
    plot_training_curves(history)
    
    # Save final model
    torch.save(final_model.state_dict(), args.out)
    print(f"\n✓ Saved final model to {args.out}")
    
    # Evaluate in environment
    print("\n" + "=" * 70)
    print("ENVIRONMENT EVALUATION")
    print("=" * 70)
    env_config = {
        'n_fixed_stops': 5,
        'max_steps': 1024,
        'dispatch_interval': 12.0,
        'vehicle_capacity': 1000,
        'stochastic_travel_time': True,
        'travel_time_std': 2.0,
        'stochastic_cancellation': True
    }
    
    env_results = evaluate_policy_on_env(final_model, env_config, n_episodes=args.eval_episodes, device=device)
    print(f"\nLearned Policy Performance:")
    print(f"  Mean return: {env_results['mean_return']:.2f} ± {env_results['std_return']:.2f}")
    print(f"  Mean passengers/hour: {env_results['mean_passengers_per_hour']:.1f} ± {env_results['std_passengers_per_hour']:.1f}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
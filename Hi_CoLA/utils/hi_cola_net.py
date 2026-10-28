import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

def load_model_features_and_labels(features, labels, train_ratio=0.7, device='cuda'):
    """
    FIXED: Now returns tensors on specified device (GPU)
    """
    policy_params = [] 
    for model_path in features:
        model = model_path
        flat_params = []
        for param in model.parameters():
            flat_params.append(np.array(param.detach().cpu().tolist()).flatten())
        feature_vector = np.concatenate(flat_params)
        policy_params.append(feature_vector)
    
    features = np.array(policy_params)
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Data shape: {X.shape[0]} policies, {X.shape[1]} parameters each")
    print(f"Thomas LB range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Input normalization
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    # Outlier removal
    if len(y) > 20:
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
        X, y = X[mask], y[mask]
        print(f"After outlier removal: {len(X)} policies")
    
    # FIXED: Move tensors to GPU immediately
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    batch_size = min(32, max(8, len(train_set) // 10))
    
    # FIXED: pin_memory for faster GPU transfer, but data already on GPU
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    return train_loader, val_loader, scaler, y.max()


class Hi_CoLA_Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8192),
            nn.LayerNorm(8192),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(8192, 8192),
            nn.LayerNorm(8192),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(8192, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.05),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
            
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            
            nn.Linear(128, 64),
            nn.GELU(),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.input_norm(x)
        return self.net(x)


def train(features, labels, model, lr=1e-4, weight_decay=1e-6, epochs=3000, 
          logit=False, train_ratio=0.7, use_improvements=True, device=None):
    """
    FIXED: Now trains on GPU
    
    Args:
        device: 'cuda', 'cpu', or None (auto-detect)
    """
    # FIXED: Auto-detect or use specified device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Training on device: {device}")
    
    # FIXED: Move model to GPU
    model = model.to(device)
    
    # FIXED: Load data on GPU
    train_loader, val_loader, scaler, label_up = load_model_features_and_labels(
        features, labels, train_ratio, device=device
    )
    
    print(f"Training Hi-CoLA with 8192 first layer: lr={lr}, epochs={epochs}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    warmup_epochs = 100
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    loss_fn = nn.SmoothL1Loss()
        
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 0
    best_model_state = None
    
    print("Starting training...")
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_loss = 0
        
        for xb, yb in train_loader:
            # Data already on GPU from loader
            pred = model(xb)
            loss = loss_fn(pred, yb)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                # Data already on GPU
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Clone the tensors; state_dict() returns references to the live weights
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        
        max_patience = 200
        if patience > max_patience:
            print(f"Early stopping at epoch {epoch+1} (patience exceeded)")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
    
    model.eval()
    
    return model, train_losses, val_losses, scaler, label_up


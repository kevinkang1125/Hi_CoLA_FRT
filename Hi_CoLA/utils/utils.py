import torch
import numpy as np
from Hi_CoLA.policy import Behavioural_v2


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


def perturb(pi, c=0.3):
    """
    Perturb the policy parameters with Gaussian noise where sigma is proportional to each weight's magnitude.

    Args:
        pi (Policy): the original policy network
        c (float): proportional constant to scale sigma = c * |weight|

    Returns:
        pi_perturbed (Policy): the perturbed policy
    """
    pi_perturbed = Behavioural_v2()
    pi_perturbed.load_state_dict(pi.state_dict())

    with torch.no_grad():
        for param in pi_perturbed.parameters():
            if param.requires_grad:
                sigma = torch.sqrt(c * param.abs() + 1e-8)
                noise = torch.randn_like(param) * sigma
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


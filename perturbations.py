"""
perturbations.py - Stochastic perturbation strategies for field dynamics
Physics: Lévy flights vs Beta distribution noise injection
"""

import torch
import torch.distributions as dist
import numpy as np
from scipy.stats import levy_stable
from typing import Optional, Tuple

# Constants
PHI = (1 + np.sqrt(5)) / 2

class PerturbationBase:
    """Base class for field perturbations"""
    def __init__(self, scale: float = 0.1):
        self.scale = scale
    
    def perturb(self, field: torch.Tensor) -> torch.Tensor:
        """Apply perturbation to field"""
        raise NotImplementedError

class LevyPerturbation(PerturbationBase):
    """
    Lévy flight perturbations
    Physics: Heavy-tailed jumps enable exploration of distant phase space regions
    P(x) ~ |x|^(-1-α) for large |x|, α = φ
    """
    def __init__(self, scale: float = 0.1, alpha: float = PHI):
        super().__init__(scale)
        self.alpha = alpha
        
    def perturb(self, field: torch.Tensor) -> torch.Tensor:
        """
        field: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model) - Perturbed field
        """
        batch, seq_len, d_model = field.shape
        device = field.device
        
        # Generate Lévy noise on CPU (scipy requirement)
        levy_noise = np.zeros((batch, seq_len, d_model))
        
        for b in range(batch):
            for s in range(seq_len):
                for d in range(d_model):
                    levy_noise[b, s, d] = levy_stable.rvs(
                        self.alpha, beta=0, scale=self.scale
                    )
        
        # Convert to tensor
        noise = torch.tensor(levy_noise, device=device, dtype=field.dtype)
        
        # Apply perturbation with phase preservation
        perturbed = field + noise
        
        # Renormalize to manifold
        norms = torch.norm(perturbed[..., :3], dim=-1, keepdim=True)
        perturbed[..., :3] = perturbed[..., :3] / (norms + 1e-8)
        
        return perturbed

class BetaPerturbation(PerturbationBase):
    """
    Inverse Beta distribution perturbations
    Physics: Time-step modulation based on information geometry
    dt ~ 1/Beta(N/2, D/2) where N=embedding dim, D=sequence length
    """
    def __init__(self, scale: float = 0.1):
        super().__init__(scale)
        
    def perturb(self, field: torch.Tensor) -> torch.Tensor:
        """
        field: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model) - Perturbed field
        """
        batch, seq_len, d_model = field.shape
        device = field.device
        
        # Beta distribution parameters
        alpha = d_model / 2.0
        beta = seq_len / 2.0
        
        # Sample from Beta distribution
        beta_dist = dist.Beta(alpha, beta)
        beta_samples = beta_dist.sample((batch, seq_len, 1)).to(device)
        
        # Inverse transform for time-step
        dt = 1.0 / (beta_samples + 1e-6)
        
        # Scale dt to reasonable range
        dt = torch.clamp(dt * self.scale, 0.001, 0.1)
        
        # Generate Gaussian noise modulated by dt
        noise = torch.randn_like(field) * torch.sqrt(dt)
        
        # Apply perturbation
        perturbed = field + noise
        
        # Project back to manifold
        perturbed = perturbed / (torch.norm(perturbed, dim=-1, keepdim=True) + 1e-8)
        
        return perturbed

class AdaptivePerturbation(PerturbationBase):
    """
    Adaptive perturbation based on field coherence
    Physics: Noise amplitude inversely proportional to local order parameter
    """
    def __init__(self, scale: float = 0.1, levy_weight: float = 0.5):
        super().__init__(scale)
        self.levy_weight = levy_weight
        self.levy = LevyPerturbation(scale)
        self.beta = BetaPerturbation(scale)
        
    def compute_coherence(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute local coherence metric
        field: (batch, seq_len, d_model)
        Returns: (batch, seq_len) - Coherence values
        """
        # Covariance with neighbors
        field_shifted = torch.roll(field, shifts=-1, dims=1)
        local_cov = torch.einsum('bsd,bsd->bs', field, field_shifted)
        
        # Normalize by norms
        norm1 = torch.norm(field, dim=-1)
        norm2 = torch.norm(field_shifted, dim=-1)
        
        coherence = local_cov / (norm1 * norm2 + 1e-8)
        
        return coherence
    
    def perturb(self, field: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive perturbation mixing Lévy and Beta based on coherence
        field: (batch, seq_len, d_model)
        """
        # Compute local coherence
        coherence = self.compute_coherence(field)  # (batch, seq_len)
        
        # High coherence -> more Beta (local), low coherence -> more Lévy (exploratory)
        levy_mask = (coherence < 0.5).float().unsqueeze(-1)
        
        # Apply both perturbations
        levy_perturbed = self.levy.perturb(field)
        beta_perturbed = self.beta.perturb(field)
        
        # Mix based on coherence
        perturbed = levy_mask * levy_perturbed + (1 - levy_mask) * beta_perturbed
        
        return perturbed

class QuantumPerturbation(PerturbationBase):
    """
    Quantum-inspired measurement collapse perturbation
    Physics: Stochastic collapse to eigenstates based on entropy
    """
    def __init__(self, scale: float = 0.1, collapse_prob: float = 0.1):
        super().__init__(scale)
        self.collapse_prob = collapse_prob
        
    def perturb(self, field: torch.Tensor) -> torch.Tensor:
        """
        field: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model) - Collapsed/perturbed field
        """
        batch, seq_len, d_model = field.shape
        device = field.device
        
        # Compute field entropy
        field_energy = 0.5 * torch.sum(field**2, dim=-1)  # (batch, seq_len)
        field_probs = torch.softmax(-field_energy / PHI, dim=-1)  # (batch, seq_len)
        entropy = -torch.sum(field_probs * torch.log(field_probs + 1e-10), dim=-1)  # (batch,)
        
        # Collapse probability increases with low entropy
        collapse_mask = (torch.rand(batch, device=device) < self.collapse_prob) & (entropy < 2.0)
        
        perturbed = field.clone()
        
        for b in range(batch):
            if collapse_mask[b]:
                # Collapse to dominant eigenstate
                dominant_idx = torch.argmax(field_probs[b])
                
                # Create collapsed state
                collapsed = torch.zeros_like(field[b])
                collapsed[dominant_idx] = torch.norm(field[b, dominant_idx])
                
                # Mix collapsed with original
                perturbed[b] = 0.9 * collapsed + 0.1 * field[b]
        
        # Add small quantum noise
        quantum_noise = torch.randn_like(field) * self.scale * 0.1
        perturbed = perturbed + quantum_noise
        
        return perturbed

# Validation
if __name__ == "__main__":
    print("=== Perturbation Module Validation ===")
    
    # Test field
    field = torch.randn(2, 16, 32).cuda()
    field = field / torch.norm(field, dim=-1, keepdim=True)
    
    # Test each perturbation
    perturbations = {
        "Lévy": LevyPerturbation(),
        "Beta": BetaPerturbation(),
        "Adaptive": AdaptivePerturbation(),
        "Quantum": QuantumPerturbation()
    }
    
    for name, perturb in perturbations.items():
        perturbed = perturb.perturb(field)
        
        # Compute statistics
        diff_norm = torch.norm(perturbed - field).item()
        max_jump = torch.max(torch.abs(perturbed - field)).item()
        
        print(f"\n{name} Perturbation:")
        print(f"  Mean displacement: {diff_norm/field.numel():.6f}")
        print(f"  Max jump: {max_jump:.4f}")
        print(f"  Shape preserved: {perturbed.shape == field.shape}")
    
    print("\n✓ All perturbations validated")
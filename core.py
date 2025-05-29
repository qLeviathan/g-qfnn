"""
core.py - Field-Theoretic Language Model Core Architecture
Physics: Tokens exist on golden spiral manifold with gravitational collapse dynamics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import math

# Universal constants
PHI = (1 + np.sqrt(5)) / 2
GAP = 1 - 1/PHI
TAU = 2 * np.pi

class FieldConfig:
    """Configuration for field-theoretic LLM"""
    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab
        d_model: int = 768,       # Hidden dimension
        n_layers: int = 12,       # Number of field layers
        max_seq_len: int = 1024,  # Maximum sequence length
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16,  # For 4090 efficiency
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        
        # Field physics parameters
        self.alpha = d_model / 2.0  # Diffusion coefficient
        self.collapse_threshold = 0.91  # Coherence threshold
        self.levy_alpha = PHI  # Lévy flight parameter
        self.dt = 0.01  # Integration timestep

class GoldenEmbedding(nn.Module):
    """
    Golden spiral cylindrical embeddings
    Shape: vocab_size -> (d_model,)
    Physics: Maps discrete tokens to continuous manifold positions
    """
    def __init__(self, config: FieldConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        
        # Pre-compute golden spiral positions
        self.register_buffer('spiral_coords', self._create_spiral_coords())
        
        # Learnable radial scaling
        self.radial_scale = nn.Parameter(torch.ones(1))
        
    def _create_spiral_coords(self) -> torch.Tensor:
        """
        Create golden spiral coordinates
        Returns: (vocab_size, 3) - cylindrical coordinates (r*cos(θ), r*sin(θ), z)
        """
        coords = torch.zeros(self.vocab_size, 3)
        
        for i in range(self.vocab_size):
            # Frequency rank determines radius
            freq_rank = i / self.vocab_size
            r = GAP + (1 - GAP) * freq_rank
            
            # Golden angle
            theta = TAU * ((i * PHI) % 1.0)
            
            coords[i] = torch.tensor([
                r * math.cos(theta),
                r * math.sin(theta),
                r  # z = r creates cone
            ])
            
        return coords
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_len) - Token indices
        Returns: (batch, seq_len, d_model) - Field embeddings
        """
        # Get base coordinates
        coords = self.spiral_coords[token_ids]  # (batch, seq_len, 3)
        
        # Scale by learnable parameter
        coords = coords * self.radial_scale
        
        # Project to d_model dimensions
        # First 3 dims are spatial, rest are zeros (to be filled by field dynamics)
        embeddings = torch.zeros(
            *token_ids.shape, self.d_model, 
            device=token_ids.device, dtype=self.config.dtype
        )
        embeddings[..., :3] = coords
        
        return embeddings

class LogPhaseEmbedding(nn.Module):
    """
    Log-phase embeddings for enhanced Hebbian learning
    Shape: vocab_size -> (d_model,)
    Physics: Logarithmic transformation amplifies phase differences
    """
    def __init__(self, config: FieldConfig):
        super().__init__()
        self.config = config
        self.golden_embed = GoldenEmbedding(config)
        self.eps = 1e-6
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_len)
        Returns: (batch, seq_len, d_model) - Log-phase embeddings
        """
        # Get golden embeddings
        golden = self.golden_embed(token_ids)  # (batch, seq_len, d_model)
        
        # Extract cylindrical components
        x, y, z = golden[..., 0], golden[..., 1], golden[..., 2]
        
        # Convert to log-phase
        r = torch.sqrt(x**2 + y**2 + self.eps)
        theta = torch.atan2(y, x)
        
        # Log transformation
        log_r = torch.log(r + self.eps)
        log_x = log_r + torch.log(torch.abs(torch.cos(theta)) + self.eps) * torch.sign(torch.cos(theta))
        log_y = log_r + torch.log(torch.abs(torch.sin(theta)) + self.eps) * torch.sign(torch.sin(theta))
        
        # Reconstruct embeddings
        log_embeddings = golden.clone()
        log_embeddings[..., 0] = log_x
        log_embeddings[..., 1] = log_y
        log_embeddings[..., 2] = torch.log(z + self.eps)
        
        return log_embeddings

class FieldEvolution(nn.Module):
    """
    Hamiltonian field evolution (replaces attention)
    Physics: ∂ψ/∂t = -iĤψ with gravitational interactions
    """
    def __init__(self, config: FieldConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Interaction strength
        self.g = nn.Parameter(torch.tensor(1.0 / PHI))
        
    def compute_gravitational_matrix(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise gravitational interactions
        field: (batch, seq_len, d_model)
        Returns: (batch, seq_len, seq_len) - Interaction matrix
        """
        batch, seq_len, _ = field.shape
        
        # Pairwise distances using einsum
        # dist_sq[b,i,j] = ||field[b,i] - field[b,j]||²
        dist_sq = torch.einsum('bid,bid->bi', field, field).unsqueeze(2) + \
                  torch.einsum('bjd,bjd->bj', field, field).unsqueeze(1) - \
                  2 * torch.einsum('bid,bjd->bij', field, field)
        
        # Gravitational potential: -g/√(r² + ε)
        dist = torch.sqrt(dist_sq + 1e-8)
        G = -self.g / dist
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=field.device))
        G = G * causal_mask
        
        return G
    
    def forward(self, field: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Evolve field according to Hamiltonian dynamics
        field: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model) - Evolved field
        """
        dt = self.config.dt
        
        for _ in range(steps):
            # Ensure consistent dtype
            field = field.to(self.g.dtype)
            
            # Compute interactions
            G = self.compute_gravitational_matrix(field)  # (batch, seq_len, seq_len)
            
            # Gravitational force on each token
            # F[b,i,d] = Σ_j G[b,i,j] * field[b,j,d]
            # Cast to same dtype as field
            force = torch.einsum('bij,bjd->bid', G.to(field.dtype), field)
            
            # Heun-Euler integration
            field_half = field + 0.5 * dt * force
            
            # Recompute force at half-step
            G_half = self.compute_gravitational_matrix(field_half)
            force_half = torch.einsum('bij,bjd->bid', G_half.to(field.dtype), field_half)
            
            # Full step
            field = field + dt * force_half
            
            # Project back to manifold (normalize)
            field = field / (torch.norm(field, dim=-1, keepdim=True) + 1e-8)
            
        return field

class CrystalMemory(nn.Module):
    """
    Crystalline memory formation through Hebbian dynamics
    Physics: Weight matrix crystallizes from field correlations
    """
    def __init__(self, config: FieldConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Crystal matrix (no gradients - formed through Hebbian updates)
        self.register_buffer('W_crystal', torch.zeros(
            config.n_layers, config.d_model, config.d_model,
            dtype=config.dtype
        ))
        
        # Learning rate modulated by golden ratio
        self.eta = 1.0 / PHI
        
    def hebbian_update(self, pre: torch.Tensor, post: torch.Tensor, layer_idx: int):
        """
        Update crystal matrix via Hebbian rule
        pre: (batch, seq_len, d_model) - Pre-synaptic activity
        post: (batch, seq_len, d_model) - Post-synaptic activity
        """
        # Compute correlation: ΔW = η * ⟨post ⊗ pre⟩
        # Average over batch and sequence
        correlation = torch.einsum('bsd,bse->de', post, pre) / (pre.shape[0] * pre.shape[1])
        
        # Update crystal with decay
        self.W_crystal[layer_idx] = (1 - self.eta/10) * self.W_crystal[layer_idx] + \
                                    self.eta * correlation
    
    def forward(self, field: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Apply crystallized transformation
        field: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        # Apply crystal matrix with dtype casting
        return torch.einsum('de,bse->bsd', self.W_crystal[layer_idx].to(field.dtype), field)

# Validation
if __name__ == "__main__":
    print("=== Core Field Architecture Validation ===")
    
    # Test with float32 instead of float16 for validation
    config = FieldConfig(vocab_size=1000, d_model=64, n_layers=4, dtype=torch.float32)
    
    # Test embeddings
    golden = GoldenEmbedding(config).to(config.device)
    log_phase = LogPhaseEmbedding(config).to(config.device)
    
    tokens = torch.randint(0, 1000, (2, 32), device=config.device)
    
    g_embed = golden(tokens)
    l_embed = log_phase(tokens)
    
    print(f"Golden embedding shape: {g_embed.shape}")
    print(f"Log-phase embedding shape: {l_embed.shape}")
    print(f"Embedding norm ratio: {torch.norm(l_embed)/torch.norm(g_embed):.2f}")
    
    # Test field evolution
    field_evo = FieldEvolution(config).to(config.device)
    evolved = field_evo(g_embed.float(), steps=5)  # Ensure float32
    print(f"Field evolution preserves shape: {evolved.shape}")
    
    # Test crystal memory
    crystal = CrystalMemory(config).to(config.device)
    crystal.hebbian_update(g_embed, evolved, layer_idx=0)
    transformed = crystal(evolved, layer_idx=0)
    print(f"Crystal memory output shape: {transformed.shape}")
    
    print("\n✓ Core architecture validated")
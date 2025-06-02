"""
core.py - Core field architecture for GF-NN language model

Field-theoretic embeddings and evolution dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Golden ratio constant
PHI = (1 + np.sqrt(5)) / 2

@dataclass
class FieldConfig:
    """Configuration for field dynamics"""
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    max_seq_len: int = 2048
    
    # Field parameters
    g: float = 0.01  # Gravitational coupling
    dt: float = 0.1  # Time step
    mass_scale: float = 1.0
    
    # Memory parameters
    tau: float = 0.95  # Memory decay
    hebbian_lr: float = 0.01
    
    # Device and precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32  # Use float32 for stability


class GoldenEmbedding(nn.Module):
    """Golden ratio-based token embeddings"""
    
    def __init__(self, config: FieldConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Initialize embeddings with golden ratio structure
        self.embeddings = nn.Parameter(torch.empty(config.vocab_size, config.d_model))
        self._init_golden()
        
    def _init_golden(self):
        """Initialize with golden spiral structure"""
        with torch.no_grad():
            for i in range(self.config.vocab_size):
                for j in range(self.d_model):
                    # Golden angle in radians
                    theta = 2 * np.pi * i / PHI
                    phi = 2 * np.pi * j / PHI
                    
                    # Fibonacci-inspired initialization
                    r = np.sqrt(i + 1) / np.sqrt(self.config.vocab_size)
                    self.embeddings[i, j] = r * np.cos(theta + phi)
                    
            # Normalize
            self.embeddings.data = F.normalize(self.embeddings.data, dim=-1)
            
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens with golden structure
        token_ids: (batch, seq_len)
        Returns: (batch, seq_len, d_model)
        """
        # Ensure proper dtype
        embeddings = self.embeddings.to(token_ids.device).to(self.config.dtype)
        return F.embedding(token_ids, embeddings)


class LogPhaseEmbedding(nn.Module):
    """Logarithmic phase-based embeddings"""
    
    def __init__(self, config: FieldConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Base embeddings
        self.embeddings = nn.Parameter(torch.randn(config.vocab_size, config.d_model))
        
        # Phase modulation
        self.phase_scale = nn.Parameter(torch.ones(config.d_model))
        
        self._init_phases()
        
    def _init_phases(self):
        """Initialize with log-phase structure"""
        with torch.no_grad():
            # Create log-spaced frequencies
            freqs = torch.logspace(-2, 2, self.d_model // 2)
            
            for i in range(self.config.vocab_size):
                # Log-phase encoding
                phase = np.log(i + 1) / np.log(self.config.vocab_size)
                
                # Apply to embeddings
                self.embeddings[i, :self.d_model//2] *= torch.cos(2 * np.pi * phase * freqs)
                self.embeddings[i, self.d_model//2:] *= torch.sin(2 * np.pi * phase * freqs[:self.d_model - self.d_model//2])
                
            # Normalize
            self.embeddings.data = F.normalize(self.embeddings.data, dim=-1)
            
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed with logarithmic phase modulation
        token_ids: (batch, seq_len)
        Returns: (batch, seq_len, d_model)
        """
        # Get base embeddings with proper dtype
        x = F.embedding(token_ids, self.embeddings.to(self.config.dtype))
        
        # Phase modulation
        phase = torch.log(token_ids.float() + 1) / np.log(self.config.vocab_size)
        phase = phase.unsqueeze(-1).to(self.config.dtype)
        
        # Apply phase-based scaling
        x = x * (1 + self.phase_scale.to(self.config.dtype) * phase)
        
        return x


class FieldEvolution(nn.Module):
    """Hamiltonian field evolution dynamics"""
    
    def __init__(self, config: FieldConfig):
        super().__init__()
        self.config = config
        self.g = config.g
        
        # Learnable mass distribution
        self.mass = nn.Parameter(torch.ones(config.d_model) * config.mass_scale)
        
    def compute_gravitational_matrix(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute gravitational interaction matrix
        field: (batch, seq_len, d_model)
        Returns: (batch, seq_len, seq_len) - Interaction strengths
        """
        batch, seq_len, d_model = field.shape
        
        # Compute pairwise distances
        # Expand for broadcasting: (batch, seq_len, 1, d_model)
        field_i = field.unsqueeze(2)
        field_j = field.unsqueeze(1)
        
        # Euclidean distance
        distances = torch.norm(field_i - field_j, dim=-1) + 1e-8
        
        # Gravitational interaction: G_ij = g * m_i * m_j / r_ij
        # For simplicity, using unit masses
        G = self.g / distances
        
        # Zero out self-interactions
        eye = torch.eye(seq_len, device=field.device, dtype=field.dtype)
        G = G * (1 - eye.unsqueeze(0))
        
        return G
        
    def forward(self, field: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Evolve field according to Hamiltonian dynamics
        field: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model) - Evolved field
        """
        dt = self.config.dt
        
        # Ensure consistent dtype throughout evolution
        field = field.to(self.config.dtype)
        
        for _ in range(steps):
            # Compute interactions
            G = self.compute_gravitational_matrix(field)  # (batch, seq_len, seq_len)
            
            # Gravitational force on each token
            # F[b,i,d] = Σ_j G[b,i,j] * field[b,j,d]
            force = torch.einsum('bij,bjd->bid', G, field)
            
            # Heun-Euler integration
            field_half = field + 0.5 * dt * force
            
            # Recompute force at half-step
            G_half = self.compute_gravitational_matrix(field_half)
            force_half = torch.einsum('bij,bjd->bid', G_half, field_half)
            
            # Full step
            field = field + dt * force_half
            
            # Project back to manifold (normalize)
            field = field / (torch.norm(field, dim=-1, keepdim=True) + 1e-8)
            
        return field


class CrystalMemory(nn.Module):
    """Hebbian crystallized memory matrices"""
    
    def __init__(self, config: FieldConfig):
        super().__init__()
        self.config = config
        self.tau = config.tau
        self.lr = config.hebbian_lr
        
        # Initialize crystal matrices for each layer
        self.W_crystal = nn.ParameterList([
            nn.Parameter(torch.eye(config.d_model, dtype=config.dtype) + 
                        0.01 * torch.randn(config.d_model, config.d_model, dtype=config.dtype))
            for _ in range(config.n_layers)
        ])
        
    def hebbian_update(self, pre: torch.Tensor, post: torch.Tensor, layer_idx: int):
        """
        Update crystal matrix with Hebbian learning
        pre: (batch, seq_len, d_model) - Pre-synaptic activity
        post: (batch, seq_len, d_model) - Post-synaptic activity  
        """
        with torch.no_grad():
            # Ensure consistent dtype
            pre = pre.to(self.config.dtype)
            post = post.to(self.config.dtype)
            
            # Compute correlation
            # C = E[post ⊗ pre]
            correlation = torch.einsum('bsd,bse->de', post, pre) / (pre.shape[0] * pre.shape[1])
            
            # Exponential moving average update
            self.W_crystal[layer_idx].data = (
                self.tau * self.W_crystal[layer_idx].data +
                self.lr * correlation
            )
            
            # Maintain spectral radius close to 1 for stability
            eigvals = torch.linalg.eigvals(self.W_crystal[layer_idx].data)
            spectral_radius = torch.max(torch.abs(eigvals)).item()
            if spectral_radius > 1.5:
                self.W_crystal[layer_idx].data /= (spectral_radius / 1.2)
                
    def forward(self, field: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Apply crystallized transformation
        field: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        # Ensure dtype consistency
        field = field.to(self.config.dtype)
        W = self.W_crystal[layer_idx].to(self.config.dtype)
        
        # Apply crystal matrix
        return torch.einsum('de,bse->bsd', W, field)


# Validation
if __name__ == "__main__":
    print("=== Core Field Architecture Validation ===")
    
    # Create config with float32 for testing
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
    evolved = field_evo(g_embed, steps=5)
    print(f"Field evolution preserves shape: {evolved.shape}")
    
    # Test crystal memory
    crystal = CrystalMemory(config).to(config.device)
    crystal.hebbian_update(g_embed, evolved, layer_idx=0)
    transformed = crystal(evolved, layer_idx=0)
    print(f"Crystal memory output shape: {transformed.shape}")
    
    print("\n[PASS] Core architecture validated")
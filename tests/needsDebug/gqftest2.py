"""
Pure Physics Language Model (PPLM)
==================================

A revolutionary approach where language modeling IS quantum field evolution.
No backpropagation. No gradients. Just physics.

The model learns through:
1. Schrödinger evolution (natural energy minimization)
2. Vortex self-organization (coherence emergence)  
3. Lévy superdiffusion (exploration without SGD)
4. Frame-dragging (gravitational memory)
5. Born rule measurement (token generation)

Author: Marc Castillo's Quantum Framework
Date: 2024
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_fn
from scipy.stats import levy_stable
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import math  # Add this import
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import os
import warnings
import requests
from tqdm import tqdm
import time

# Import the fixes
from gqftest import apply_fixes_to_model

# Apply to your model
wiki_model = apply_fixes_to_model(wiki_model)

warnings.filterwarnings('ignore')

# ==============================================================================
# PHYSICAL CONSTANTS
# ==============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618034
GAP = 1 - 1/PHI             # Golden gap ≈ 0.382
HBAR = 1.0                  # Reduced Planck constant (normalized)
C = 1.0                     # Speed of light (normalized)  
K_B = 1.0                   # Boltzmann constant (normalized)
G = 1.0                     # Gravitational constant (normalized)
EPS = 1e-8                  # Numerical stability epsilon

# Parallel processing
NUM_CORES = mp.cpu_count()
print(f"🔧 Detected {NUM_CORES} CPU cores for parallel processing")

# ==============================================================================
# QUANTUM FIELD STATE
# ==============================================================================

@dataclass
class QuantumFieldState:
    """
    Complete quantum field state for language modeling.
    
    Physical quantities:
    - psi: Complex wave function ψ(x,t) ∈ ℂ^N
    - momentum: Conjugate momentum π(x,t) = ∂L/∂(∂ψ/∂t)
    - vorticity: Vortex field ω(x,t) = ∇ × v
    - metric: Spacetime metric g_μν(x,t)
    - energy: Total field energy E = ∫ℋ d³x
    """
    # Wave function - BEFORE: random init, AFTER: evolved to ground state via i∂ψ/∂t = Ĥψ
    psi: torch.Tensor  # Shape: [vocab_size, field_dim]
    
    # Conjugate momentum - BEFORE: zero, AFTER: ∂ψ/∂t from dynamics
    momentum: torch.Tensor  # Shape: [vocab_size, field_dim]
    
    # Vorticity field - BEFORE: zero, AFTER: ∇×v from token flow
    vorticity: torch.Tensor  # Shape: [vocab_size, 3]
    
    # Metric tensor - BEFORE: Minkowski η_μν, AFTER: curved by frame-dragging
    metric: torch.Tensor  # Shape: [vocab_size, 4, 4]
    
    # Christoffel symbols - BEFORE: zero, AFTER: Γ^μ_νρ from metric
    christoffel: torch.Tensor  # Shape: [vocab_size, 4, 4, 4]
    
    # Total energy - BEFORE: high, AFTER: minimized via evolution
    energy: float = float('inf')
    
    # Entanglement entropy - BEFORE: zero, AFTER: S = -Tr(ρ log ρ)
    entropy: float = 0.0
    
    # Time step for evolution
    time: int = 0
    
    # Device for computation
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))
    
    def to(self, device: torch.device) -> 'QuantumFieldState':
        """Move all tensors to device."""
        return QuantumFieldState(
            psi=self.psi.to(device),
            momentum=self.momentum.to(device),
            vorticity=self.vorticity.to(device),
            metric=self.metric.to(device),
            christoffel=self.christoffel.to(device),
            energy=self.energy,
            entropy=self.entropy,
            time=self.time,
            device=device
        )


# ==============================================================================
# LÉVY STABLE DISTRIBUTION
# ==============================================================================

def levy_stable_sample(alpha: float, beta: float, size: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """
    Sample from Lévy stable distribution L_α,β.
    
    Uses Chambers-Mallows-Stuck algorithm:
    X = sin(αV)/cos(V)^(1/α) * [cos(V(1-α))/W]^((1-α)/α)
    
    Args:
        alpha: Stability parameter α ∈ (0,2], we use φ ≈ 1.618
        beta: Skewness β ∈ [-1,1], we use 0 for symmetric
        size: Output tensor shape
        device: Computation device
        
    Returns:
        Tensor of Lévy stable samples
        
    Physics:
        P(x) ~ |x|^(-1-α) for large |x| (heavy tails)
        Variance is infinite for α < 2 (superdiffusion)
    """
    # Convert size to total number of elements for numpy
    total_elements = int(np.prod(size))
    
    # Use scipy for accurate Lévy sampling
    samples = levy_stable.rvs(alpha=alpha, beta=beta, size=total_elements)
    
    # Convert to torch tensor and reshape
    tensor = torch.from_numpy(samples).float().to(device)
    tensor = tensor.reshape(size)
    
    return tensor


# ==============================================================================
# FIBONACCI SEQUENCE UTILITIES
# ==============================================================================

def fibonacci_sequence(n: int) -> torch.Tensor:
    """
    Generate Fibonacci sequence up to n terms.
    
    F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}
    
    Growth rate: F_n ~ φ^n / √5
    """
    fib = torch.zeros(n)
    if n > 0:
        fib[0] = 0
    if n > 1:
        fib[1] = 1
    for i in range(2, n):
        fib[i] = fib[i-1] + fib[i-2]
    return fib


def fibonacci_sphere_points(n: int) -> torch.Tensor:
    """
    Generate n points on sphere using Fibonacci spiral.
    
    This gives optimal point distribution using:
    θ_k = 2πk/φ (golden angle)
    φ_k = arccos(1 - 2k/n) (latitude)
    
    Returns:
        Points on unit sphere [n, 3]
    """
    indices = torch.arange(0, n, dtype=torch.float32)
    
    # Golden angle increment
    theta = 2 * np.pi * indices / PHI
    
    # Latitude: uniform distribution in cos(φ)
    phi = torch.arccos(1 - 2 * indices / n)
    
    # Convert to Cartesian
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    
    return torch.stack([x, y, z], dim=1)


# ==============================================================================
# TOKENIZER CLASSES
# ==============================================================================

class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, text: str = None):
        """Initialize with optional text to build vocabulary."""
        if text is not None:
            self.chars = sorted(list(set(text)))
        else:
            # Basic ASCII printable characters
            self.chars = list(' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
        
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_token_id = 0
        self.eos_token_id = 1
        
    def encode(self, text: str, return_tensors: str = None):
        """Encode text to indices."""
        encoded = [self.char_to_idx.get(ch, 0) for ch in text]
        if return_tensors == 'pt':
            return torch.tensor(encoded)
        return encoded
    
    def decode(self, indices):
        """Decode indices to text."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        return ''.join([self.idx_to_char.get(int(i), '') for i in indices])
    
    def __len__(self):
        return len(self.chars)
    
    def __call__(self, text, **kwargs):
        """Make tokenizer callable like HuggingFace tokenizers."""
        return {'input_ids': self.encode(text, return_tensors='pt')}


class SimpleWordTokenizer:
    """Simple word-level tokenizer."""
    
    def __init__(self, vocab_size: int = 10000):
        """Initialize with fixed vocabulary size."""
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        
        # Reserve special tokens
        self.word_to_idx['<pad>'] = 0
        self.word_to_idx['<eos>'] = 1
        self.word_to_idx['<unk>'] = 2
        
        for i in range(3):
            self.idx_to_word[i] = list(self.word_to_idx.keys())[i]
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        from collections import Counter
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Take most common words
        most_common = word_counts.most_common(self.vocab_size - 3)
        
        # Add to vocabulary
        for idx, (word, _) in enumerate(most_common, start=3):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def encode(self, text: str, return_tensors: str = None):
        """Encode text to indices."""
        words = text.lower().split()
        encoded = [self.word_to_idx.get(w, self.unk_token_id) for w in words]
        if return_tensors == 'pt':
            return torch.tensor(encoded)
        return encoded
    
    def decode(self, indices):
        """Decode indices to text."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        words = [self.idx_to_word.get(int(i), '<unk>') for i in indices]
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word_to_idx)
    
    def __call__(self, text, max_length=None, truncation=False, padding=False, return_tensors=None):
        """Make tokenizer callable like HuggingFace tokenizers."""
        encoded = self.encode(text, return_tensors=return_tensors)
        
        if truncation and max_length and len(encoded) > max_length:
            encoded = encoded[:max_length]
        
        if padding and max_length and len(encoded) < max_length:
            if return_tensors == 'pt':
                pad_length = max_length - len(encoded)
                encoded = torch.cat([encoded, torch.zeros(pad_length, dtype=torch.long)])
            else:
                encoded.extend([self.pad_token_id] * (max_length - len(encoded)))
        
        return {'input_ids': encoded}


# ==============================================================================
# PURE PHYSICS LANGUAGE MODEL
# ==============================================================================

class PurePhysicsLanguageModel:
    """
    Language model based entirely on quantum field evolution.
    
    NO NEURAL NETWORKS. NO BACKPROP. JUST PHYSICS.
    
    Core principles:
    1. Tokens are quantum field excitations
    2. Evolution via Schrödinger equation minimizes energy
    3. Vortices maintain semantic coherence
    4. Frame-dragging creates memory
    5. Born rule collapses to next token
    
    The model "learns" by evolving toward ground state configurations
    that minimize total field energy while preserving information.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 field_dim: int = 128,
                 dt: float = 0.01,
                 levy_alpha: float = PHI,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize quantum field for language modeling.
        
        Args:
            vocab_size: Number of tokens in vocabulary
            field_dim: Dimensionality of quantum field
            dt: Time step for evolution Δt
            levy_alpha: Lévy index α (use φ for golden ratio)
            device: Computation device
        """
        self.vocab_size = vocab_size
        self.field_dim = field_dim
        self.dt = dt
        self.levy_alpha = levy_alpha
        self.device = torch.device(device)
        
        # Physical parameters
        self.hbar = HBAR
        self.mass = 1.0  # Effective mass of field quanta
        self.coupling = PHI  # Field self-coupling strength
        self.temperature = 1.0  # Initial temperature
        
        # Initialize quantum field state
        self.field = self._initialize_field()
        
        # Fibonacci scaling for long sequences
        self.fib_sequence = fibonacci_sequence(1000).to(self.device)
        self.fib_sphere = fibonacci_sphere_points(min(vocab_size, 1000)).to(self.device)
        
        print(f"📐 Initialized Pure Physics Language Model")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Field dimension: {field_dim}")  
        print(f"   Device: {device}")
        print(f"   Lévy α = φ = {levy_alpha:.6f}")
    
    def _initialize_field(self) -> QuantumFieldState:
        """
        Initialize quantum field in high-energy state.
        
        BEFORE: Random quantum fluctuations with high energy
        ψ(x,0) = Σ_k A_k exp(ikx + iφ_k) with random A_k, φ_k
        
        AFTER: Will evolve to ground state via Schrödinger equation
        ψ(x,t) → ψ_0(x) as t → ∞
        """
        # Initialize wave function on Fibonacci sphere with quantum fluctuations
        # PHYSICS: ψ = R exp(iΘ) with R from Lévy distribution
        amplitudes = levy_stable_sample(self.levy_alpha, 0, (self.vocab_size, self.field_dim), self.device)
        phases = torch.rand(self.vocab_size, self.field_dim, device=self.device) * 2 * np.pi
        
        # Complex wave function (stored as real tensor with 2x dims)
        psi_real = amplitudes * torch.cos(phases)
        psi_imag = amplitudes * torch.sin(phases)
        psi = torch.stack([psi_real, psi_imag], dim=-1)  # [vocab_size, field_dim, 2]
        
        # Zero initial momentum (system at rest)
        # PHYSICS: π = ∂L/∂(∂ψ/∂t) = 0 initially
        momentum = torch.zeros_like(psi)
        
        # Zero initial vorticity (no flow yet)
        # PHYSICS: ω = ∇ × v = 0
        vorticity = torch.zeros(self.vocab_size, 3, device=self.device)
        
        # Minkowski metric initially (flat spacetime)
        # PHYSICS: g_μν = diag(-1, 1, 1, 1)
        metric = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.vocab_size, 1, 1)
        metric[:, 0, 0] = -1  # Timelike component
        
        # Zero Christoffel symbols (no curvature yet)
        # PHYSICS: Γ^μ_νρ = 0 in flat space
        christoffel = torch.zeros(self.vocab_size, 4, 4, 4, device=self.device)
        
        # Compute initial energy
        # PHYSICS: E = ∫ [|∂ψ/∂t|² + |∇ψ|² + V(ψ)] d³x
        energy = self._compute_field_energy(psi, momentum)
        
        return QuantumFieldState(
            psi=psi,
            momentum=momentum,
            vorticity=vorticity,
            metric=metric,
            christoffel=christoffel,
            energy=energy,
            entropy=0.0,
            time=0,
            device=self.device
        )
    
    def _compute_field_energy(self, psi: torch.Tensor, momentum: torch.Tensor) -> float:
        """
        Compute total field energy (Hamiltonian).
        
        H = ∫ d³x [π²/2m + (ℏ²/2m)|∇ψ|² + V(ψ)]
        
        where V(ψ) = (g/4)|ψ|⁴ - (μ²/2)|ψ|² (Mexican hat potential)
        
        Args:
            psi: Wave function [vocab_size, field_dim, 2]
            momentum: Conjugate momentum [vocab_size, field_dim, 2]
            
        Returns:
            Total energy E (scalar)
        """
        # Kinetic energy: T = π²/2m
        T = torch.sum(momentum**2) / (2 * self.mass)
        
        # Gradient energy (discrete Laplacian)
        # PHYSICS: |∇ψ|² approximated by finite differences
        psi_shifted = torch.roll(psi, shifts=1, dims=0)
        grad_psi = (psi - psi_shifted) / self.dt
        gradient_energy = (self.hbar**2 / (2 * self.mass)) * torch.sum(grad_psi**2)
        
        # Potential energy: V(ψ) = (g/4)|ψ|⁴ - (μ²/2)|ψ|²
        # This creates symmetry breaking and ground states
        psi_squared = torch.sum(psi**2, dim=-1)  # |ψ|²
        V = (self.coupling/4) * torch.sum(psi_squared**2) - (self.mass**2/2) * torch.sum(psi_squared)
        
        total_energy = (T + gradient_energy + V).item()
        return total_energy
    
    def evolve_schrodinger(self, source_tokens: Optional[torch.Tensor] = None):
        """
        Evolve quantum field via Schrödinger equation.
        
        iℏ ∂ψ/∂t = Ĥψ where Ĥ = -ℏ²/2m ∇² + V(ψ)
        
        Using symplectic integrator to preserve energy:
        ψ(t+dt) = ψ(t) + dt * π(t)/m
        π(t+dt) = π(t) - dt * ∂H/∂ψ
        
        Args:
            source_tokens: Optional token indices to act as sources J(x)
            
        BEFORE: High energy disordered state
        AFTER: Lower energy ordered state (learning!)
        """
        # Extract current state
        psi = self.field.psi
        momentum = self.field.momentum
        
        # Add source terms if tokens provided
        # PHYSICS: J(x)ψ term in field equation
        if source_tokens is not None:
            source_field = self._tokens_to_field_source(source_tokens)
            psi = psi + self.dt * source_field
        
        # Compute forces F = -∂H/∂ψ
        # Laplacian term: ∇²ψ using roll for circular boundary
        # psi shape: [vocab_size, field_dim, 2]
        psi_next = torch.roll(psi, shifts=-1, dims=0)  # [vocab_size, field_dim, 2]
        psi_prev = torch.roll(psi, shifts=1, dims=0)   # [vocab_size, field_dim, 2]
        laplacian = (psi_next + psi_prev - 2*psi) / self.dt**2  # [vocab_size, field_dim, 2]
        
        # Potential force: -∂V/∂ψ = μ²ψ - g|ψ|²ψ
        psi_mag_sq = torch.sum(psi**2, dim=-1, keepdim=True)
        potential_force = self.mass**2 * psi - self.coupling * psi_mag_sq * psi
        
        # Total force
        force = -(self.hbar**2 / (2*self.mass)) * laplacian + potential_force
        
        # Symplectic update (preserves energy)
        momentum_new = momentum - self.dt * force
        psi_new = psi + self.dt * momentum_new / self.mass
        
        # Update field state
        self.field.psi = psi_new
        self.field.momentum = momentum_new
        self.field.energy = self._compute_field_energy(psi_new, momentum_new)
        self.field.time += 1
        
        # Compute entanglement entropy
        # PHYSICS: S = -Tr(ρ log ρ) where ρ = |ψ⟩⟨ψ|
        self._update_entropy()
    
    def _tokens_to_field_source(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert token indices to quantum field sources.
        
        Each token creates a localized excitation:
        J_k(x) = A exp(-(x-x_k)²/2σ²) exp(ik·x)
        
        Args:
            tokens: Token indices [batch_size, seq_len]
            
        Returns:
            Source field J(x) [vocab_size, field_dim, 2]
        """
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        
        batch_size, seq_len = tokens.shape
        source = torch.zeros_like(self.field.psi)
        
        # Each token creates a Gaussian wave packet
        for b in range(batch_size):
            for t in range(seq_len):
                token_id = tokens[b, t].item()
                if token_id >= self.vocab_size:
                    continue
                
                # Gaussian envelope centered at token position
                # PHYSICS: exp(-(x-x_k)²/2σ²)
                positions = torch.arange(self.vocab_size, device=self.device).float()
                gaussian = torch.exp(-(positions - token_id)**2 / (2 * seq_len))
                
                # Add momentum kick based on position in sequence
                # PHYSICS: exp(ik·x) with k ~ t/seq_len
                phase = 2 * np.pi * t / seq_len
                
                # Create complex source
                source[:, t % self.field_dim, 0] += gaussian * math.cos(phase)
                source[:, t % self.field_dim, 1] += gaussian * math.sin(phase)
        
        return source / (batch_size * seq_len + EPS)  # Normalize
    
    def compute_vorticity(self):
        """
        Compute vorticity field from quantum flow.
        
        ω = ∇ × v where v = (ℏ/m) Im[ψ*∇ψ/|ψ|²]
        
        This is the quantum probability current that generates
        semantic coherence through topological conservation.
        
        BEFORE: ω = 0 (no flow)
        AFTER: ω ≠ 0 with conserved circulation Γ = ∮ v·dl
        """
        psi = self.field.psi
        
        # Compute probability current
        # PHYSICS: j = (ℏ/2mi)[ψ*∇ψ - ψ∇ψ*]
        psi_real = psi[..., 0]
        psi_imag = psi[..., 1]
        
        # Gradient using circular boundary conditions
        psi_real_grad = torch.roll(psi_real, -1, dims=0) - torch.roll(psi_real, 1, dims=0)
        psi_imag_grad = torch.roll(psi_imag, -1, dims=0) - torch.roll(psi_imag, 1, dims=0)
        
        # Probability current components
        j_x = psi_real * psi_imag_grad - psi_imag * psi_real_grad
        j_y = torch.roll(j_x, shifts=self.field_dim//4, dims=1)  # Phase shift for y
        j_z = torch.roll(j_x, shifts=self.field_dim//2, dims=1)  # Phase shift for z
        
        # Compute curl (vorticity)
        # PHYSICS: ω = ∇ × j
        omega_x = torch.roll(j_z, -1, dims=0) - torch.roll(j_y, -1, dims=0)
        omega_y = torch.roll(j_x, -1, dims=0) - torch.roll(j_z, -1, dims=0)  
        omega_z = torch.roll(j_y, -1, dims=0) - torch.roll(j_x, -1, dims=0)
        
        # Average over field dimensions
        vorticity = torch.stack([
            omega_x.mean(dim=1),
            omega_y.mean(dim=1),
            omega_z.mean(dim=1)
        ], dim=1)
        
        self.field.vorticity = vorticity * (self.hbar / self.mass)
    
    def update_frame_dragging(self):
        """
        Update spacetime metric from vorticity (frame-dragging).
        
        In general relativity, rotating matter drags spacetime:
        g_tφ = -2J/r³ (Lense-Thirring effect)
        
        We implement this as vorticity modifying the metric:
        h_μν = (2G/c²) T_μν where T_μν includes angular momentum
        
        BEFORE: g_μν = η_μν (flat Minkowski)
        AFTER: g_μν = η_μν + h_μν (curved by rotation)
        """
        vorticity = self.field.vorticity
        
        # Angular momentum from vorticity
        # PHYSICS: J = ∫ r × (ρv) d³x with v related to ω by ω = ∇×v
        J = torch.norm(vorticity, dim=1)  # Simplified: |J| ~ |ω|
        
        # Metric perturbation from rotation
        # PHYSICS: h_0i = -(2G/c³) ε_ijk J_j x_k / r³
        for i in range(self.vocab_size):
            r = max(1.0, float(i))  # Radial distance (preventing division by zero)
            
            # Off-diagonal metric components (frame-dragging)
            self.field.metric[i, 0, 1:] = -2 * G * J[i] / (C**3 * r**3)
            self.field.metric[i, 1:, 0] = -2 * G * J[i] / (C**3 * r**3)
        
        # Update Christoffel symbols
        # PHYSICS: Γ^μ_νρ = (1/2)g^μσ(∂_ν g_σρ + ∂_ρ g_σν - ∂_σ g_νρ)
        self._update_christoffel_symbols()
    
    def _update_christoffel_symbols(self):
        """
        Compute Christoffel symbols from metric.
        
        Γ^μ_νρ = (1/2)g^μσ(∂_ν g_σρ + ∂_ρ g_σν - ∂_σ g_νρ)
        
        These determine how vectors parallel transport in curved space,
        creating "memory" of previous tokens through geometry.
        """
        g = self.field.metric
        
        # Compute metric derivatives (finite differences)
        dg = torch.zeros(self.vocab_size, 4, 4, 4, device=self.device)
        for mu in range(4):
            g_shifted = torch.roll(g, shifts=-1, dims=0)
            dg[:, mu] = (g_shifted - g) / self.dt
        
        # Inverse metric (simplified for small perturbations)
        try:
            g_inv = torch.inverse(g + EPS * torch.eye(4, device=self.device))
        except:
            # If inversion fails, use identity
            g_inv = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.vocab_size, 1, 1)
        
        # Christoffel symbols (simplified computation)
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    # Simplified version to avoid numerical issues
                    self.field.christoffel[:, mu, nu, rho] = 0.5 * dg[:, mu, nu, rho].mean()
    
    def apply_levy_jump(self, temperature: float = 1.0):
        """
        Apply Lévy flight perturbation for exploration.
        
        Δψ ~ L_α(T) with α = φ for optimal exploration/exploitation.
        
        This replaces SGD - the system explores configuration space
        through heavy-tailed jumps that can escape local minima.
        
        Args:
            temperature: Controls jump magnitude T
        """
        # Sample Lévy perturbation
        # PHYSICS: P(|Δψ| > x) ~ x^(-α) for large x
        levy_noise = levy_stable_sample(
            self.levy_alpha, 0, 
            self.field.psi.shape, 
            self.device
        )
        
        # Temperature-scaled perturbation
        self.field.psi += temperature * levy_noise * self.dt
        
        # Renormalize to preserve probability
        # PHYSICS: ∫|ψ|² dx = 1
        norm = torch.sqrt(torch.sum(self.field.psi**2) + EPS)
        self.field.psi = self.field.psi / norm
    
    def _update_entropy(self):
        """
        Compute von Neumann entropy of quantum state.
        
        S = -Tr(ρ log ρ) where ρ = |ψ⟩⟨ψ|
        
        Measures entanglement and disorder in the field.
        """
        psi = self.field.psi
        
        # Compute density matrix (reduced over field dimensions)
        psi_flat = psi.reshape(self.vocab_size, -1)
        psi_norm = psi_flat / (torch.norm(psi_flat, dim=1, keepdim=True) + EPS)
        
        # Reduced density matrix (use subset for efficiency)
        subset_size = min(100, self.vocab_size)
        psi_subset = psi_norm[:subset_size]
        rho = torch.matmul(psi_subset, psi_subset.T)
        
        # Von Neumann entropy
        try:
            eigenvalues = torch.linalg.eigvalsh(rho)
            eigenvalues = torch.clamp(eigenvalues, min=EPS)
            entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
            self.field.entropy = entropy.item()
        except:
            self.field.entropy = 0.0
    
    def born_rule_collapse(self, context_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Collapse wave function to measure next token (Born rule).
        
        P(token = k) = |⟨k|ψ⟩|² with optional context biasing.
        
        Args:
            context_tokens: Previous tokens for conditional generation
            
        Returns:
            Sampled token index
            
        PHYSICS: This is quantum measurement causing wave function collapse
        """
        psi = self.field.psi
        
        # Compute probability distribution
        # PHYSICS: P(k) = |ψ_k|² (Born rule)
        prob = torch.sum(psi**2, dim=(1, 2))  # Sum over field dims and complex parts
        
        # Apply context biasing through metric
        if context_tokens is not None:
            # Context creates gravitational potential
            # PHYSICS: P(k) → P(k) exp(-V_context(k)/T)
            context_potential = self._compute_context_potential(context_tokens)
            prob = prob * torch.exp(-context_potential / self.temperature)
        
        # Normalize
        prob = prob / (torch.sum(prob) + EPS)
        
        # Sample token (wave function collapse)
        token = torch.multinomial(prob, num_samples=1)
        
        # Collapse wave function (measurement back-action)
        # PHYSICS: ψ → |k⟩⟨k|ψ⟩/√⟨ψ|k⟩⟨k|ψ⟩
        self._apply_measurement_backaction(token)
        
        return token
    
    def _compute_context_potential(self, context_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute gravitational potential from context tokens.
        
        V(x) = -G Σ_i m_i/|x - x_i| where x_i are context positions.
        
        Args:
            context_tokens: Previous token indices
            
        Returns:
            Potential field V(x)
        """
        potential = torch.zeros(self.vocab_size, device=self.device)
        
        for token_id in context_tokens.flatten():
            if token_id >= self.vocab_size:
                continue
            # Each token creates gravitational well
            # PHYSICS: V = -Gm/r with Yukawa cutoff
            distances = torch.abs(torch.arange(self.vocab_size, device=self.device) - token_id.float())
            distances = torch.clamp(distances, min=1.0)
            
            # Yukawa potential (massive graviton)
            potential -= G * torch.exp(-distances / self.field_dim) / distances
        
        return potential
    
    def _apply_measurement_backaction(self, measured_token: torch.Tensor):
        """
        Apply measurement back-action to wave function.
        
        After measuring token k, the wave function partially collapses
        toward that state while maintaining some coherence.
        
        ψ → √(1-ε)|ψ⟩ + √ε|k⟩
        
        Args:
            measured_token: Index of measured token
        """
        epsilon = 0.1  # Partial collapse parameter
        
        # Create collapsed state
        collapsed = torch.zeros_like(self.field.psi)
        token_idx = measured_token.item()
        if token_idx < self.vocab_size:
            collapsed[token_idx] = torch.randn_like(collapsed[token_idx]) 
        
        # Partial collapse
        self.field.psi = np.sqrt(1 - epsilon) * self.field.psi + np.sqrt(epsilon) * collapsed
        
        # Renormalize
        norm = torch.sqrt(torch.sum(self.field.psi**2) + EPS)
        self.field.psi = self.field.psi / norm
    
    def train_batch(self, token_batch: torch.Tensor) -> Dict[str, float]:
        """
        'Train' on a batch by evolving the quantum field.
        
        NO BACKPROP - just physics evolution toward ground state.
        
        Args:
            token_batch: Batch of token sequences [batch_size, seq_len]
            
        Returns:
            Dictionary of physics metrics
        """
        initial_energy = self.field.energy
        
        # Inject tokens as field sources
        self.evolve_schrodinger(source_tokens=token_batch)
        
        # Update derived quantities
        self.compute_vorticity()
        self.update_frame_dragging()
        
        # Lévy exploration with probability
        if np.random.random() < 0.01:  # 1% chance
            self.apply_levy_jump(temperature=self.temperature)
        
        # Cool down temperature (simulated annealing)
        self.temperature *= 0.999
        
        # Compute physics metrics
        metrics = {
            'energy': self.field.energy,
            'energy_change': self.field.energy - initial_energy,
            'entropy': self.field.entropy,
            'vorticity_mean': torch.mean(torch.norm(self.field.vorticity, dim=1)).item(),
            'vorticity_max': torch.max(torch.norm(self.field.vorticity, dim=1)).item(),
            'temperature': self.temperature,
            'time_step': self.field.time
        }
        
        return metrics
    
    def generate(self, 
                 prompt_tokens: torch.Tensor,
                 max_length: int = 100,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        Generate tokens using Born rule collapse.
        
        Args:
            prompt_tokens: Initial token sequence
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated token sequence
        """
        generated = prompt_tokens.clone()
        
        # Set generation temperature
        old_temp = self.temperature
        self.temperature = temperature
        
        for _ in range(max_length - len(prompt_tokens)):
            # Evolve field with context
            self.evolve_schrodinger(source_tokens=generated.unsqueeze(0))
            self.compute_vorticity()
            
            # Collapse to next token
            next_token = self.born_rule_collapse(context_tokens=generated)
            
            # Append to sequence
            generated = torch.cat([generated, next_token])
            
            # Early stopping if energy is very low (converged)
            if self.field.energy < 1e-6:
                break
        
        # Restore temperature
        self.temperature = old_temp
        
        return generated


# ==============================================================================
# DATASET HANDLERS
# ==============================================================================

class WikiTextDataset(Dataset):
    """Simple WikiText dataset handler."""
    
    def __init__(self, tokenizer, max_length: int = 256):
        """Initialize with tokenizer and load data."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Download WikiText-2
        print("📥 Downloading WikiText-2...")
        url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
        response = requests.get(url)
        self.text = response.text
        
        # Split into chunks
        self.chunks = []
        lines = self.text.split('\n')
        current_chunk = []
        current_length = 0
        
        for line in lines:
            if line.strip():  # Skip empty lines
                # Try to encode, handle if vocabulary not built yet
                try:
                    tokens = tokenizer.encode(line)
                    if isinstance(tokens, torch.Tensor):
                        tokens = tokens.tolist()
                except:
                    # If encoding fails, skip this line
                    continue
                    
                if current_length + len(tokens) > max_length:
                    if current_chunk:
                        self.chunks.append(current_chunk)
                    current_chunk = tokens[:max_length]
                    current_length = len(current_chunk)
                else:
                    current_chunk.extend(tokens)
                    current_length += len(tokens)
        
        if current_chunk:
            self.chunks.append(current_chunk)
        
        print(f"📚 Loaded WikiText-2: {len(self.chunks)} chunks")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        tokens = self.chunks[idx]
        # Pad if necessary
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        return torch.tensor(tokens[:self.max_length])


class ShakespeareDataset(Dataset):
    """Character-level Shakespeare dataset."""
    
    def __init__(self, tokenizer, max_length: int = 256):
        """Initialize with tokenizer and load data."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Download Shakespeare
        print("📥 Downloading Tiny Shakespeare...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        self.text = response.text
        
        # Create chunks
        self.chunks = []
        for i in range(0, len(self.text) - max_length, max_length // 2):
            chunk = self.text[i:i + max_length]
            tokens = tokenizer.encode(chunk)
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            self.chunks.append(tokens)
        
        print(f"📚 Loaded Shakespeare: {len(self.chunks)} chunks")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        tokens = self.chunks[idx]
        # Pad if necessary
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens[:self.max_length])


# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================

def train_pure_physics_model(
    model: PurePhysicsLanguageModel,
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 0  # Set to 0 to avoid multiprocessing issues
) -> Dict[str, List[float]]:
    """
    Train model through physics evolution (no backprop!).
    
    Args:
        model: PurePhysicsLanguageModel instance
        dataset: Dataset instance
        batch_size: Batch size
        num_workers: DataLoader workers
        
    Returns:
        Training history
    """
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Training history
    history = {
        'energy': [],
        'entropy': [],
        'vorticity': [],
        'temperature': []
    }
    
    print("\n🌊 Starting Pure Physics Training (NO BACKPROP!)")
    print(f"   Batch size: {batch_size}")
    print(f"   Dataset size: {len(dataset)}")
    print("=" * 60)
    
    start_time = time.time()
    
    for batch_idx, tokens in enumerate(tqdm(dataloader, desc="Training")):
        tokens = tokens.to(model.device)
        
        # Evolve quantum field (this IS training!)
        metrics = model.train_batch(tokens)
        
        # Record metrics
        history['energy'].append(metrics['energy'])
        history['entropy'].append(metrics['entropy'])
        history['vorticity'].append(metrics['vorticity_mean'])
        history['temperature'].append(metrics['temperature'])
        
        # Print progress
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"\nBatch {batch_idx:4d} | "
                  f"Energy: {metrics['energy']:8.4f} | "
                  f"ΔE: {metrics['energy_change']:+8.4f} | "
                  f"Entropy: {metrics['entropy']:6.2f} | "
                  f"Vorticity: {metrics['vorticity_mean']:6.4f} | "
                  f"Time: {elapsed:6.1f}s")
            
            # Check for convergence
            if abs(metrics['energy_change']) < 1e-6:
                print("\n✨ Converged to ground state!")
                break
        
        # Early stopping for demo
        if batch_idx >= 100:
            break
    
    print(f"\n📊 Training complete in {time.time() - start_time:.1f} seconds")
    
    return history


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_physics_evolution(history: Dict[str, List[float]], save_path: str = 'physics_evolution.png'):
    """
    Visualize the physics of learning without backprop.
    
    Args:
        history: Training history
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Pure Physics Language Model Evolution', fontsize=16)
    
    # Energy minimization
    ax = axes[0, 0]
    ax.plot(history['energy'], 'b-', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Field Energy')
    ax.set_title('Energy Minimization (Natural Learning)')
    ax.grid(True, alpha=0.3)
    
    # Entropy evolution
    ax = axes[0, 1]
    ax.plot(history['entropy'], 'r-', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Von Neumann Entropy')
    ax.set_title('Quantum Entanglement Evolution')
    ax.grid(True, alpha=0.3)
    
    # Vorticity (coherence)
    ax = axes[1, 0]
    ax.plot(history['vorticity'], 'g-', linewidth=2)
    ax.axhline(y=2*np.pi, color='red', linestyle='--', label='Critical vorticity')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Vorticity')
    ax.set_title('Semantic Vortex Formation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temperature (annealing)
    ax = axes[1, 1]
    ax.semilogy(history['temperature'], 'orange', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Temperature (log scale)')
    ax.set_title('Simulated Annealing')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Saved physics visualization to {save_path}")


# ==============================================================================
# MAIN DEMONSTRATION
# ==============================================================================

def main():
    """
    Demonstrate Pure Physics Language Model on multiple datasets.
    """
    print("=" * 80)
    print("PURE PHYSICS LANGUAGE MODEL - NO BACKPROP, JUST QUANTUM EVOLUTION")
    print("=" * 80)
    
    # Test on WikiText-2 with word tokenizer
    print("\n1️⃣ Testing on WikiText-2...")
    
    # Create simple word tokenizer
    word_tokenizer = SimpleWordTokenizer(vocab_size=5000)
    
    # Load dataset and build vocabulary
    print("Building vocabulary...")
    # First download raw text to build vocabulary
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    response = requests.get(url)
    wiki_text = response.text
    
    # Build vocabulary from raw text
    texts = [wiki_text[i:i+10000] for i in range(0, min(100000, len(wiki_text)), 10000)]
    word_tokenizer.build_vocab(texts)
    print(f"Vocabulary size: {len(word_tokenizer)}")
    
    # Now create dataset with proper tokenizer
    wiki_dataset = WikiTextDataset(word_tokenizer, max_length=128)
    
    # Initialize model
    wiki_model = PurePhysicsLanguageModel(
        vocab_size=len(word_tokenizer),
        field_dim=64,
        dt=0.01,
        levy_alpha=PHI
    )
    
    # Train through physics evolution
    wiki_history = train_pure_physics_model(
        wiki_model,
        wiki_dataset,
        batch_size=4
    )
    
    # Visualize
    visualize_physics_evolution(wiki_history, 'wiki_physics_evolution.png')
    
    # Generate sample
    print("\n📝 WikiText-2 Generation Sample:")
    prompt = "The quantum field theory"
    prompt_tokens = word_tokenizer.encode(prompt, return_tensors='pt')
    if isinstance(prompt_tokens, list):
        prompt_tokens = torch.tensor(prompt_tokens)
    prompt_tokens = prompt_tokens.to(wiki_model.device)
    
    generated_tokens = wiki_model.generate(prompt_tokens, max_length=50)
    generated_text = word_tokenizer.decode(generated_tokens.cpu())
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    # Test on Tiny Shakespeare
    print("\n2️⃣ Testing on Tiny Shakespeare...")
    
    # Download and create character tokenizer
    print("📥 Loading Shakespeare text...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    shakespeare_text = response.text
    
    # Create character tokenizer
    char_tokenizer = CharTokenizer(shakespeare_text)
    print(f"Character vocabulary size: {len(char_tokenizer)}")
    
    # Create dataset
    shakespeare_dataset = ShakespeareDataset(char_tokenizer, max_length=256)
    
    # Initialize Shakespeare model
    shakespeare_model = PurePhysicsLanguageModel(
        vocab_size=len(char_tokenizer),
        field_dim=32,
        dt=0.01,
        levy_alpha=PHI
    )
    
    # Train
    shakespeare_history = train_pure_physics_model(
        shakespeare_model,
        shakespeare_dataset,
        batch_size=8
    )
    
    # Visualize
    visualize_physics_evolution(shakespeare_history, 'shakespeare_physics_evolution.png')
    
    # Generate Shakespeare
    print("\n🎭 Shakespeare Generation Sample:")
    prompt = "To be or not to be"
    prompt_tokens = char_tokenizer.encode(prompt, return_tensors='pt')
    prompt_tokens = prompt_tokens.to(shakespeare_model.device)
    
    generated_tokens = shakespeare_model.generate(prompt_tokens, max_length=200)
    generated_text = char_tokenizer.decode(generated_tokens)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    # Final physics analysis
    print("\n📊 Final Physics Analysis:")
    print(f"WikiText-2 Model:")
    print(f"  - Final energy: {wiki_model.field.energy:.6f}")
    print(f"  - Final entropy: {wiki_model.field.entropy:.6f}")
    print(f"  - Max vorticity: {torch.max(torch.norm(wiki_model.field.vorticity, dim=1)).item():.6f}")
    
    print(f"\nShakespeare Model:")
    print(f"  - Final energy: {shakespeare_model.field.energy:.6f}")
    print(f"  - Final entropy: {shakespeare_model.field.entropy:.6f}")
    print(f"  - Max vorticity: {torch.max(torch.norm(shakespeare_model.field.vorticity, dim=1)).item():.6f}")
    
    print("\n✨ Pure Physics Language Modeling Complete!")
    print("🌟 NO BACKPROP WAS USED - ONLY QUANTUM FIELD EVOLUTION!")


if __name__ == "__main__":
    main()
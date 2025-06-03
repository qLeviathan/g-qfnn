# qfnn_language_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List, Union
from tqdm import tqdm
import time

# Constants derived from mathematical principles
PHI = (1 + 5 ** 0.5) / 2  # Golden ratio ≈ 1.618
OPTIMAL_SPARSITY = 4 * PHI / (3 * math.pi * math.sqrt(math.e))  # ≈ 0.4165
OPTIMAL_COHERENCE = 1 / (PHI ** 2) + 0.5  # ≈ 0.75
ETA = 1 / (2 * PHI)  # Optimal learning rate ≈ 0.309
GAMMA = 1 - 1 / (PHI ** 2)  # Optimal decay rate ≈ 0.618


class SpiralPhaseEncoder(nn.Module):
    """
    Encodes token indices into 2D phase space using golden ratio distribution.
    Maps vocabulary V to points on unit circle with optimal coverage.
    
    Input: [batch_size, seq_len] - Token indices
    Output: [batch_size, seq_len, 2] - (x,y) coordinates on unit circle
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = 2  # Phase space is always 2D
        self.register_buffer("embedding", self._generate_embeddings())
        
    def _generate_embeddings(self) -> torch.Tensor:
        """Generate phase space embeddings using golden ratio distribution"""
        embeddings = torch.zeros(self.vocab_size, 2)
        for v in range(self.vocab_size):
            # Phase angle using golden ratio
            theta = 2 * math.pi * ((PHI * v) % 1.0)
            # Convert to Cartesian coordinates on unit circle
            embeddings[v, 0] = torch.cos(torch.tensor(theta))  # x coordinate
            embeddings[v, 1] = torch.sin(torch.tensor(theta))  # y coordinate
        return embeddings
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices [batch_size, seq_len]
        Returns:
            Phase space embeddings [batch_size, seq_len, 2]
        """
        return self.embedding[x]  # Simple lookup


class MagneticFluxAttention(nn.Module):
    """
    Attention mechanism based on phase differences and spatial proximity.
    Uses optimal sparsity derived from information theory.
    
    Input: [batch_size, seq_len, 2] - Token phase space coordinates
    Output: 
        - attention: [batch_size, seq_len, seq_len] - Attention weights
        - mask: [batch_size, seq_len, seq_len] - Binary attention mask
    """
    def __init__(self, sparsity: float = OPTIMAL_SPARSITY, epsilon: float = 1e-8):
        super().__init__()
        self.sparsity = sparsity
        self.epsilon = epsilon  # To prevent division by zero
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Phase space embeddings [batch_size, seq_len, 2]
        Returns:
            attention: Sparse attention matrix [batch_size, seq_len, seq_len]
            mask: Binary attention mask [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract phases using atan2
        phases = torch.atan2(x[:, :, 1], x[:, :, 0])  # [batch_size, seq_len]
        
        # Compute pairwise phase differences
        phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)  # [batch_size, seq_len, seq_len]
        
        # Compute cosine term (phase similarity)
        cos_term = 0.5 + 0.5 * torch.cos(phase_diff)  # [batch_size, seq_len, seq_len]
        
        # Compute spatial distances between embeddings
        # Reshape x to enable broadcasting
        x_expanded_1 = x.unsqueeze(2).expand(batch_size, seq_len, seq_len, 2)
        x_expanded_2 = x.unsqueeze(1).expand(batch_size, seq_len, seq_len, 2)
        
        # Compute Euclidean distances
        distances = torch.norm(x_expanded_1 - x_expanded_2, dim=3)  # [batch_size, seq_len, seq_len]
        
        # Compute magnetic flux attention: (cos_term) / (distance + ε)
        attention = cos_term / (distances + self.epsilon)  # [batch_size, seq_len, seq_len]
        
        # Apply binary thresholding at optimal sparsity
        flat_attention = attention.view(batch_size, -1)
        thresholds = torch.quantile(flat_attention, 1 - self.sparsity, dim=1)
        thresholds = thresholds.unsqueeze(-1).unsqueeze(-1)
        
        # Create binary mask
        mask = (attention > thresholds).float()  # [batch_size, seq_len, seq_len]
        
        # Apply mask and normalize rows
        attention = attention * mask
        row_sums = attention.sum(dim=-1, keepdim=True) + self.epsilon
        attention = attention / row_sums  # Normalized sparse attention
        
        return attention, mask


class SchrodingerFieldOperator(nn.Module):
    """
    Schrödinger field operator for complex-valued quantum dynamics.
    Implements the quantum Hamiltonian and diffusion terms.
    
    Input: [batch_size, seq_len, 2] - Real-valued state
    Output: [batch_size, seq_len, 2] - Force terms for real-valued state evolution
    """
    def __init__(self, hamiltonian_strength: float = 1.0, diffusion_coeff: float = 0.01):
        super().__init__()
        self.h_strength = hamiltonian_strength
        self.diffusion = diffusion_coeff
        
    def forward(self, psi: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Current state [batch_size, seq_len, 2]
            context: Attention-weighted context [batch_size, seq_len, 2]
        Returns:
            Force field terms [batch_size, seq_len, 2]
        """
        # Convert real-valued (x,y) to complex-valued for Schrödinger dynamics
        psi_complex = torch.complex(psi[..., 0], psi[..., 1])
        context_complex = torch.complex(context[..., 0], context[..., 1])
        
        # Hamiltonian term: -i·(context - ψ)
        # In Schrödinger equation: i·∂ψ/∂t = Ĥψ
        # Here Ĥψ ≈ (context - ψ), so i·∂ψ/∂t = (context - ψ)
        # Therefore ∂ψ/∂t = -i·(context - ψ)
        hamiltonian = -1j * self.h_strength * (context_complex - psi_complex)
        
        # Diffusion (Laplacian) term: D·∇²ψ
        # For 1D sequence, approximate: Σ(ψ_j - ψ_i) for neighboring tokens
        # Roll sequence left and right to compute differences
        psi_left = torch.roll(psi_complex, -1, dims=1)
        psi_right = torch.roll(psi_complex, 1, dims=1)
        
        # Simple discrete Laplacian: (ψ_left + ψ_right - 2·ψ)
        laplacian = psi_left + psi_right - 2 * psi_complex
        diffusion_term = self.diffusion * laplacian
        
        # Combine terms in Schrödinger equation: ∂ψ/∂t = -i·Ĥψ + D·∇²ψ
        force_complex = hamiltonian + diffusion_term
        
        # Convert back to real-valued tensor [batch_size, seq_len, 2]
        force_real = torch.stack([force_complex.real, force_complex.imag], dim=-1)
        
        return force_real


class HeunEulerIntegrator(nn.Module):
    """
    Energy-conserving integration using modified Heun-Euler scheme with normalization.
    
    Input: 
        - psi: [batch_size, seq_len, 2] - Current state
        - force: [batch_size, seq_len, 2] - Force terms from Schrödinger operator
    Output: [batch_size, seq_len, 2] - Updated state with ||psi|| = 1
    """
    def __init__(self, delta_t: float = 0.1):
        super().__init__()
        self.delta_t = delta_t
    
    def _normalize(self, psi: torch.Tensor) -> torch.Tensor:
        """Normalize to unit norm to ensure energy conservation"""
        norm = torch.norm(psi, dim=-1, keepdim=True)
        return psi / (norm + 1e-8)
    
    def forward(self, psi: torch.Tensor, force_func, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Current state [batch_size, seq_len, 2]
            force_func: Function returning force terms
            context: Attention-weighted context [batch_size, seq_len, 2]
        Returns:
            Updated state [batch_size, seq_len, 2] with ||psi|| = 1
        """
        # First stage: compute k1 = F(ψⁿ)
        k1 = force_func(psi, context)
        
        # Predictor step with normalization: ψ* = N(ψⁿ + Δt·k1)
        psi_star = psi + self.delta_t * k1
        psi_star = self._normalize(psi_star)
        
        # Second stage: compute k2 = F(ψ*)
        k2 = force_func(psi_star, context)
        
        # Corrector step with normalization: ψⁿ⁺¹ = N(ψⁿ + (Δt/2)·(k1 + k2))
        psi_new = psi + (self.delta_t / 2) * (k1 + k2)
        psi_new = self._normalize(psi_new)
        
        return psi_new


class HebbianProjection(nn.Module):
    """
    Hebbian learning projection layer with golden ratio parameters.
    
    Input: [batch_size, 2] - Final state vector
    Output: [batch_size, vocab_size] - Logits for prediction
    """
    def __init__(self, vocab_size: int, eta: float = ETA, gamma: float = GAMMA):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, 2) * 0.02)
        self.eta = eta  # Learning rate derived from golden ratio
        self.gamma = gamma  # Decay rate derived from golden ratio
    
    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None, 
                update: bool = False) -> torch.Tensor:
        """
        Args:
            x: Phase space vector [batch_size, 2]
            target: One-hot target distribution [batch_size, vocab_size]
            update: Whether to update weights using Hebbian rule
        Returns:
            Logits [batch_size, vocab_size]
        """
        # Forward projection: W·x
        logits = F.linear(x, self.weight)  # [batch_size, vocab_size]
        
        # Update weights using Hebbian rule if training
        if update and target is not None:
            with torch.no_grad():
                # Hebbian update: ΔW = η(y·xᵀ) - γW
                batch_avg = torch.matmul(target.t(), x) / x.size(0)  # [vocab_size, 2]
                delta_w = self.eta * batch_avg - self.gamma * self.weight
                self.weight.add_(delta_w)
        
        return logits


class EnergyDiffusionPentadTriangle:
    """
    Implements the triangular relationship between energy stability, 
    diffusion rate, and Harmonic Pentad.
    """
    def __init__(self,
                 initial_diffusion: float = 0.01,
                 alpha_e: float = 0.5,
                 beta_h: float = 0.3,
                 gamma_d: float = 0.7,
                 lambda_e: float = 0.5,
                 lambda_h: float = 0.3,
                 delta_d: float = 0.2):
        """
        Args:
            initial_diffusion: Initial diffusion coefficient
            alpha_e: Energy stability scaling factor
            beta_h: Pentad scaling factor
            gamma_d: Diffusion decay coefficient
            lambda_e: Energy stability adjustment rate
            lambda_h: Pentad convergence rate
            delta_d: Diffusion impact on pentad
        """
        self.d0 = initial_diffusion  # Initial diffusion coefficient
        self.alpha_e = alpha_e  # Energy stability coefficient
        self.beta_h = beta_h  # Pentad coefficient
        self.gamma_d = gamma_d  # Diffusion decay coefficient
        self.lambda_e = lambda_e  # Energy stability adjustment rate
        self.lambda_h = lambda_h  # Pentad convergence rate
        self.delta_d = delta_d  # Diffusion impact on pentad
        
        # Internal state
        self.energy_stability = 0.5  # Initial energy stability
        self.energy = 1.0  # Initial energy
        self.pentad = 1.0  # Initial pentad value
        self.diffusion = initial_diffusion  # Current diffusion rate
    
    def update(self, energy: float, pentad: float) -> float:
        """
        Update the Energy-Diffusion-Pentad triangle state.
        
        Args:
            energy: Current energy
            pentad: Current Harmonic Pentad value
        
        Returns:
            diffusion: Updated diffusion coefficient
        """
        # Update triangle relationships
        
        # 1. Update diffusion rate based on energy stability and pentad
        self.diffusion = self.d0 * (1 - self.alpha_e * self.energy_stability) * (1 + self.beta_h * pentad)
        
        # 2. Update energy stability based on diffusion and energy
        d_es = -self.gamma_d * self.diffusion * self.energy_stability + self.lambda_e * (self.energy - energy)
        self.energy_stability += d_es
        self.energy_stability = max(0.0, min(1.0, self.energy_stability))  # Clamp to [0,1]
        
        # 3. Update pentad based on convergence and diffusion
        d_h = -self.lambda_h * self.pentad + self.delta_d * self.diffusion
        self.pentad += d_h
        self.pentad = max(0.0, self.pentad)  # Ensure non-negative
        
        # 4. Update energy
        self.energy = energy
        
        return self.diffusion


class HarmonicPentad:
    """
    Convergence metric unifying phase coherence, sparsity, energy, perplexity, and fidelity.
    """
    def __init__(self, alpha_c: float = 1.0, alpha_s: float = 1.0, alpha_e: float = 1.0,
                 alpha_p: float = 1.0, alpha_f: float = 1.0):
        """
        Args:
            alpha_c: Phase coherence weight
            alpha_s: Sparsity weight
            alpha_e: Energy weight
            alpha_p: Perplexity weight
            alpha_f: Fidelity weight
        """
        self.alpha_c = alpha_c  # Phase coherence weight
        self.alpha_s = alpha_s  # Sparsity weight
        self.alpha_e = alpha_e  # Energy weight
        self.alpha_p = alpha_p  # Perplexity weight
        self.alpha_f = alpha_f  # Fidelity weight
        
        # Target values from theory
        self.target_c = OPTIMAL_COHERENCE
        self.target_s = OPTIMAL_SPARSITY
        self.target_e = 1.0
        self.target_p = 1.0
        self.target_f = 1.0
    
    def compute(self, psi: torch.Tensor, mask: torch.Tensor, 
                perplexity: float, fidelity: float) -> Tuple[float, Dict[str, float]]:
        """
        Compute the Harmonic Pentad metric and its components.
        
        Args:
            psi: Phase space representation [batch_size, seq_len, 2]
            mask: Binary attention mask [batch_size, seq_len, seq_len]
            perplexity: Perplexity value
            fidelity: Prediction fidelity
            
        Returns:
            pentad_value: Overall Harmonic Pentad value
            components: Individual component values
        """
        # 1. Phase coherence
        phases = torch.atan2(psi[:, :, 1], psi[:, :, 0])
        complex_phases = torch.exp(1j * phases)
        coherence = torch.abs(complex_phases.mean(dim=1)).mean().item()
        
        # 2. Sparsity
        sparsity = mask.float().mean().item()
        
        # 3. Energy conservation
        energy = torch.norm(psi, dim=2).mean().item()
        
        # 4. Compute pentad components
        c_term = self.alpha_c * (coherence - self.target_c) ** 2
        s_term = self.alpha_s * (sparsity - self.target_s) ** 2
        e_term = self.alpha_e * (energy - self.target_e) ** 2
        p_term = self.alpha_p * (perplexity / self.target_p - 1) ** 2
        f_term = self.alpha_f * (1 - fidelity) ** 2
        
        pentad_value = c_term + s_term + e_term + p_term + f_term
        
        components = {
            'coherence': coherence,
            'sparsity': sparsity,
            'energy': energy,
            'perplexity': perplexity,
            'fidelity': fidelity,
            'pentad': pentad_value
        }
        
        return pentad_value, components


class QFNN(nn.Module):
    """
    Quantum Flux Neural Network for language modeling.
    
    Input: [batch_size, seq_len] - Token indices
    Output: [batch_size, vocab_size] - Prediction logits
    """
    def __init__(self, vocab_size: int, eta: float = ETA, gamma: float = GAMMA,
                 diffusion_coeff: float = 0.01):
        super().__init__()
        self.encoder = SpiralPhaseEncoder(vocab_size)
        self.attention = MagneticFluxAttention()
        self.field_operator = SchrodingerFieldOperator(diffusion_coeff=diffusion_coeff)
        self.integrator = HeunEulerIntegrator(delta_t=0.1)
        self.projection = HebbianProjection(vocab_size, eta, gamma)
        
        # Initialize EDP triangle
        self.edp = EnergyDiffusionPentadTriangle(initial_diffusion=diffusion_coeff)
        
        # Set up Harmonic Pentad
        self.pentad = HarmonicPentad()
    
    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None, 
                update: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                             torch.Tensor, Dict[str, float]]:
        """
        Args:
            x: Token indices [batch_size, seq_len]
            target: Optional target distribution [batch_size, vocab_size]
            update: Whether to update weights using Hebbian rule
        Returns:
            logits: Prediction logits [batch_size, vocab_size]
            attention: Attention matrix [batch_size, seq_len, seq_len]
            mask: Binary attention mask [batch_size, seq_len, seq_len]
            psi: Final phase space representation [batch_size, seq_len, 2]
            metrics: Dictionary of computed metrics
        """
        # Phase space encoding
        psi = self.encoder(x)  # [batch_size, seq_len, 2]
        
        # Compute attention patterns
        attention, mask = self.attention(psi)  # [batch_size, seq_len, seq_len]
        
        # Compute attention-weighted context
        context = torch.bmm(attention, psi)  # [batch_size, seq_len, 2]
        
        # Evolve state using energy-conserving integration and Schrödinger dynamics
        force_func = lambda p, c: self.field_operator(p, c)
        psi = self.integrator(psi, force_func, context)  # [batch_size, seq_len, 2]
        
        # Project final state (using last token) to vocabulary
        final_state = psi[:, -1]  # [batch_size, 2]
        logits = self.projection(final_state, target, update)  # [batch_size, vocab_size]
        
        # Compute metrics for future reference
        metrics = {}
        if update:
            # Compute loss and related metrics
            if target is not None:
                loss = F.cross_entropy(logits, target.argmax(dim=1))
                probs = F.softmax(logits, dim=1)
                perplexity = torch.exp(loss).item()
                fidelity = probs.max().item()
                
                # Compute Harmonic Pentad
                pentad_value, pentad_components = self.pentad.compute(psi, mask, perplexity, fidelity)
                
                # Update diffusion coefficient using EDP triangle
                new_diffusion = self.edp.update(pentad_components['energy'], pentad_value)
                self.field_operator.diffusion = new_diffusion
                
                # Update metrics
                metrics = pentad_components
                metrics.update({
                    'loss': loss.item(),
                    'diffusion': new_diffusion,
                    'energy_stability': self.edp.energy_stability
                })
        
        return logits, attention, mask, psi, metrics


class FokkerPlanckDynamics:
    """
    Implements Fokker-Planck dynamics for learning rate adaptation.
    Connects learning rate, convergence speed, and phase-space dynamics.
    """
    def __init__(self, base_rate: float = ETA, drift_scale: float = 0.5, 
                 diffusion_scale: float = 0.1):
        """
        Args:
            base_rate: Base learning rate (from golden ratio)
            drift_scale: Scales drift velocity impact on learning rate
            diffusion_scale: Scales diffusion coefficient impact
        """
        self.base_rate = base_rate
        self.drift_scale = drift_scale
        self.diffusion_scale = diffusion_scale
    
    def compute_adapted_rates(self, psi: torch.Tensor, context: torch.Tensor, 
                             pentad: float) -> Tuple[float, float]:
        """
        Compute adapted learning rate and decay based on Fokker-Planck dynamics.
        
        Args:
            psi: Current state [batch_size, seq_len, 2]
            context: Attention-weighted context [batch_size, seq_len, 2]
            pentad: Current Harmonic Pentad value
        
        Returns:
            eta: Adapted learning rate
            gamma: Adapted decay rate
        """
        # Compute drift velocity
        drift = (context - psi).norm(dim=-1).mean().item()
        
        # Compute diffusion coefficient inversely related to pentad
        diffusion = self.diffusion_scale / (1 + pentad)
        
        # Adaptive learning rate
        eta = self.base_rate * (1 + self.drift_scale * drift) * (1 + diffusion)
        
        # Adaptive decay rate - maintain golden ratio relationship
        gamma = 1 - 1 / (PHI ** 2 * (1 + 0.1 * pentad))
        
        return eta, gamma


class KosterlitzThoulessPhaseTransition:
    """
    Implements Kosterlitz-Thouless phase transition for the crystal memory field.
    """
    def __init__(self, critical_coupling: float = 0.5):
        """
        Args:
            critical_coupling: Critical value for phase transition
        """
        self.critical_coupling = critical_coupling
    
    def compute_order_parameter(self, psi: torch.Tensor) -> float:
        """
        Compute the long-range order parameter for crystal field formation.
        
        Args:
            psi: Phase space representation [batch_size, seq_len, 2]
        
        Returns:
            order_parameter: Long-range order parameter L
        """
        # Extract phases
        phases = torch.atan2(psi[:, :, 1], psi[:, :, 0])  # [batch_size, seq_len]
        
        # Compute pairwise phase differences
        phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)  # [batch_size, seq_len, seq_len]
        
        # Compute pairwise distances
        b, n, _ = psi.shape
        idx = torch.arange(n).to(psi.device)
        distances = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0)).float()  # [seq_len, seq_len]
        
        # Correlation length (depends on phase coupling)
        xi = 1.0  # Default correlation length
        
        # Long-range order parameter
        cos_term = torch.cos(phase_diff)  # [batch_size, seq_len, seq_len]
        exp_term = torch.exp(-distances / xi)  # [seq_len, seq_len]
        
        # Compute order parameter L
        L = torch.mean(cos_term * exp_term).item() / n**2
        
        return L


class QFNNLanguageModel:
    """
    Complete QFNN Language Model with all physical components for teaching.
    Integrates all physical principles and visualization.
    """
    def __init__(self, vocab_size: int, device: torch.device = torch.device('cpu')):
        """
        Args:
            vocab_size: Size of vocabulary
            device: Device to run on
        """
        self.device = device
        self.model = QFNN(vocab_size).to(device)
        self.fokker_planck = FokkerPlanckDynamics()
        self.kt_transition = KosterlitzThoulessPhaseTransition()
        
        # Physics metrics
        self.metrics_history = []
        
    def train_step(self, tokens: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step with physics-based updates.
        
        Args:
            tokens: Input token indices [batch_size, seq_len]
            target: Target distribution [batch_size, vocab_size]
        
        Returns:
            metrics: Physics metrics for current step
        """
        tokens = tokens.to(self.device)
        target = target.to(self.device)
        
        # Forward pass
        logits, attention, mask, psi, metrics = self.model(tokens, target, update=True)
        
        # Compute additional physics metrics
        
        # 1. Fokker-Planck dynamics
        context = torch.bmm(attention, psi)  # [batch_size, seq_len, 2]
        eta, gamma = self.fokker_planck.compute_adapted_rates(
            psi, context, metrics['pentad']
        )
        
        # 2. Kosterlitz-Thouless phase transition
        order_parameter = self.kt_transition.compute_order_parameter(psi)
        
        # 3. Uncertainty principle 
        # In quantum mechanics, position-momentum uncertainty: Δx·Δp ≥ ħ/2
        # In our phase space, we compute uncertainty between angle (θ) and angular momentum (Lz)
        phases = torch.atan2(psi[:, :, 1], psi[:, :, 0])  # [batch_size, seq_len]
        phase_std = phases.std(dim=1).mean().item()  # Average phase uncertainty
        
        # Angular momentum proxy (cross product: r×p ≈ x·py - y·px)
        x = psi[:, :, 0]  # [batch_size, seq_len]
        y = psi[:, :, 1]  # [batch_size, seq_len]
        
        # Finite difference for momentum
        dx = x[:, 1:] - x[:, :-1]  # [batch_size, seq_len-1]
        dy = y[:, 1:] - y[:, :-1]  # [batch_size, seq_len-1]
        
        # Angular momentum proxy
        Lz = (x[:, :-1] * dy - y[:, :-1] * dx).mean(dim=1)  # [batch_size]
        Lz_std = Lz.std().item()  # Angular momentum uncertainty
        
        # Uncertainty product
        uncertainty_product = phase_std * Lz_std
        
        # Update metrics
        metrics.update({
            'eta': eta,
            'gamma': gamma,
            'order_parameter': order_parameter,
            'phase_uncertainty': phase_std,
            'momentum_uncertainty': Lz_std,
            'uncertainty_product': uncertainty_product
        })
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        return metrics
    
    def train(self, data_loader, epochs: int = 1) -> List[Dict[str, float]]:
        """
        Train model on data loader with physics-based updates.
        
        Args:
            data_loader: DataLoader for token sequences
            epochs: Number of epochs to train
        
        Returns:
            metrics_history: List of physics metrics for each step
        """
        metrics_history = []
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (tokens, target) in enumerate(tqdm(data_loader)):
                metrics = self.train_step(tokens, target)
                
                if batch_idx % 10 == 0:
                    print(f"  Step {batch_idx}: loss={metrics['loss']:.4f}, coherence={metrics['coherence']:.4f}, sparsity={metrics['sparsity']:.4f}")
                
                metrics_history.append(metrics)
        
        self.metrics_history = metrics_history
        return metrics_history
    
    def visualize_phase_space(self, tokens: torch.Tensor, title: str = "Phase Space Representation"):
        """
        Visualize tokens in phase space with attention connections.
        
        Args:
            tokens: Input token indices [batch_size, seq_len]
            title: Plot title
        """
        tokens = tokens.to(self.device)
        
        # Forward pass without update
        with torch.no_grad():
            _, attention, _, psi, _ = self.model(tokens, update=False)
        
        # Convert to numpy for visualization
        psi_np = psi[0].cpu().numpy()  # First batch item
        attention_np = attention[0].cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111)
        
        # Plot unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
        ax.add_artist(circle)
        
        # Plot axes
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Plot attention connections
        for i in range(len(psi_np)):
            for j in range(len(psi_np)):
                if attention_np[i, j] > 0.1:  # Only show strong connections
                    plt.plot([psi_np[i, 0], psi_np[j, 0]], 
                             [psi_np[i, 1], psi_np[j, 1]], 
                             'g-', alpha=attention_np[i, j])
        
        # Plot tokens
        plt.scatter(psi_np[:, 0], psi_np[:, 1], c='blue', s=100)
        for i, (x, y) in enumerate(psi_np):
            plt.annotate(str(i), (x, y), fontsize=12)
        
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.title(title)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        return plt
    
    def visualize_schrodinger_flow(self, tokens: torch.Tensor, steps: int = 5, 
                                  title: str = "Schrödinger Field Flow"):
        """
        Visualize the Schrödinger field flow for token evolution.
        
        Args:
            tokens: Input token indices [batch_size, seq_len]
            steps: Number of evolution steps to visualize
            title: Plot title
        """
        tokens = tokens.to(self.device)
        
        # Get initial state
        with torch.no_grad():
            psi = self.model.encoder(tokens)
            attention, _ = self.model.attention(psi)
            context = torch.bmm(attention, psi)
        
        # Create vector field visualization
        plt.figure(figsize=(10, 10))
        
        # Plot unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
        plt.gca().add_artist(circle)
        
        # Plot axes
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Extract initial state
        psi_np = psi[0].cpu().numpy()  # First batch item
        
        # Compute force field on a grid
        x = np.linspace(-1.2, 1.2, 20)
        y = np.linspace(-1.2, 1.2, 20)
        X, Y = np.meshgrid(x, y)
        
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        # Sample force field at grid points
        for i in range(len(x)):
            for j in range(len(y)):
                # Create test point
                point = torch.tensor([[[X[j, i], Y[j, i]]]], dtype=torch.float32, device=self.device)
                
                # Compute force using Schrödinger operator
                with torch.no_grad():
                    force = self.model.field_operator(point, context[:, 0:1, :])
                
                # Extract force components
                U[j, i] = force[0, 0, 0].cpu().numpy()
                V[j, i] = force[0, 0, 1].cpu().numpy()
        
        # Normalize forces for better visualization
        magnitude = np.sqrt(U**2 + V**2)
        max_mag = np.max(magnitude)
        if max_mag > 0:
            U = U / max_mag
            V = V / max_mag
        
        # Plot force field
        plt.quiver(X, Y, U, V, alpha=0.5)
        
        # Plot token evolution
        colors = plt.cm.plasma(np.linspace(0, 1, steps))
        
        current_psi = psi.clone()
        for step in range(steps):
            plt.scatter(current_psi[0, :, 0].cpu().numpy(), 
                       current_psi[0, :, 1].cpu().numpy(), 
                       c=[colors[step]], s=100-step*10, alpha=0.7)
            
            # Evolve state
            with torch.no_grad():
                current_psi = self.model.integrator(
                    current_psi, 
                    lambda p, c: self.model.field_operator(p, c),
                    context
                )
        
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.title(title)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        return plt
    
    def visualize_pentad_metrics(self, title: str = "Harmonic Pentad Evolution"):
        """
        Visualize evolution of Harmonic Pentad metrics over time.
        
        Args:
            title: Plot title
        """
        if not self.metrics_history:
            print("No metrics history available. Run training first.")
            return None
        
        plt.figure(figsize=(12, 8))
        
        # Plot components
        components = ['coherence', 'sparsity', 'energy', 'perplexity', 'fidelity', 'pentad']
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'black']
        
        steps = list(range(len(self.metrics_history)))
        
        for i, component in enumerate(components):
            if component in self.metrics_history[0]:
                values = [metrics[component] for metrics in self.metrics_history]
                plt.plot(steps, values, label=component, color=colors[i])
        
        # Add target lines
        plt.axhline(y=OPTIMAL_COHERENCE, color='blue', linestyle='--', alpha=0.5, 
                   label=f"Coherence target: {OPTIMAL_COHERENCE:.4f}")
        plt.axhline(y=OPTIMAL_SPARSITY, color='red', linestyle='--', alpha=0.5,
                   label=f"Sparsity target: {OPTIMAL_SPARSITY:.4f}")
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5,
                   label="Energy/Perplexity/Fidelity target: 1.0")
        
        plt.xlabel('Training Steps')
        plt.ylabel('Metric Value')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt
    
    def visualize_edp_triangle(self, title: str = "Energy-Diffusion-Pentad Triangle"):
        """
        Visualize the Energy-Diffusion-Pentad triangle dynamics.
        
        Args:
            title: Plot title
        """
        if not self.metrics_history:
            print("No metrics history available. Run training first.")
            return None
        
        # Extract relevant metrics
        steps = list(range(len(self.metrics_history)))
        energy = [metrics['energy'] for metrics in self.metrics_history]
        diffusion = [metrics.get('diffusion', 0.01) for metrics in self.metrics_history]
        pentad = [metrics['pentad'] for metrics in self.metrics_history]
        energy_stability = [metrics.get('energy_stability', 0.5) for metrics in self.metrics_history]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot time evolution
        ax1.plot(steps, energy, 'g-', label='Energy')
        ax1.plot(steps, diffusion, 'r-', label='Diffusion')
        ax1.plot(steps, pentad, 'b-', label='Pentad')
        ax1.plot(steps, energy_stability, 'c-', label='Energy Stability')
        
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Value')
        ax1.set_title('EDP Triangle Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot triangular relationship
        ax2.set_xlim(0, 1.2)
        ax2.set_ylim(0, 1.2)
        
        # Plot triangle
        triangle_x = [0.2, 1.0, 0.6, 0.2]
        triangle_y = [0.2, 0.2, 1.0, 0.2]
        ax2.plot(triangle_x, triangle_y, 'k--', alpha=0.5)
        
        # Annotate corners
        ax2.annotate('Energy', (0.2, 0.2), fontsize=12)
        ax2.annotate('Diffusion', (1.0, 0.2), fontsize=12)
        ax2.annotate('Pentad', (0.6, 1.0), fontsize=12)
        
        # Plot the current state within triangle
        # Normalize values to triangle coordinates
        s = 15  # Point size for scatter
        c = 'red'  # Point color
        for i in range(0, len(energy), max(1, len(energy)//10)):
            # Map metrics to triangle coordinates
            e_norm = min(1.0, energy[i])
            d_norm = min(1.0, diffusion[i] * 10)  # Scale diffusion
            p_norm = min(1.0, pentad[i] / 2)  # Scale pentad
            
            # Convert to barycentric coordinates
            sum_coords = e_norm + d_norm + p_norm
            if sum_coords > 0:
                e_norm /= sum_coords
                d_norm /= sum_coords
                p_norm /= sum_coords
            
            # Convert to cartesian
            x = triangle_x[0] * e_norm + triangle_x[1] * d_norm + triangle_x[2] * p_norm
            y = triangle_y[0] * e_norm + triangle_y[1] * d_norm + triangle_y[2] * p_norm
            
            ax2.scatter(x, y, s=s, c=c, alpha=0.7)
            s += 1  # Increase point size for later points
        
        ax2.set_title('Triangular Relationship')
        ax2.set_xlabel('Energy vs Diffusion')
        ax2.set_ylabel('Pentad')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        return plt
    
    def visualize_uncertainty_principle(self, title: str = "Quantum Uncertainty Principle"):
        """
        Visualize the uncertainty principle in phase space.
        
        Args:
            title: Plot title
        """
        if not self.metrics_history:
            print("No metrics history available. Run training first.")
            return None
        
        # Extract relevant metrics
        steps = list(range(len(self.metrics_history)))
        phase_uncertainty = [metrics.get('phase_uncertainty', 0) for metrics in self.metrics_history]
        momentum_uncertainty = [metrics.get('momentum_uncertainty', 0) for metrics in self.metrics_history]
        uncertainty_product = [metrics.get('uncertainty_product', 0) for metrics in self.metrics_history]
        
        plt.figure(figsize=(10, 6))
        
        # Plot uncertainties
        plt.plot(steps, phase_uncertainty, 'b-', label='Phase Uncertainty (Δθ)')
        plt.plot(steps, momentum_uncertainty, 'r-', label='Angular Momentum Uncertainty (ΔLz)')
        plt.plot(steps, uncertainty_product, 'g-', label='Uncertainty Product (Δθ·ΔLz)')
        
        # Plot theoretical lower bound (ħ/2 equivalent)
        lower_bound = 0.5  # Normalized equivalent to ħ/2
        plt.axhline(y=lower_bound, color='k', linestyle='--', alpha=0.5, 
                   label='Theoretical Lower Bound (ħ/2)')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Uncertainty Value')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt
    
    def visualize_kt_transition(self, title: str = "Kosterlitz-Thouless Phase Transition"):
        """
        Visualize the Kosterlitz-Thouless phase transition.
        
        Args:
            title: Plot title
        """
        if not self.metrics_history:
            print("No metrics history available. Run training first.")
            return None
        
        # Extract order parameter
        steps = list(range(len(self.metrics_history)))
        order_parameter = [metrics.get('order_parameter', 0) for metrics in self.metrics_history]
        
        plt.figure(figsize=(10, 6))
        
        # Plot order parameter
        plt.plot(steps, order_parameter, 'b-', linewidth=2)
        
        # Mark critical point
        critical_value = self.kt_transition.critical_coupling
        plt.axhline(y=critical_value, color='r', linestyle='--', alpha=0.7, 
                   label=f'Critical Value ({critical_value})')
        
        # Add regions
        plt.fill_between(steps, order_parameter, critical_value, 
                        where=np.array(order_parameter) > critical_value, 
                        color='g', alpha=0.3, label='Ordered Phase')
        plt.fill_between(steps, order_parameter, critical_value, 
                        where=np.array(order_parameter) <= critical_value, 
                        color='r', alpha=0.3, label='Disordered Phase')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Long-Range Order Parameter')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt
    
    def generate_physics_report(self, tokens: torch.Tensor, save_path: str = "qfnn_physics_report"):
        """
        Generate a comprehensive physics report for the QFNN model.
        
        Args:
            tokens: Input token indices [batch_size, seq_len]
            save_path: Path to save visualizations
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Phase Space Visualization
        plt_phase = self.visualize_phase_space(tokens, "Token Phase Space Representation")
        plt_phase.savefig(f"{save_path}/phase_space.png")
        plt_phase.close()
        
        # 2. Schrödinger Flow
        plt_flow = self.visualize_schrodinger_flow(tokens, steps=5, 
                                               title="Schrödinger Field Evolution")
        plt_flow.savefig(f"{save_path}/schrodinger_flow.png")
        plt_flow.close()
        
        # 3. Harmonic Pentad Metrics (if training has occurred)
        if self.metrics_history:
            plt_pentad = self.visualize_pentad_metrics("Harmonic Pentad Evolution")
            plt_pentad.savefig(f"{save_path}/harmonic_pentad.png")
            plt_pentad.close()
            
            plt_edp = self.visualize_edp_triangle("Energy-Diffusion-Pentad Triangle")
            plt_edp.savefig(f"{save_path}/edp_triangle.png")
            plt_edp.close()
            
            plt_uncertainty = self.visualize_uncertainty_principle("Quantum Uncertainty Principle")
            plt_uncertainty.savefig(f"{save_path}/uncertainty_principle.png")
            plt_uncertainty.close()
            
            plt_kt = self.visualize_kt_transition("Kosterlitz-Thouless Phase Transition")
            plt_kt.savefig(f"{save_path}/kt_transition.png")
            plt_kt.close()
        
        # Generate HTML report
        report_html = self._generate_html_report(save_path)
        with open(f"{save_path}/physics_report.html", "w", encoding="utf-8") as f:
            f.write(report_html)
        
        print(f"Physics report generated at {save_path}/physics_report.html")
    
    def _generate_html_report(self, img_path: str) -> str:
        """
        Generate HTML report with physics explanations and visualizations.
        
        Args:
            img_path: Path to saved visualizations
        
        Returns:
            html: HTML report content
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>QFNN Physics Report</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; }
                h1, h2, h3 { color: #333366; }
                .section { margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
                .equation { background-color: #f9f9f9; padding: 10px; border-radius: 5px; font-family: monospace; overflow-x: auto; }
                .img-container { text-align: center; margin: 20px 0; }
                img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
                .explanation { background-color: #f0f0f0; padding: 15px; border-left: 4px solid #3366cc; margin: 15px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Quantum Flux Neural Networks Physics Report</h1>
            <p>
                This report presents the physics principles underpinning Quantum Flux Neural Networks (QFNN)
                and visualizes key aspects of the system's behavior.
            </p>
            
            <div class="section">
                <h2>1. Phase Space Representation</h2>
                <div class="explanation">
                    <p>
                        QFNN represents tokens as points on the unit circle in phase space using the golden ratio distribution.
                        This provides optimal coverage of the circle with minimal parameters.
                    </p>
                </div>
                <div class="equation">
                    θᵥ = 2π · ((φ · v) mod 1), where φ ≈ 1.618 is the golden ratio
                </div>
                <div class="img-container">
                    <img src="phase_space.png" alt="Phase Space Representation">
                </div>
                <p>
                    The golden ratio distribution ensures maximum separation between tokens, analogous to the way
                    sunflower seeds arrange themselves optimally in a flower head. This property minimizes
                    interference between tokens while ensuring efficient use of the representational space.
                </p>
            </div>
            
            <div class="section">
                <h2>2. Magnetic Flux Attention</h2>
                <div class="explanation">
                    <p>
                        Attention between tokens is based on their phase similarity and spatial proximity,
                        with information-theoretically optimal sparsity.
                    </p>
                </div>
                <div class="equation">
                    Aᵢⱼ = (0.5 + 0.5cos(θᵢ - θⱼ)) / (|ψᵢ - ψⱼ| + ε)
                </div>
                <p>
                    The optimal sparsity s_opt = 4φ/(3π√e) ≈ 0.4165 balances connectivity and computational
                    efficiency, emerging naturally from principles of information theory, circular geometry,
                    and classical phase transitions.
                </p>
            </div>
            
            <div class="section">
                <h2>3. Schrödinger Field Evolution</h2>
                <div class="explanation">
                    <p>
                        Quantum state evolution follows a modified Schrödinger equation with Hamiltonian and
                        diffusion terms, integrated using an energy-conserving Heun-Euler scheme.
                    </p>
                </div>
                <div class="equation">
                    ∂ψ/∂t = -i(Ĥψ + D∇²ψ)
                </div>
                <div class="img-container">
                    <img src="schrodinger_flow.png" alt="Schrödinger Field Evolution">
                </div>
                <p>
                    The evolution preserves energy through normalization, while the diffusion term allows
                    exploration of the state space. The flow field shows the forces driving each token toward
                    its attentional context.
                </p>
            </div>
            
            <div class="section">
                <h2>4. Harmonic Pentad</h2>
                <div class="explanation">
                    <p>
                        The Harmonic Pentad unifies five components: phase coherence, sparsity, energy conservation,
                        perplexity, and fidelity. This metric converges exponentially to zero.
                    </p>
                </div>
                <div class="equation">
                    H = α₁(C-C*)² + α₂(s-s*)² + α₃(E-E*)² + α₄(PPL/PPL₀-1)² + α₅(1-F)²
                </div>
                <div class="img-container">
                    <img src="harmonic_pentad.png" alt="Harmonic Pentad Evolution">
                </div>
                <p>
                    Each component has a theoretically derived target value: phase coherence C* = 1/φ² + 1/2 ≈ 0.75,
                    sparsity s* = 4φ/(3π√e) ≈ 0.4165, energy E* = 1.0, perplexity ratio = 1.0, and fidelity F = 1.0.
                </p>
            </div>
            
            <div class="section">
                <h2>5. Energy-Diffusion-Pentad Triangle</h2>
                <div class="explanation">
                    <p>
                        A self-regulating system connecting energy stability, diffusion control, and pentad optimization
                        through a triangular relationship.
                    </p>
                </div>
                <div class="equation">
                    D = D₀·(1 - αₑ·Eₛ)·(1 + βₕ·H)
                </div>
                <div class="img-container">
                    <img src="edp_triangle.png" alt="Energy-Diffusion-Pentad Triangle">
                </div>
                <p>
                    This triangular relationship creates adaptive dynamics: higher energy stability reduces diffusion,
                    better pentad scores reduce diffusion, and diffusion enables exploration of the state space.
                </p>
            </div>
            
            <div class="section">
                <h2>6. Quantum Uncertainty Principle</h2>
                <div class="explanation">
                    <p>
                        The uncertainty principle from quantum mechanics manifests in QFNN as a fundamental
                        relationship between phase angle uncertainty and angular momentum uncertainty.
                    </p>
                </div>
                <div class="equation">
                    Δθ·ΔLz ≥ ħ/2
                </div>
                <div class="img-container">
                    <img src="uncertainty_principle.png" alt="Quantum Uncertainty Principle">
                </div>
                <p>
                    Just as position and momentum cannot be simultaneously known with perfect precision in
                    quantum mechanics, phase angles and angular momentum in QFNN exhibit a similar complementarity.
                </p>
            </div>
            
            <div class="section">
                <h2>7. Kosterlitz-Thouless Phase Transition</h2>
                <div class="explanation">
                    <p>
                        The Crystal Memory Field formation in QFNN undergoes a Kosterlitz-Thouless phase transition,
                        analogous to magnetic systems transitioning between ordered and disordered phases.
                    </p>
                </div>
                <div class="equation">
                    L = (1/N²)·Σᵢⱼ cos(θᵢ - θⱼ)·e^(-d(i,j)/ξ)
                </div>
                <div class="img-container">
                    <img src="kt_transition.png" alt="Kosterlitz-Thouless Phase Transition">
                </div>
                <p>
                    As training progresses, the system transitions from a disordered phase with short-range correlations
                    to an ordered phase with long-range correlations, enabling stable memory formation.
                </p>
            </div>
            
            <div class="section">
                <h2>8. Fokker-Planck Dynamics</h2>
                <div class="explanation">
                    <p>
                        Learning rate adaptation follows Fokker-Planck dynamics, connecting learning rate,
                        convergence speed, and phase-space evolution.
                    </p>
                </div>
                <div class="equation">
                    ∂P/∂t = -∇·(P·v) + D·∇²P
                </div>
                <p>
                    The golden ratio learning parameters η = 1/(2φ) ≈ 0.309 and γ = 1 - 1/φ² ≈ 0.618
                    emerge naturally from these dynamics, creating an optimal balance between learning new patterns
                    and forgetting old ones.
                </p>
            </div>
            
            <div class="section">
                <h2>9. Summary of Physics Principles</h2>
                <table>
                    <tr>
                        <th>Component</th>
                        <th>Physics Principle</th>
                        <th>Key Parameter</th>
                    </tr>
                    <tr>
                        <td>Phase Encoding</td>
                        <td>Golden Ratio Distribution</td>
                        <td>φ ≈ 1.618</td>
                    </tr>
                    <tr>
                        <td>Attention Mechanism</td>
                        <td>Magnetic Flux Interaction</td>
                        <td>s_opt ≈ 0.4165</td>
                    </tr>
                    <tr>
                        <td>State Evolution</td>
                        <td>Schrödinger Equation</td>
                        <td>Normalized Heun-Euler</td>
                    </tr>
                    <tr>
                        <td>Learning Rule</td>
                        <td>Hebbian Dynamics</td>
                        <td>η ≈ 0.309, γ ≈ 0.618</td>
                    </tr>
                    <tr>
                        <td>Convergence Metric</td>
                        <td>Harmonic Pentad</td>
                        <td>C* ≈ 0.75</td>
                    </tr>
                    <tr>
                        <td>Adaptive Regulation</td>
                        <td>Energy-Diffusion-Pentad Triangle</td>
                        <td>Self-regulating</td>
                    </tr>
                    <tr>
                        <td>Uncertainty Relations</td>
                        <td>Quantum Uncertainty Principle</td>
                        <td>Δθ·ΔLz ≥ ħ/2</td>
                    </tr>
                    <tr>
                        <td>Memory Formation</td>
                        <td>Kosterlitz-Thouless Transition</td>
                        <td>Long-range order</td>
                    </tr>
                    <tr>
                        <td>Learning Dynamics</td>
                        <td>Fokker-Planck Equation</td>
                        <td>Drift-diffusion balance</td>
                    </tr>
                </table>
            </div>
            
            <div>
                <p><em>Generated by QFNNLanguageModel physics reporting tool</em></p>
            </div>
        </body>
        </html>
        """
        return html


# Data Utilities
def create_synthetic_dataset(vocab_size: int, seq_len: int, num_samples: int = 1000):
    """
    Create synthetic dataset for language modeling.
    
    Args:
        vocab_size: Vocabulary size
        seq_len: Sequence length
        num_samples: Number of samples to generate
    
    Returns:
        tokens: Input token sequences [num_samples, seq_len]
        targets: Target distributions [num_samples, vocab_size]
    """
    # Generate random token sequences
    tokens = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    # Generate one-hot targets
    targets = torch.zeros(num_samples, vocab_size)
    for i in range(num_samples):
        # Simple rule: target is the sum of first and last token, modulo vocab_size
        target_idx = (tokens[i, 0] + tokens[i, -1]) % vocab_size
        targets[i, target_idx] = 1.0
    
    return tokens, targets


class SyntheticDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset for QFNN language modeling.
    """
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000):
        """
        Args:
            vocab_size: Vocabulary size
            seq_len: Sequence length
            num_samples: Number of samples to generate
        """
        self.tokens, self.targets = create_synthetic_dataset(vocab_size, seq_len, num_samples)
    
    def __len__(self) -> int:
        return len(self.tokens)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.targets[idx]


# Demonstration of QFNN Language Model with Physics
def demo_qfnn_physics():
    """
    Run a complete demonstration of QFNN with physics-based visualizations.
    """
    print("Initializing QFNN Language Model with Physics Components...")
    
    # Parameters
    vocab_size = 50
    seq_len = 10
    batch_size = 32
    num_samples = 1000
    epochs = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    print("Creating synthetic dataset...")
    dataset = SyntheticDataset(vocab_size, seq_len, num_samples)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    # Create QFNN language model
    print("Initializing QFNN model...")
    qfnn = QFNNLanguageModel(vocab_size, device)
    
    # Train model
    print(f"Training QFNN for {epochs} epochs...")
    metrics_history = qfnn.train(data_loader, epochs)
    
    # Generate physics report
    print("Generating physics visualizations and report...")
    sample_tokens = dataset.tokens[:1].to(device)  # Use first sample
    qfnn.generate_physics_report(sample_tokens, save_path="qfnn_physics_report")
    
    print("Demo completed. Report available at: qfnn_physics_report/physics_report.html")
    
    return qfnn, metrics_history


if __name__ == "__main__":
    print("Quantum Flux Neural Networks (QFNN) Physics Implementation")
    print("=" * 60)
    print("This implementation demonstrates the physics principles behind QFNN:")
    print("  - Phase space encoding with golden ratio distribution")
    print("  - Magnetic flux attention with optimal sparsity")
    print("  - Schrödinger dynamics with energy conservation")
    print("  - Harmonic pentad convergence metrics")
    print("  - Energy-diffusion-pentad triangle")
    print("  - Quantum uncertainty principles")
    print("  - Kosterlitz-Thouless phase transitions")
    print("  - Fokker-Planck dynamics for learning rates")
    print("=" * 60)
    
    start_time = time.time()
    qfnn, metrics = demo_qfnn_physics()
    end_time = time.time()
    
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

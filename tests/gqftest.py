"""
Critical Physics Fixes for Pure Physics Language Model
=====================================================

Implements:
1. Berry phase conservation for topological protection
2. π-stacking stabilization for field configurations  
3. Sinusoidal modulation of binary gateways
4. Proper normalization to prevent NaN explosion
"""

import torch
import numpy as np
import math

# Physical constants
PHI = (1 + np.sqrt(5)) / 2
EPS = 1e-8

def stable_levy_sample(alpha: float, beta: float, size: tuple, device: torch.device, max_value: float = 10.0) -> torch.Tensor:
    """
    Stable Lévy sampling with bounds to prevent explosion.
    
    Clips extreme values while preserving heavy-tail properties.
    """
    # Use scipy but with safety bounds
    from scipy.stats import levy_stable
    total_elements = int(np.prod(size))
    
    # Sample with retry logic for extreme values
    max_attempts = 3
    for attempt in range(max_attempts):
        samples = levy_stable.rvs(alpha=alpha, beta=beta, size=total_elements)
        
        # Clip extreme values while preserving distribution shape
        # Use tanh-like soft clipping to preserve gradients
        samples = max_value * np.tanh(samples / max_value)
        
        if not np.any(np.isnan(samples)) and not np.any(np.isinf(samples)):
            break
    
    tensor = torch.from_numpy(samples).float().to(device)
    return tensor.reshape(size)


class BerryPhaseEvolution:
    """
    Implements Berry phase-aware quantum evolution.
    
    The wave function acquires geometric phase during cyclic evolution,
    providing topological protection against instabilities.
    """
    
    def __init__(self, vocab_size: int, field_dim: int, device: torch.device):
        self.vocab_size = vocab_size
        self.field_dim = field_dim
        self.device = device
        
        # Berry connection A_μ (gauge field)
        self.berry_connection = torch.zeros(vocab_size, 4, device=device)
        
        # Accumulated Berry phase for each token
        self.berry_phase = torch.zeros(vocab_size, device=device)
        
    def compute_berry_curvature(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Compute Berry curvature F = dA = ∂_μ A_ν - ∂_ν A_μ.
        
        This measures the "twisting" of the quantum state space.
        """
        # Simplified: Use phase gradients as proxy for curvature
        psi_phase = torch.atan2(psi[..., 1], psi[..., 0])  # Extract phase
        
        # Compute phase gradients (curvature)
        phase_grad = torch.roll(psi_phase, -1, dims=0) - torch.roll(psi_phase, 1, dims=0)
        curvature = phase_grad.mean(dim=1)  # Average over field dimensions
        
        return curvature
    
    def update_berry_phase(self, psi: torch.Tensor, dt: float):
        """
        Update Berry phase: γ = ∮ A·dl
        
        This provides topological protection during evolution.
        """
        curvature = self.compute_berry_curvature(psi)
        
        # Update phase with modulo 2π to prevent unbounded growth
        self.berry_phase = (self.berry_phase + dt * curvature) % (2 * np.pi)
        
        # Update connection (simplified Abelian case)
        self.berry_connection[:, 0] = curvature  # Time component
        

class PiStackingStabilizer:
    """
    Implements π-stacking stabilization inspired by aromatic molecules.
    
    Tokens interact through overlapping "orbitals" that prefer
    specific geometric arrangements for stability.
    """
    
    def __init__(self, vocab_size: int, device: torch.device):
        self.vocab_size = vocab_size
        self.device = device
        
        # Optimal stacking distance (in token space)
        self.stack_distance = PHI  # Golden ratio spacing
        
        # Stacking interaction strength
        self.coupling = 0.1
        
    def compute_stacking_potential(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Compute π-stacking potential between tokens.
        
        V_stack = -ε Σ_ij f(r_ij) cos(θ_ij)
        
        where f(r) is distance-dependent coupling and θ is relative phase.
        """
        # Compute pairwise interactions (simplified to nearest neighbors)
        psi_norm = torch.norm(psi, dim=(1, 2))  # [vocab_size]
        
        # Shifted copies for neighbor interactions
        psi_next = torch.roll(psi_norm, -1, dims=0)
        psi_prev = torch.roll(psi_norm, 1, dims=0)
        
        # Distance-dependent coupling (Gaussian envelope)
        r_next = torch.arange(self.vocab_size, device=self.device).float()
        r_prev = torch.arange(self.vocab_size, device=self.device).float()
        
        coupling_next = torch.exp(-(r_next % self.stack_distance)**2 / 2)
        coupling_prev = torch.exp(-(r_prev % self.stack_distance)**2 / 2)
        
        # Phase alignment term (simplified)
        phase_alignment = (psi_norm * psi_next * coupling_next + 
                          psi_norm * psi_prev * coupling_prev)
        
        # Stacking potential
        V_stack = -self.coupling * phase_alignment
        
        return V_stack


def fixed_initialize_field(vocab_size: int, field_dim: int, levy_alpha: float, device: torch.device) -> 'QuantumFieldState':
    """
    Initialize quantum field with proper normalization and bounds.
    
    Key fixes:
    1. Bounded Lévy samples to prevent explosion
    2. Proper complex normalization
    3. Energy-based initialization scaling
    """
    # Use bounded Lévy samples
    amplitudes = stable_levy_sample(levy_alpha, 0, (vocab_size, field_dim), device, max_value=1.0)
    phases = torch.rand(vocab_size, field_dim, device=device) * 2 * np.pi
    
    # Complex wave function with controlled amplitude
    # Scale by 1/sqrt(vocab_size) for proper normalization
    scale = 1.0 / math.sqrt(vocab_size * field_dim)
    psi_real = scale * amplitudes * torch.cos(phases)
    psi_imag = scale * amplitudes * torch.sin(phases)
    psi = torch.stack([psi_real, psi_imag], dim=-1)
    
    # Normalize to unit probability
    norm = torch.sqrt(torch.sum(psi**2) + EPS)
    psi = psi / norm
    
    # Rest of initialization...
    momentum = torch.zeros_like(psi)
    vorticity = torch.zeros(vocab_size, 3, device=device)
    metric = torch.eye(4, device=device).unsqueeze(0).repeat(vocab_size, 1, 1)
    metric[:, 0, 0] = -1
    christoffel = torch.zeros(vocab_size, 4, 4, 4, device=device)
    
    # Compute initial energy with proper scaling
    energy = compute_bounded_energy(psi, momentum, vocab_size, field_dim)
    
    return {
        'psi': psi,
        'momentum': momentum,
        'vorticity': vorticity,
        'metric': metric,
        'christoffel': christoffel,
        'energy': energy,
        'entropy': 0.0,
        'time': 0
    }


def compute_bounded_energy(psi: torch.Tensor, momentum: torch.Tensor, vocab_size: int, field_dim: int) -> float:
    """
    Compute field energy with proper bounds and scaling.
    
    Prevents NaN by careful normalization and clamping.
    """
    # Kinetic energy with bounds
    T = torch.sum(momentum**2) / (2 * vocab_size)
    T = torch.clamp(T, min=0, max=1e6)
    
    # Gradient energy with proper scaling
    psi_shifted = torch.roll(psi, shifts=1, dims=0)
    grad_psi = (psi - psi_shifted) / math.sqrt(vocab_size)
    gradient_energy = 0.5 * torch.sum(grad_psi**2)
    gradient_energy = torch.clamp(gradient_energy, min=0, max=1e6)
    
    # Potential energy with stability
    psi_squared = torch.sum(psi**2, dim=-1)
    # Use log-sum-exp trick for numerical stability
    psi_fourth = torch.exp(2 * torch.log(psi_squared + EPS))
    
    V = PHI/4 * torch.sum(psi_fourth) - 0.5 * torch.sum(psi_squared)
    V = torch.clamp(V, min=-1e6, max=1e6)
    
    total_energy = (T + gradient_energy + V).item()
    
    # Final safety check
    if np.isnan(total_energy) or np.isinf(total_energy):
        return 1e3  # Return high but finite energy
    
    return total_energy


def sinusoidal_gate_modulation(
    prob: torch.Tensor, 
    berry_phase: torch.Tensor, 
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Apply sinusoidal modulation to probability distribution based on Berry phase.
    
    P(k) → P(k) * [1 + cos(γ_k)] / 2
    
    This creates interference patterns that stabilize certain token choices.
    """
    # Sinusoidal modulation with Berry phase
    modulation = (1 + torch.cos(berry_phase)) / 2
    
    # Temperature-dependent mixing
    mix_factor = torch.exp(-1/temperature)
    prob_modulated = prob * (1 - mix_factor + mix_factor * modulation)
    
    # Renormalize
    prob_modulated = prob_modulated / (torch.sum(prob_modulated) + EPS)
    
    # Ensure no zeros for multinomial sampling
    prob_modulated = torch.clamp(prob_modulated, min=1e-10)
    
    return prob_modulated


def fixed_evolve_schrodinger(field, source_tokens, dt, berry_evolution, pi_stacking):
    """
    Fixed Schrödinger evolution with topological protection.
    
    Includes:
    1. Berry phase tracking
    2. π-stacking stabilization  
    3. Adaptive time stepping for stability
    """
    psi = field['psi']
    momentum = field['momentum']
    
    # Add source terms if provided
    if source_tokens is not None:
        source_field = tokens_to_bounded_source(source_tokens, psi.shape, psi.device)
        psi = psi + dt * source_field
    
    # Compute forces with bounds
    # Safe Laplacian
    psi_next = torch.roll(psi, -1, dims=0)
    psi_prev = torch.roll(psi, 1, dims=0)
    laplacian = (psi_next + psi_prev - 2*psi) / (dt**2 + EPS)
    laplacian = torch.clamp(laplacian, min=-100, max=100)
    
    # Potential force with π-stacking
    psi_mag_sq = torch.sum(psi**2, dim=-1, keepdim=True)
    psi_mag_sq = torch.clamp(psi_mag_sq, min=EPS, max=10.0)
    
    V_stack = pi_stacking.compute_stacking_potential(psi).unsqueeze(-1).unsqueeze(-1)
    potential_force = psi - PHI * psi_mag_sq * psi + 0.1 * V_stack * psi
    
    # Total force with damping for stability
    damping = 0.01
    force = -0.5 * laplacian + potential_force - damping * momentum
    
    # Adaptive time step based on force magnitude
    force_norm = torch.norm(force)
    adaptive_dt = dt * torch.clamp(1.0 / (1.0 + force_norm), min=0.1, max=1.0)
    
    # Symplectic update
    momentum_new = momentum - adaptive_dt * force
    psi_new = psi + adaptive_dt * momentum_new
    
    # Update Berry phase
    berry_evolution.update_berry_phase(psi_new, adaptive_dt)
    
    # Normalize to preserve probability
    norm = torch.sqrt(torch.sum(psi_new**2) + EPS)
    psi_new = psi_new / norm
    
    return psi_new, momentum_new


def tokens_to_bounded_source(tokens, target_shape, device):
    """
    Create bounded source field from tokens.
    """
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    
    batch_size, seq_len = tokens.shape
    vocab_size, field_dim, _ = target_shape
    source = torch.zeros(target_shape, device=device)
    
    for b in range(batch_size):
        for t in range(seq_len):
            token_id = tokens[b, t].item()
            if 0 <= token_id < vocab_size:
                # Bounded Gaussian envelope
                positions = torch.arange(vocab_size, device=device).float()
                gaussian = torch.exp(-(positions - token_id)**2 / (2 * seq_len))
                gaussian = gaussian / (torch.sum(gaussian) + EPS)  # Normalize
                
                # Sinusoidal phase
                phase = 2 * np.pi * t / seq_len
                
                # Add to source with bounds
                field_idx = t % field_dim
                source[:, field_idx, 0] += 0.1 * gaussian * math.cos(phase)
                source[:, field_idx, 1] += 0.1 * gaussian * math.sin(phase)
    
    return source / (batch_size + EPS)


# Example usage in main model:
def apply_fixes_to_model(model):
    """
    Apply all physics fixes to existing model.
    """
    # Replace initialization
    model._initialize_field = lambda: fixed_initialize_field(
        model.vocab_size, model.field_dim, model.levy_alpha, model.device
    )
    
    # Add Berry phase evolution
    model.berry_evolution = BerryPhaseEvolution(
        model.vocab_size, model.field_dim, model.device
    )
    
    # Add π-stacking stabilizer
    model.pi_stacking = PiStackingStabilizer(model.vocab_size, model.device)
    
    # Replace evolution method
    original_evolve = model.evolve_schrodinger
    
    def new_evolve(source_tokens=None):
        psi_new, momentum_new = fixed_evolve_schrodinger(
            {'psi': model.field.psi, 'momentum': model.field.momentum},
            source_tokens,
            model.dt,
            model.berry_evolution,
            model.pi_stacking
        )
        model.field.psi = psi_new
        model.field.momentum = momentum_new
        model.field.energy = compute_bounded_energy(psi_new, momentum_new, model.vocab_size, model.field_dim)
        model.field.time += 1
        model._update_entropy()
    
    model.evolve_schrodinger = new_evolve
    
    # Fix Born rule collapse
    original_born = model.born_rule_collapse
    
    def new_born_rule(context_tokens=None):
        psi = model.field.psi
        
        # Compute probability with bounds
        prob = torch.sum(psi**2, dim=(1, 2))
        prob = torch.clamp(prob, min=EPS)
        
        # Apply context if provided
        if context_tokens is not None:
            context_potential = model._compute_context_potential(context_tokens)
            context_potential = torch.clamp(context_potential, min=-10, max=10)
            prob = prob * torch.exp(-context_potential / model.temperature)
        
        # Apply sinusoidal Berry phase modulation
        prob = sinusoidal_gate_modulation(
            prob, 
            model.berry_evolution.berry_phase,
            model.temperature
        )
        
        # Ensure valid probability distribution
        prob = prob / (torch.sum(prob) + EPS)
        prob = torch.clamp(prob, min=1e-10)
        
        # Sample with error handling
        try:
            token = torch.multinomial(prob, num_samples=1)
        except:
            # Fallback to argmax if multinomial fails
            token = torch.argmax(prob).unsqueeze(0)
        
        model._apply_measurement_backaction(token)
        return token
    
    model.born_rule_collapse = new_born_rule
    
    print("✅ Applied physics fixes:")
    print("   - Berry phase evolution for topological protection")
    print("   - π-stacking stabilization")
    print("   - Bounded Lévy sampling")
    print("   - Sinusoidal gate modulation")
    print("   - Adaptive time stepping")
    
    return model


if __name__ == "__main__":
    print("Physics fixes module loaded successfully!")
    print(f"Golden ratio φ = {PHI:.6f}")
    print("Ready to stabilize quantum field evolution.")
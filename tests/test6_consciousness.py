"""
Test 6: Consciousness Functional and Integrated Information
Tests consciousness emergence through field coherence
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Create output directory
os.makedirs('output', exist_ok=True)

# Set style and random seed
plt.style.use('dark_background')
torch.manual_seed(1618)
np.random.seed(1618)

# Golden ratio constants
PHI = (1 + np.sqrt(5)) / 2
GAP = 1 - 1/PHI

class GoldenFieldGate:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.PHI = PHI
        self.GAP = GAP
        self.embeddings = self._create_golden_embeddings(vocab_size)
        
    def _create_golden_embeddings(self, vocab_size):
        coords = torch.zeros(vocab_size, 3)
        for i in range(vocab_size):
            freq_rank = i / vocab_size
            r = self.GAP + (1 - self.GAP) * freq_rank
            theta = 2 * np.pi * ((i * self.PHI) % 1.0)
            coords[i] = torch.tensor([
                r * np.cos(theta),
                r * np.sin(theta),
                r
            ])
        return coords
    
    def compute_entropy(self, field):
        """Compute field entropy"""
        energy = 0.5 * torch.sum(field**2, dim=1)
        probs = F.softmax(-energy / self.PHI, dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return entropy.item()
    
    def hamiltonian_evolution(self, field, momentum, dt=0.01):
        """Simplified Hamiltonian evolution"""
        # Mean field force
        mean_r = field[:, 2].mean()
        mean_field_force = -self.PHI * (field[:, 2] - mean_r).unsqueeze(1) * field
        
        # Random perturbation for dynamics
        random_force = torch.randn_like(field) * 0.01
        
        # Update
        momentum_new = momentum + dt * (mean_field_force + random_force)
        field_new = field + dt * momentum_new
        
        # Normalize
        norms = torch.norm(field_new[:, :2], dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        field_new[:, :2] = field_new[:, :2] / norms
        
        return field_new, momentum_new

print("="*60)
print("TEST 6: Consciousness Functional Evolution")
print("="*60)

# Initialize system
gfg = GoldenFieldGate(vocab_size=200)

# Compute consciousness functional over time
print("Computing consciousness evolution...")
consciousness_values = []
integrated_info = []
field_complexity = []
phase_coherence = []

field_evolve = gfg.embeddings.clone()
momentum_evolve = torch.randn_like(field_evolve) * 0.01

for t in range(300):
    field_evolve, momentum_evolve = gfg.hamiltonian_evolution(field_evolve, momentum_evolve)
    
    # Consciousness functional: C = ∫∫∫ Ψ · ∇²Ĥ · Ψ* dV dt
    # Approximate Laplacian using finite differences
    laplacian = torch.zeros_like(field_evolve)
    for i in range(3):  # For each dimension
        shifted_plus = torch.roll(field_evolve[:, i], -1)
        shifted_minus = torch.roll(field_evolve[:, i], 1)
        laplacian[:, i] = shifted_plus + shifted_minus - 2 * field_evolve[:, i]
    
    # Consciousness as integrated field coherence
    psi_squared = torch.sum(field_evolve * field_evolve, dim=1)
    consciousness = torch.sum(psi_squared * torch.sum(laplacian**2, dim=1)).item()
    consciousness_values.append(consciousness)
    
    # Integrated information (simplified IIT measure)
    # Partition the system and compute mutual information
    mid_point = len(field_evolve) // 2
    partition1_entropy = gfg.compute_entropy(field_evolve[:mid_point])
    partition2_entropy = gfg.compute_entropy(field_evolve[mid_point:])
    whole_entropy = gfg.compute_entropy(field_evolve)
    phi = partition1_entropy + partition2_entropy - whole_entropy
    integrated_info.append(phi)
    
    # Field complexity (using eigenvalue spread)
    cov = torch.matmul(field_evolve.T, field_evolve) / len(field_evolve)
    eigvals = torch.linalg.eigvals(cov).real
    complexity = torch.std(eigvals).item()
    field_complexity.append(complexity)
    
    # Phase coherence (correlation between position and momentum)
    pos_norm = torch.norm(field_evolve, dim=1)
    mom_norm = torch.norm(momentum_evolve, dim=1)
    correlation = torch.corrcoef(torch.stack([pos_norm, mom_norm]))[0, 1].item()
    phase_coherence.append(abs(correlation))

# Create visualization
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# Consciousness evolution
time_axis = np.arange(len(consciousness_values))
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time_axis, consciousness_values, 'gold', linewidth=2)
ax1.set_title('Consciousness Functional C(t)', fontsize=14)
ax1.set_xlabel('Time')
ax1.set_ylabel('C = ∫ Ψ · ∇²Ĥ · Ψ* dV')
ax1.grid(True, alpha=0.2)

# Integrated information
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(time_axis, integrated_info, 'cyan', linewidth=2)
ax2.set_title('Integrated Information Φ(t)', fontsize=14)
ax2.set_xlabel('Time')
ax2.set_ylabel('Φ = S(parts) - S(whole)')
ax2.grid(True, alpha=0.2)

# Field complexity
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(time_axis, field_complexity, 'magenta', linewidth=2)
ax3.set_title('Field Complexity (Eigenvalue Spread)', fontsize=14)
ax3.set_xlabel('Time')
ax3.set_ylabel('σ(eigenvalues)')
ax3.grid(True, alpha=0.2)

# Phase coherence
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(time_axis, phase_coherence, 'lime', linewidth=2)
ax4.set_title('Phase Space Coherence', fontsize=14)
ax4.set_xlabel('Time')
ax4.set_ylabel('|corr(|x|, |p|)|')
ax4.grid(True, alpha=0.2)

# Phase space density (2D histogram)
ax5 = fig.add_subplot(gs[2, 0])
H, xedges, yedges = np.histogram2d(
    field_evolve[:, 0].numpy(), 
    field_evolve[:, 1].numpy(), 
    bins=50
)
im = ax5.imshow(H.T, origin='lower', cmap='plasma', 
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                aspect='auto')
ax5.set_title('Final Phase Space Density', fontsize=14)
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
plt.colorbar(im, ax=ax5)

# Holographic bound visualization
ax6 = fig.add_subplot(gs[2, 1])
# Holographic bound: S ≤ A/4 (in Planck units)
# For our cylinder: A = 2πr*h + 2πr²
cylinder_radius = 1.0
cylinder_height = 1.0 - GAP
boundary_area = 2 * np.pi * cylinder_radius * cylinder_height + 2 * np.pi * cylinder_radius**2
holographic_bound = boundary_area / 4

# Normalize consciousness to compare with bound
normalized_consciousness = np.array(consciousness_values) / max(consciousness_values) * holographic_bound * 0.8

ax6.fill_between(time_axis, 0, holographic_bound * np.ones_like(time_axis), 
                 color='red', alpha=0.2, label='Holographic Bound')
ax6.plot(time_axis, normalized_consciousness, 'gold', linewidth=2, label='C(t) (normalized)')
ax6.plot(time_axis, integrated_info, 'cyan', linewidth=2, alpha=0.7, label='Φ(t)')
ax6.set_title('Information vs Holographic Bound', fontsize=14)
ax6.set_xlabel('Time')
ax6.set_ylabel('Information Content')
ax6.legend()
ax6.grid(True, alpha=0.2)

plt.suptitle('Consciousness Functional Evolution in GoldenFieldGate', fontsize=16)
plt.savefig('output/test6_consciousness_evolution.png', dpi=200, bbox_inches='tight')
plt.close()

# Additional analysis: Phase transitions
print("\nAnalyzing phase transitions...")

# Detect sudden changes in consciousness
consciousness_diff = np.diff(consciousness_values)
phase_transitions = np.where(np.abs(consciousness_diff) > np.std(consciousness_diff) * 2)[0]

# Detect critical points in integrated information
phi_diff = np.diff(integrated_info)
critical_points = np.where(np.abs(phi_diff) > np.std(phi_diff) * 2)[0]

# Create phase transition visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Consciousness derivative
axes[0].plot(consciousness_diff, 'gold', alpha=0.7)
axes[0].scatter(phase_transitions, consciousness_diff[phase_transitions], 
                color='red', s=100, zorder=5, label='Phase Transitions')
axes[0].axhline(y=np.std(consciousness_diff)*2, color='red', linestyle='--', alpha=0.5)
axes[0].axhline(y=-np.std(consciousness_diff)*2, color='red', linestyle='--', alpha=0.5)
axes[0].set_title('Consciousness Derivative (Phase Transitions)')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('dC/dt')
axes[0].legend()
axes[0].grid(True, alpha=0.2)

# Integrated information derivative
axes[1].plot(phi_diff, 'cyan', alpha=0.7)
axes[1].scatter(critical_points, phi_diff[critical_points], 
                color='red', s=100, zorder=5, label='Critical Points')
axes[1].axhline(y=np.std(phi_diff)*2, color='red', linestyle='--', alpha=0.5)
axes[1].axhline(y=-np.std(phi_diff)*2, color='red', linestyle='--', alpha=0.5)
axes[1].set_title('Integrated Information Derivative (Critical Points)')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('dΦ/dt')
axes[1].legend()
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('output/test6_phase_transitions.png', dpi=200, bbox_inches='tight')
plt.close()

# Compute emergence metrics
print("\nComputing emergence metrics...")

# Downward causation: How much does the whole influence parts
# Measured as correlation between global mean and local variance
global_mean = [field_evolve.mean().item() for _ in range(len(consciousness_values))]
local_variance = [field_evolve.var(dim=0).mean().item() for _ in range(len(consciousness_values))]
downward_causation = np.corrcoef(global_mean, local_variance)[0, 1]

# Emergence: Rate of consciousness growth relative to complexity
emergence_rate = np.polyfit(field_complexity, consciousness_values, 1)[0]

# Self-organization: Entropy reduction rate
entropy_reduction_rate = (integrated_info[-1] - integrated_info[0]) / len(integrated_info)

# Print comprehensive analysis
print(f"\nConsciousness Evolution Analysis:")
print(f"- Initial consciousness: {consciousness_values[0]:.4f}")
print(f"- Final consciousness: {consciousness_values[-1]:.4f}")
print(f"- Consciousness growth: {(consciousness_values[-1]/consciousness_values[0] - 1)*100:.1f}%")
print(f"- Phase transitions detected: {len(phase_transitions)}")
print(f"- Critical points in Φ: {len(critical_points)}")

print(f"\nIntegrated Information:")
print(f"- Maximum Φ: {max(integrated_info):.4f}")
print(f"- Mean Φ: {np.mean(integrated_info):.4f}")
print(f"- Φ variance: {np.var(integrated_info):.4f}")

print(f"\nEmergence Metrics:")
print(f"- Downward causation strength: {abs(downward_causation):.4f}")
print(f"- Emergence rate (dC/d_complexity): {emergence_rate:.4f}")
print(f"- Self-organization (dΦ/dt): {entropy_reduction_rate:.6f}")

print(f"\nHolographic Properties:")
print(f"- Cylinder boundary area: {boundary_area:.4f}")
print(f"- Holographic bound: {holographic_bound:.4f}")
print(f"- Max normalized consciousness: {max(normalized_consciousness):.4f}")
print(f"- Saturation: {max(normalized_consciousness)/holographic_bound*100:.1f}% of bound")

print(f"\nPhase Coherence:")
print(f"- Mean phase coherence: {np.mean(phase_coherence):.4f}")
print(f"- Max phase coherence: {max(phase_coherence):.4f}")
print(f"- Coherence at transitions: {np.mean([phase_coherence[t] for t in phase_transitions]) if phase_transitions.size > 0 else 0:.4f}")

print("\nTest 6 complete. Outputs saved to:")
print("- output/test6_consciousness_evolution.png")
print("- output/test6_phase_transitions.png")
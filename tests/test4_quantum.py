"""
Test 4: Quantum-like Superposition and Collapse Events
Tests wave function evolution, measurement, and Lévy flights
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
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

print("="*60)
print("TEST 4: Quantum Superposition & Collapse Events")
print("="*60)

# Simulate quantum-like behavior
n_tokens = 100
n_steps = 300

# Initialize in superposition
superposition_state = torch.ones(n_tokens, dtype=torch.complex64) / np.sqrt(n_tokens)
collapse_history = []
wave_function = []
entropy_history = []

print("Simulating wave function evolution...")
for t in range(n_steps):
    # Wave function evolution (unitary)
    phase = torch.exp(1j * 2 * np.pi * torch.rand(n_tokens))
    superposition_state = superposition_state * phase
    
    # Measurement probability
    prob = torch.abs(superposition_state)**2
    prob = prob / prob.sum()  # Normalize
    
    # Collapse events (low entropy = collapse)
    entropy = -torch.sum(prob * torch.log(prob + 1e-10))
    entropy_history.append(entropy.item())
    
    if entropy < 2.0:  # Collapse threshold
        # Collapse to eigenstate
        collapsed_idx = torch.multinomial(prob, 1)
        superposition_state = torch.zeros_like(superposition_state)
        superposition_state[collapsed_idx] = 1.0
        collapse_history.append((t, collapsed_idx.item(), entropy.item()))
    
    wave_function.append(prob.clone())

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Wave function evolution
wave_array = torch.stack(wave_function).numpy()
im = axes[0, 0].imshow(wave_array.T[:50, :], aspect='auto', cmap='hot', origin='lower')
axes[0, 0].set_title('Wave Function Evolution |ψ|²')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Token State')
plt.colorbar(im, ax=axes[0, 0])

# Mark collapse events
for t, idx, _ in collapse_history:
    axes[0, 0].axvline(x=t, color='cyan', alpha=0.5, linewidth=0.5)

# Entropy evolution
axes[0, 1].plot(entropy_history, 'magenta', linewidth=2)
axes[0, 1].axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='Collapse Threshold')
axes[0, 1].set_title('Entropy Evolution')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Von Neumann Entropy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.2)

# Collapse events
if collapse_history:
    collapse_times = [c[0] for c in collapse_history]
    collapse_entropies = [c[2] for c in collapse_history]
    collapse_states = [c[1] for c in collapse_history]
    
    scatter = axes[1, 0].scatter(collapse_times, collapse_states, 
                                c=collapse_entropies, s=100, 
                                cmap='plasma', edgecolor='white', alpha=0.8)
    axes[1, 0].set_title('Collapse Events')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Collapsed State Index')
    axes[1, 0].grid(True, alpha=0.2)
    plt.colorbar(scatter, ax=axes[1, 0], label='Pre-collapse Entropy')

# Lévy flight exploration
print("\nSimulating Lévy flight exploration...")
alpha = PHI  # Lévy parameter = golden ratio
levy_positions = []
current_pos = torch.zeros(2)

for _ in range(500):
    # Lévy distributed step
    step_size = levy_stable.rvs(alpha, beta=0, scale=0.1)
    angle = np.random.rand() * 2 * np.pi
    
    # Use numpy for consistent types
    step_x = step_size * np.cos(angle)
    step_y = step_size * np.sin(angle)
    step = torch.tensor([step_x, step_y], dtype=torch.float32)
    
    current_pos += step
    levy_positions.append(current_pos.clone())

levy_array = torch.stack(levy_positions).numpy()

axes[1, 1].plot(levy_array[:, 0], levy_array[:, 1], 'cyan', alpha=0.5, linewidth=0.5)
axes[1, 1].scatter(levy_array[0, 0], levy_array[0, 1], 
                   c='green', s=100, marker='o', edgecolor='white', label='Start', zorder=5)
axes[1, 1].scatter(levy_array[-1, 0], levy_array[-1, 1], 
                   c='red', s=100, marker='s', edgecolor='white', label='End', zorder=5)
axes[1, 1].scatter(levy_array[::50, 0], levy_array[::50, 1], 
                   c='gold', s=50, edgecolor='white', alpha=0.7, zorder=4)
axes[1, 1].set_title(f'Lévy Flight Exploration (α = φ = {PHI:.3f})')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Y')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.2)
axes[1, 1].set_aspect('equal')

plt.tight_layout()
plt.savefig('output/test4_quantum_dynamics.png', dpi=200, bbox_inches='tight')
plt.close()

# Additional analysis: Phase space portrait
print("\nGenerating phase space portrait...")

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
    
    def hamiltonian_evolution(self, field, momentum, dt=0.01):
        n_tokens = len(field)
        
        # Simplified evolution for phase portrait
        # Mean field force
        mean_r = field[:, 2].mean()
        mean_field_force = -self.PHI * (field[:, 2] - mean_r).unsqueeze(1) * field
        
        # Update
        momentum_new = momentum + dt * mean_field_force
        field_new = field + dt * momentum_new
        
        # Normalize
        norms = torch.norm(field_new[:, :2], dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        field_new[:, :2] = field_new[:, :2] / norms
        
        return field_new, momentum_new

gfg = GoldenFieldGate(vocab_size=100)
field_subset = gfg.embeddings.clone()
momentum_subset = torch.randn_like(field_subset) * 0.01

phase_trajectory_x = []
phase_trajectory_p = []

for _ in range(300):
    field_subset, momentum_subset = gfg.hamiltonian_evolution(field_subset, momentum_subset, dt=0.01)
    # Track mean position and momentum
    phase_trajectory_x.append(field_subset[:, 0].mean().item())
    phase_trajectory_p.append(momentum_subset[:, 0].mean().item())

# Plot phase portrait
plt.figure(figsize=(8, 8))
plt.plot(phase_trajectory_x, phase_trajectory_p, 'gold', alpha=0.8, linewidth=1.5)
plt.scatter(phase_trajectory_x[0], phase_trajectory_p[0], 
           c='green', s=100, marker='o', edgecolor='white', label='Start', zorder=5)
plt.scatter(phase_trajectory_x[-1], phase_trajectory_p[-1], 
           c='red', s=100, marker='s', edgecolor='white', label='End', zorder=5)
plt.title('Phase Space Portrait (Mean Field)')
plt.xlabel('⟨X⟩')
plt.ylabel('⟨P_X⟩')
plt.legend()
plt.grid(True, alpha=0.2)
plt.axis('equal')
plt.savefig('output/test4_phase_portrait.png', dpi=200, bbox_inches='tight')
plt.close()

# Quantum observable expectation values
print("\nComputing quantum observables...")
def quantum_observable(field, operator='pauli_z'):
    """Compute expectation value of quantum observable"""
    # Create density matrix ρ = |ψ⟩⟨ψ|
    psi = field[:, :2].flatten()  # Use x,y components
    psi = psi / torch.norm(psi)
    rho = torch.outer(psi, psi)
    
    # Define operators
    if operator == 'pauli_z':
        op = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    elif operator == 'pauli_x':
        op = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    else:  # pauli_y
        op = torch.tensor([[0.0, -1j], [1j, 0.0]], dtype=torch.complex64)
    
    # Expectation value ⟨O⟩ = Tr(ρO)
    # For simplicity, use first 2x2 block of density matrix
    rho_2x2 = rho[:2, :2]
    expectation = torch.trace(torch.matmul(rho_2x2, op)).real
    
    return expectation.item()

# Compute observables during evolution
observables_z = []
observables_x = []
field_test = gfg.embeddings[:10].clone()

for i in range(50):
    # Evolve slightly
    noise = torch.randn_like(field_test) * 0.1
    field_test += noise
    field_test = field_test / torch.norm(field_test, dim=1, keepdim=True)
    
    obs_z = quantum_observable(field_test, 'pauli_z')
    obs_x = quantum_observable(field_test, 'pauli_x')
    observables_z.append(obs_z)
    observables_x.append(obs_x)

plt.figure(figsize=(10, 5))
plt.plot(observables_z, 'b-', label='⟨σ_z⟩', linewidth=2)
plt.plot(observables_x, 'r-', label='⟨σ_x⟩', linewidth=2)
plt.xlabel('Evolution Step')
plt.ylabel('Expectation Value')
plt.title('Quantum Observable Evolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('output/test4_observables.png', dpi=200, bbox_inches='tight')
plt.close()

# Print analysis
print(f"\nQuantum Dynamics Analysis:")
print(f"- Total collapse events: {len(collapse_history)}")
print(f"- Average time between collapses: {n_steps/max(len(collapse_history), 1):.1f} steps")
print(f"- Mean entropy: {np.mean(entropy_history):.4f}")
print(f"- Entropy variance: {np.var(entropy_history):.4f}")
print(f"\nLévy Flight Statistics:")
print(f"- Total distance traveled: {np.sum(np.sqrt(np.sum(np.diff(levy_array, axis=0)**2, axis=1))):.2f}")
print(f"- Final displacement: {np.sqrt(np.sum(levy_array[-1]**2)):.2f}")
print(f"- Exploration efficiency: {np.sqrt(np.sum(levy_array[-1]**2)) / len(levy_array):.4f}")
print(f"\nPhase Space Properties:")
print(f"- Phase space area covered: ~{np.std(phase_trajectory_x) * np.std(phase_trajectory_p) * np.pi:.4f}")
print(f"- Mean position oscillation: {np.std(phase_trajectory_x):.4f}")
print(f"- Mean momentum oscillation: {np.std(phase_trajectory_p):.4f}")

print("\nTest 4 complete. Outputs saved to:")
print("- output/test4_quantum_dynamics.png")
print("- output/test4_phase_portrait.png")
print("- output/test4_observables.png")
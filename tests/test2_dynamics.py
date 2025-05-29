"""
Test 2: Gravitational Field Dynamics
Tests field evolution, energy conservation, and coherence
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
    """Implementation focused on dynamics"""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.PHI = PHI
        self.GAP = GAP
        self.embeddings = self._create_golden_embeddings(vocab_size)
        
    def _create_golden_embeddings(self, vocab_size):
        """Create golden spiral cylindrical embeddings"""
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
    
    def compute_gravitational_field(self, active_tokens):
        """Compute gravitational interactions between active tokens"""
        active_field = self.embeddings[active_tokens]
        
        # Distance matrix with causal mask
        diff = active_field.unsqueeze(1) - active_field.unsqueeze(0)
        dist = torch.sqrt((diff**2).sum(-1) + 1e-8)
        
        # Gravitational matrix (negative for attraction)
        M = -dist / self.PHI
        
        # Apply causal mask (lower triangular)
        seq_len = len(active_tokens)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        M = M * causal_mask
        
        return M, active_field
    
    def hamiltonian_evolution(self, field, momentum, dt=0.01):
        """Evolve field according to Hamiltonian dynamics"""
        n_tokens = len(field)
        
        # Create token indices for this subset
        token_indices = torch.arange(n_tokens)
        
        # Compute forces from gravitational potential
        M, _ = self.compute_gravitational_field(token_indices)
        forces = -torch.matmul(M, field[:n_tokens])
        
        # Mean field contribution
        mean_r = field[:, 2].mean()
        mean_field_force = -self.PHI * (field[:, 2] - mean_r).unsqueeze(1) * field
        
        # Heun-Euler integration
        momentum_half = momentum + 0.5 * dt * (forces + mean_field_force)
        field_new = field + dt * momentum_half
        
        # Project back to manifold (normalize xy components)
        norms = torch.norm(field_new[:, :2], dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        field_new[:, :2] = field_new[:, :2] / norms
        
        # Update momentum with new forces
        M_new, _ = self.compute_gravitational_field(token_indices)
        forces_new = -torch.matmul(M_new, field_new[:n_tokens])
        momentum_new = momentum_half + 0.5 * dt * forces_new
        
        return field_new, momentum_new
    
    def compute_coherence(self, field):
        """Compute field coherence metric"""
        # Covariance matrix of field positions
        centered = field - field.mean(dim=0)
        cov = torch.matmul(centered.T, centered) / len(field)
        
        # Coherence = ratio of largest eigenvalue to trace
        eigvals = torch.linalg.eigvals(cov).real
        coherence = eigvals.max() / eigvals.sum()
        
        return coherence.item()
    
    def compute_entropy(self, field):
        """Compute field entropy"""
        # Create probability distribution from field energy
        energy = 0.5 * torch.sum(field**2, dim=1)
        probs = F.softmax(-energy / self.PHI, dim=0)
        
        # Shannon entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        
        return entropy.item()

# Initialize system
print("="*60)
print("TEST 2: Gravitational Field Evolution")
print("="*60)

gfg = GoldenFieldGate(vocab_size=500)

# Select subset of tokens for visualization
active_tokens = torch.randint(0, 500, (50,))
M, active_field = gfg.compute_gravitational_field(active_tokens)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Gravitational matrix
im1 = axes[0, 0].imshow(-M.numpy(), cmap='RdBu', aspect='auto')
axes[0, 0].set_title('Gravitational Interaction Matrix')
axes[0, 0].set_xlabel('Token j')
axes[0, 0].set_ylabel('Token i')
plt.colorbar(im1, ax=axes[0, 0])

# Force field visualization
field_2d = active_field[:, :2].numpy()
forces = torch.matmul(M, active_field).numpy()

axes[0, 1].quiver(field_2d[:, 0], field_2d[:, 1], 
                  forces[:, 0], forces[:, 1], 
                  color='cyan', alpha=0.7, scale=5)
axes[0, 1].scatter(field_2d[:, 0], field_2d[:, 1], c='gold', s=50, edgecolor='white')
axes[0, 1].set_title('Gravitational Force Field')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')
axes[0, 1].set_aspect('equal')
axes[0, 1].grid(True, alpha=0.2)

# Energy landscape
x = np.linspace(-1.2, 1.2, 100)
y = np.linspace(-1.2, 1.2, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(len(active_field)):
    dist = np.sqrt((X - field_2d[i, 0])**2 + (Y - field_2d[i, 1])**2)
    Z += -1 / (PHI * (dist + 0.1))

contour = axes[0, 2].contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
axes[0, 2].scatter(field_2d[:, 0], field_2d[:, 1], c='red', s=50, edgecolor='white', zorder=5)
axes[0, 2].set_title('Gravitational Potential Energy')
axes[0, 2].set_xlabel('X')
axes[0, 2].set_ylabel('Y')
axes[0, 2].set_aspect('equal')
plt.colorbar(contour, ax=axes[0, 2])

# Hamiltonian evolution - use a subset for efficiency
print("\nRunning Hamiltonian evolution...")
n_evolve = 100  # Number of tokens to evolve
field = gfg.embeddings[:n_evolve].clone()
momentum = torch.randn_like(field) * 0.01

energies = []
coherences = []
entropies = []

for t in range(200):
    field, momentum = gfg.hamiltonian_evolution(field, momentum, dt=0.01)
    
    # Compute metrics
    # Energy = kinetic + potential
    kinetic_energy = 0.5 * torch.sum(momentum**2)
    # For potential energy, we need the interaction matrix for current field
    M_current, _ = gfg.compute_gravitational_field(torch.arange(n_evolve))
    potential_energy = -0.5 * torch.sum(M_current)
    total_energy = kinetic_energy + potential_energy
    
    coherence = gfg.compute_coherence(field)
    entropy = gfg.compute_entropy(field)
    
    energies.append(total_energy.item())
    coherences.append(coherence)
    entropies.append(entropy)

# Plot evolution metrics
time_steps = np.arange(len(energies))

axes[1, 0].plot(time_steps, energies, 'gold', linewidth=2)
axes[1, 0].set_title('Hamiltonian Energy Conservation')
axes[1, 0].set_xlabel('Time Steps')
axes[1, 0].set_ylabel('Total Energy')
axes[1, 0].grid(True, alpha=0.2)

axes[1, 1].plot(time_steps, coherences, 'cyan', linewidth=2)
axes[1, 1].axhline(y=0.91, color='red', linestyle='--', label='Collapse Threshold')
axes[1, 1].set_title('Field Coherence Evolution')
axes[1, 1].set_xlabel('Time Steps')
axes[1, 1].set_ylabel('Coherence')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.2)

axes[1, 2].plot(time_steps, entropies, 'magenta', linewidth=2)
axes[1, 2].set_title('Entropy Reduction')
axes[1, 2].set_xlabel('Time Steps')
axes[1, 2].set_ylabel('Shannon Entropy')
axes[1, 2].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('output/test2_field_dynamics.png', dpi=200, bbox_inches='tight')
plt.close()

# Print analysis
print(f"\nDynamics Analysis:")
print(f"- Initial energy: {energies[0]:.4f}")
print(f"- Final energy: {energies[-1]:.4f}")
print(f"- Energy drift: {abs(energies[-1] - energies[0]):.6f}")
print(f"- Max coherence: {max(coherences):.4f}")
print(f"- Min entropy: {min(entropies):.4f}")
print(f"- Entropy reduction: {entropies[0] - entropies[-1]:.4f}")

# Check for phase transitions
coherence_jumps = np.diff(coherences)
large_jumps = np.where(np.abs(coherence_jumps) > 0.05)[0]
print(f"\nPhase transitions detected at steps: {large_jumps}")

print("\nTest 2 complete. Output saved to: output/test2_field_dynamics.png")
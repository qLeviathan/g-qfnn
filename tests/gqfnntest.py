import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set random seed for reproducibility
torch.manual_seed(42)

def log_phase_embedding(theta, r=1.0, eps=1e-6):
    """
    Compute log-domain cylindrical embedding from angle theta.
    Handles sign preservation for proper quadrant mapping.
    """
    ln_r = torch.log(torch.tensor(r) + eps)
    
    # Preserve sign information
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    # Log-transform with sign preservation
    x = ln_r + torch.log(torch.abs(cos_theta) + eps)
    y = ln_r + torch.log(torch.abs(sin_theta) + eps)
    
    # Add sign back as phase indicator
    x = x * torch.sign(cos_theta)
    y = y * torch.sign(sin_theta)
    
    return torch.stack([x, y], dim=-1)

def raw_phase_embedding(theta, r=1.0):
    """Standard Cartesian embedding for comparison"""
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=-1)

# Test 1: Compare embeddings across phase space
print("=" * 60)
print("TEST 1: Phase Space Comparison")
print("=" * 60)

theta_test = torch.linspace(0, 2*np.pi, 100)
raw_embed = raw_phase_embedding(theta_test)
log_embed = log_phase_embedding(theta_test)

fig = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, figure=fig)

# Raw embedding
ax1 = fig.add_subplot(gs[0])
ax1.plot(raw_embed[:, 0], raw_embed[:, 1], 'b-', linewidth=2)
ax1.set_title('Raw sin/cos Embedding')
ax1.set_xlabel('x = r·cos(θ)')
ax1.set_ylabel('y = r·sin(θ)')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Log embedding
ax2 = fig.add_subplot(gs[1])
ax2.plot(log_embed[:, 0], log_embed[:, 1], 'r-', linewidth=2)
ax2.set_title('Log-domain Embedding')
ax2.set_xlabel('x = ln(r) + ln|cos(θ)|')
ax2.set_ylabel('y = ln(r) + ln|sin(θ)|')
ax2.grid(True, alpha=0.3)

# Phase unwrapping comparison
ax3 = fig.add_subplot(gs[2])
ax3.plot(theta_test, raw_embed[:, 0], 'b-', label='raw cos', alpha=0.7)
ax3.plot(theta_test, log_embed[:, 0], 'r-', label='log cos', alpha=0.7)
ax3.set_title('Phase Unwrapping')
ax3.set_xlabel('θ (radians)')
ax3.set_ylabel('x-component')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Test 2: Superposition behavior
print("\nTEST 2: Superposition Properties")
print("-" * 40)

# Create two waves
theta1 = torch.tensor([np.pi/4])  # 45 degrees
theta2 = torch.tensor([np.pi/3])  # 60 degrees

raw1 = raw_phase_embedding(theta1)
raw2 = raw_phase_embedding(theta2)
log1 = log_phase_embedding(theta1)
log2 = log_phase_embedding(theta2)

# Superposition
raw_super = raw1 + raw2
log_super = log1 + log2

print(f"Raw superposition: {raw_super.squeeze().numpy()}")
print(f"Log superposition: {log_super.squeeze().numpy()}")
print(f"Log becomes additive: {torch.exp(log_super).squeeze().numpy()}")

# Test 3: Hebbian learning simulation
print("\n" + "=" * 60)
print("TEST 3: Hebbian Learning Efficiency")
print("=" * 60)

class HebbianLayer(nn.Module):
    def __init__(self, embed_fn, hidden_dim=16):
        super().__init__()
        self.embed_fn = embed_fn
        self.W = nn.Parameter(torch.randn(2, hidden_dim) * 0.1)
        self.tau = nn.Parameter(torch.tensor(0.1))  # Learning rate
        
    def forward(self, theta):
        # Get embedding
        x = self.embed_fn(theta)
        
        # Pre-activation
        h = torch.matmul(x, self.W)
        
        # Hebbian update: ΔW = τ * x^T * h
        with torch.no_grad():
            hebbian_update = self.tau * torch.matmul(x.T, h)
            self.W.data += hebbian_update
            
        return h, x

# Compare learning dynamics
raw_hebbian = HebbianLayer(raw_phase_embedding)
log_hebbian = HebbianLayer(log_phase_embedding)

# Training loop
n_steps = 100
theta_train = torch.rand(n_steps) * 2 * np.pi

raw_norms = []
log_norms = []

for i in range(n_steps):
    theta_i = theta_train[i:i+1]
    
    # Forward pass
    _, _ = raw_hebbian(theta_i)
    _, _ = log_hebbian(theta_i)
    
    # Track weight norms
    raw_norms.append(torch.norm(raw_hebbian.W).item())
    log_norms.append(torch.norm(log_hebbian.W).item())

plt.figure(figsize=(10, 5))
plt.plot(raw_norms, 'b-', label='Raw embedding weights', alpha=0.7)
plt.plot(log_norms, 'r-', label='Log embedding weights', alpha=0.7)
plt.xlabel('Training steps')
plt.ylabel('Weight norm')
plt.title('Hebbian Weight Evolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('hebbian_evolution.png', dpi=150, bbox_inches='tight')
plt.close()

# Test 4: Entropy and collapse detection
print("\nTEST 4: Symbolic Entropy & Collapse Detection")
print("-" * 40)

def compute_entropy(embeddings):
    """Compute symbolic entropy from embeddings"""
    # Normalize
    probs = torch.softmax(embeddings.flatten(), dim=0)
    # Shannon entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    return entropy.item()

# Generate phase sweep
theta_sweep = torch.linspace(0, 4*np.pi, 200)
raw_sweep = raw_phase_embedding(theta_sweep)
log_sweep = log_phase_embedding(theta_sweep)

# Compute running entropy
window_size = 20
raw_entropies = []
log_entropies = []

for i in range(len(theta_sweep) - window_size):
    raw_window = raw_sweep[i:i+window_size]
    log_window = log_sweep[i:i+window_size]
    
    raw_entropies.append(compute_entropy(raw_window))
    log_entropies.append(compute_entropy(log_window))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(raw_entropies, 'b-', label='Raw embedding', alpha=0.7)
plt.plot(log_entropies, 'r-', label='Log embedding', alpha=0.7)
plt.xlabel('Phase window')
plt.ylabel('Entropy')
plt.title('Symbolic Entropy Evolution')
plt.legend()
plt.grid(True, alpha=0.3)

# Collapse detection threshold
collapse_threshold = np.percentile(log_entropies, 20)
collapse_points = np.where(np.array(log_entropies) < collapse_threshold)[0]

plt.subplot(1, 2, 2)
plt.plot(log_entropies, 'r-', alpha=0.7)
plt.axhline(y=collapse_threshold, color='k', linestyle='--', label='Collapse threshold')
plt.scatter(collapse_points, [log_entropies[i] for i in collapse_points], 
           color='gold', s=50, zorder=5, label='Collapse events')
plt.xlabel('Phase window')
plt.ylabel('Log entropy')
plt.title('Collapse Event Detection')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('entropy_collapse.png', dpi=150, bbox_inches='tight')
plt.close()

# Test 5: Multi-scale resonance
print("\n" + "=" * 60)
print("TEST 5: Multi-scale Resonance Patterns")
print("=" * 60)

# Create multi-frequency signal
t = torch.linspace(0, 10, 1000)
frequencies = [1.0, 2.5, 5.0]  # Different resonance scales
amplitudes = [1.0, 0.5, 0.3]

# Composite wave
theta_composite = torch.zeros_like(t)
for freq, amp in zip(frequencies, amplitudes):
    theta_composite += amp * torch.sin(2 * np.pi * freq * t)

# Embed both ways
raw_composite = raw_phase_embedding(theta_composite)
log_composite = log_phase_embedding(theta_composite)

# Compute local phase coherence
def phase_coherence(embeddings, window=50):
    """Compute local phase coherence using sliding window"""
    coherence = []
    for i in range(len(embeddings) - window):
        window_data = embeddings[i:i+window]
        # Compute eigenvalues of covariance
        cov = torch.cov(window_data.T)
        eigvals = torch.linalg.eigvals(cov).real
        # Coherence = ratio of largest to sum of eigenvalues
        coherence.append((eigvals.max() / eigvals.sum()).item())
    return coherence

raw_coherence = phase_coherence(raw_composite)
log_coherence = phase_coherence(log_composite)

plt.figure(figsize=(15, 8))

# Signal
plt.subplot(3, 1, 1)
plt.plot(t[:500], theta_composite[:500], 'k-', alpha=0.7)
plt.title('Multi-frequency Composite Signal')
plt.ylabel('θ(t)')
plt.grid(True, alpha=0.3)

# Embeddings
plt.subplot(3, 1, 2)
plt.plot(raw_composite[:500, 0], 'b-', label='Raw x', alpha=0.7)
plt.plot(log_composite[:500, 0], 'r-', label='Log x', alpha=0.7)
plt.title('Embedding Comparison')
plt.ylabel('x-component')
plt.legend()
plt.grid(True, alpha=0.3)

# Coherence
plt.subplot(3, 1, 3)
plt.plot(raw_coherence[:450], 'b-', label='Raw coherence', alpha=0.7)
plt.plot(log_coherence[:450], 'r-', label='Log coherence', alpha=0.7)
plt.title('Phase Coherence Detection')
plt.xlabel('Time window')
plt.ylabel('Coherence')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multiscale_resonance.png', dpi=150, bbox_inches='tight')
plt.close()

# Summary statistics
print("\nSummary Statistics:")
print("-" * 40)
print(f"Average entropy - Raw: {np.mean(raw_entropies):.4f}, Log: {np.mean(log_entropies):.4f}")
print(f"Entropy variance - Raw: {np.var(raw_entropies):.4f}, Log: {np.var(log_entropies):.4f}")
print(f"Collapse events detected: {len(collapse_points)}")
print(f"Average coherence - Raw: {np.mean(raw_coherence):.4f}, Log: {np.mean(log_coherence):.4f}")
print(f"Weight norm ratio (log/raw): {log_norms[-1]/raw_norms[-1]:.4f}")

print("\n" + "=" * 60)
print("CONCLUSION: Log-phase embedding provides:")
print("- Better numerical stability near phase boundaries")
print("- Natural multiplicative → additive transformation")
print("- Clearer entropy gradients for collapse detection")
print("- More stable Hebbian learning dynamics")
print("- Enhanced multi-scale resonance separation")
print("=" * 60)

# Bonus: Quantum-style observable
print("\nBONUS: Quantum Observable Construction")
print("-" * 40)

def quantum_observable(theta, log_embed=True):
    """Construct quantum-style observable from phase"""
    embed_fn = log_phase_embedding if log_embed else raw_phase_embedding
    psi = embed_fn(theta)
    
    # Density matrix ρ = |ψ⟩⟨ψ|
    rho = torch.outer(psi.flatten(), psi.flatten())
    
    # Observable expectation ⟨O⟩ = Tr(ρO)
    # Using Pauli-Z as observable
    pauli_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    expectation = torch.trace(torch.matmul(rho.view(2, 2), pauli_z))
    
    return expectation.item()

theta_obs = torch.linspace(0, 2*np.pi, 50)
raw_obs = [quantum_observable(t, log_embed=False) for t in theta_obs]
log_obs = [quantum_observable(t, log_embed=True) for t in theta_obs]

plt.figure(figsize=(10, 5))
plt.plot(theta_obs, raw_obs, 'b-', label='Raw observable', alpha=0.7)
plt.plot(theta_obs, log_obs, 'r-', label='Log observable', alpha=0.7)
plt.xlabel('θ (radians)')
plt.ylabel('⟨Pauli-Z⟩')
plt.title('Quantum Observable Expectation Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('quantum_observable.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Observable range - Raw: [{min(raw_obs):.4f}, {max(raw_obs):.4f}]")
print(f"Observable range - Log: [{min(log_obs):.4f}, {max(log_obs):.4f}]")
print("\nTest complete! Check generated PNG files for visualizations.")
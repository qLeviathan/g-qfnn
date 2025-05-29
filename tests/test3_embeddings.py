"""
Test 3: Log-Phase vs Standard Embeddings in GoldenFieldGate
Compares embedding strategies and Hebbian learning efficiency
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
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

def log_phase_embedding(theta, r=1.0, eps=1e-6):
    """Log-domain cylindrical embedding"""
    # Ensure r is a float, not a tensor
    if isinstance(r, torch.Tensor):
        r = r.item()
    
    ln_r = np.log(r + eps)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    # Convert to numpy for consistent operations
    cos_val = cos_theta.item() if isinstance(cos_theta, torch.Tensor) else cos_theta
    sin_val = sin_theta.item() if isinstance(sin_theta, torch.Tensor) else sin_theta
    
    x = ln_r + np.log(abs(cos_val) + eps) * np.sign(cos_val)
    y = ln_r + np.log(abs(sin_val) + eps) * np.sign(sin_val)
    
    return torch.tensor([x, y], dtype=torch.float32)

print("="*60)
print("TEST 3: Log-Phase vs Standard Embeddings")
print("="*60)

# Create vocabulary embeddings
vocab_size = 500
theta_vocab = torch.tensor([2 * np.pi * ((i * PHI) % 1.0) for i in range(vocab_size)])
r_vocab = torch.tensor([GAP + (1 - GAP) * (i / vocab_size) for i in range(vocab_size)])

# Standard GFG embeddings
standard_embed = torch.stack([
    r_vocab * torch.cos(theta_vocab),
    r_vocab * torch.sin(theta_vocab)
], dim=1)

# Log-phase embeddings
log_embed = torch.stack([
    log_phase_embedding(theta_vocab[i], r_vocab[i]) for i in range(vocab_size)
])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Embedding comparison
axes[0, 0].scatter(standard_embed[:, 0], standard_embed[:, 1], 
                   c=range(500), cmap='twilight', s=10, alpha=0.6)
axes[0, 0].set_title('Standard GFG Embeddings')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')
axes[0, 0].set_aspect('equal')
axes[0, 0].grid(True, alpha=0.2)

axes[0, 1].scatter(log_embed[:, 0], log_embed[:, 1], 
                   c=range(500), cmap='twilight', s=10, alpha=0.6)
axes[0, 1].set_title('Log-Phase Embeddings')
axes[0, 1].set_xlabel('log(X)')
axes[0, 1].set_ylabel('log(Y)')
axes[0, 1].grid(True, alpha=0.2)

# Distance matrix comparison (subset for visibility)
subset_size = 100
dist_standard = torch.cdist(standard_embed[:subset_size], standard_embed[:subset_size])
dist_log = torch.cdist(log_embed[:subset_size], log_embed[:subset_size])

im2 = axes[0, 2].imshow(dist_standard, cmap='viridis', aspect='auto')
axes[0, 2].set_title('Standard Distance Matrix')
plt.colorbar(im2, ax=axes[0, 2])

im3 = axes[1, 0].imshow(dist_log, cmap='viridis', aspect='auto')
axes[1, 0].set_title('Log-Phase Distance Matrix')
plt.colorbar(im3, ax=axes[1, 0])

# Hebbian learning comparison
print("\nRunning Hebbian learning comparison...")
# Initialize with correct dimensions (2x2 for 2D embeddings)
hebbian_standard = torch.zeros(2, 2)
hebbian_log = torch.zeros(2, 2)

# Track weight evolution
standard_norms = []
log_norms = []

# For visualization, we'll also track full matrices
hebbian_standard_full = torch.zeros(subset_size, subset_size)
hebbian_log_full = torch.zeros(subset_size, subset_size)

for i in range(subset_size - 1):
    # Standard Hebbian update (2D embeddings)
    pre = standard_embed[i]  # Shape: (2,)
    post = standard_embed[i+1]  # Shape: (2,)
    hebbian_standard += 0.01 * PHI * torch.outer(post, pre)
    standard_norms.append(torch.norm(hebbian_standard).item())
    
    # Log Hebbian update
    pre_log = log_embed[i]  # Shape: (2,)
    post_log = log_embed[i+1]  # Shape: (2,)
    hebbian_log += 0.01 * PHI * torch.outer(post_log, pre_log)
    log_norms.append(torch.norm(hebbian_log).item())
    
    # For visualization matrices (token-to-token connections)
    for j in range(i+1):
        hebbian_standard_full[i+1, j] = torch.dot(standard_embed[i+1], standard_embed[j]).item()
        hebbian_log_full[i+1, j] = torch.dot(log_embed[i+1], log_embed[j]).item()

# Symmetrize visualization matrices
hebbian_standard_full = hebbian_standard_full + hebbian_standard_full.T
hebbian_log_full = hebbian_log_full + hebbian_log_full.T

im4 = axes[1, 1].imshow(hebbian_standard_full, cmap='RdBu', aspect='auto')
axes[1, 1].set_title('Standard Hebbian Connectivity')
plt.colorbar(im4, ax=axes[1, 1])

im5 = axes[1, 2].imshow(hebbian_log_full, cmap='RdBu', aspect='auto')
axes[1, 2].set_title('Log-Phase Hebbian Connectivity')
plt.colorbar(im5, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('output/test3_embedding_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

# Additional analysis plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Weight norm evolution
ax1.plot(standard_norms, 'b-', label='Standard', alpha=0.7, linewidth=2)
ax1.plot(log_norms, 'r-', label='Log-Phase', alpha=0.7, linewidth=2)
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Weight Norm')
ax1.set_title('Hebbian Weight Evolution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Signal amplification ratio
ratio = np.array(log_norms) / (np.array(standard_norms) + 1e-8)
ax2.plot(ratio, 'gold', linewidth=2)
ax2.axhline(y=ratio[-1], color='red', linestyle='--', 
            label=f'Final ratio: {ratio[-1]:.1f}x')
ax2.set_xlabel('Training Steps')
ax2.set_ylabel('Log/Standard Ratio')
ax2.set_title('Signal Amplification')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/test3_hebbian_analysis.png', dpi=200, bbox_inches='tight')
plt.close()

# Entropy comparison
def compute_entropy(embeddings):
    """Compute symbolic entropy from embeddings"""
    # Normalize to probability distribution
    flat = embeddings.flatten()
    flat_positive = flat - flat.min() + 1e-8
    probs = flat_positive / flat_positive.sum()
    # Shannon entropy
    entropy = -torch.sum(probs * torch.log(probs))
    return entropy.item()

# Compute entropy for windows
window_size = 20
standard_entropies = []
log_entropies = []

for i in range(0, vocab_size - window_size, 5):
    standard_window = standard_embed[i:i+window_size]
    log_window = log_embed[i:i+window_size]
    
    standard_entropies.append(compute_entropy(standard_window))
    log_entropies.append(compute_entropy(log_window))

# Plot entropy comparison
plt.figure(figsize=(10, 5))
plt.plot(standard_entropies, 'b-', label='Standard', alpha=0.7, linewidth=2)
plt.plot(log_entropies, 'r-', label='Log-Phase', alpha=0.7, linewidth=2)
plt.xlabel('Window Position')
plt.ylabel('Entropy')
plt.title('Symbolic Entropy: Standard vs Log-Phase')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('output/test3_entropy_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

# Additional analysis: Phase space coverage
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Standard embedding phase space density
H_std, xedges, yedges = np.histogram2d(
    standard_embed[:, 0].numpy(),
    standard_embed[:, 1].numpy(),
    bins=30
)
im_std = ax1.imshow(H_std.T, origin='lower', cmap='hot', 
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto')
ax1.set_title('Standard Embedding Density')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
plt.colorbar(im_std, ax=ax1)

# Log-phase embedding phase space density
H_log, xedges, yedges = np.histogram2d(
    log_embed[:, 0].numpy(),
    log_embed[:, 1].numpy(),
    bins=30
)
im_log = ax2.imshow(H_log.T, origin='lower', cmap='hot',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    aspect='auto')
ax2.set_title('Log-Phase Embedding Density')
ax2.set_xlabel('log(X)')
ax2.set_ylabel('log(Y)')
plt.colorbar(im_log, ax=ax2)

plt.tight_layout()
plt.savefig('output/test3_phase_space_density.png', dpi=200, bbox_inches='tight')
plt.close()

# Print analysis
print(f"\nEmbedding Analysis:")
print(f"- Standard embedding range: X=[{standard_embed[:, 0].min():.3f}, {standard_embed[:, 0].max():.3f}]")
print(f"- Log-phase embedding range: X=[{log_embed[:, 0].min():.3f}, {log_embed[:, 0].max():.3f}]")
print(f"\nHebbian Learning (2x2 weight matrices):")
print(f"- Final weight norm (standard): {standard_norms[-1]:.4f}")
print(f"- Final weight norm (log-phase): {log_norms[-1]:.4f}")
print(f"- Signal amplification: {ratio[-1]:.1f}x")
print(f"\nConnectivity Analysis:")
print(f"- Standard connectivity range: [{hebbian_standard_full.min():.3f}, {hebbian_standard_full.max():.3f}]")
print(f"- Log-phase connectivity range: [{hebbian_log_full.min():.3f}, {hebbian_log_full.max():.3f}]")
print(f"\nEntropy Analysis:")
print(f"- Mean entropy (standard): {np.mean(standard_entropies):.4f}")
print(f"- Mean entropy (log-phase): {np.mean(log_entropies):.4f}")
print(f"- Entropy variance (standard): {np.var(standard_entropies):.4f}")
print(f"- Entropy variance (log-phase): {np.var(log_entropies):.4f}")
print(f"\nPhase Space Coverage:")
print(f"- Standard: {np.count_nonzero(H_std)} / {H_std.size} bins occupied ({100*np.count_nonzero(H_std)/H_std.size:.1f}%)")
print(f"- Log-phase: {np.count_nonzero(H_log)} / {H_log.size} bins occupied ({100*np.count_nonzero(H_log)/H_log.size:.1f}%)")

print("\nTest 3 complete. Outputs saved to:")
print("- output/test3_embedding_comparison.png")
print("- output/test3_hebbian_analysis.png")
print("- output/test3_entropy_comparison.png")
print("- output/test3_phase_space_density.png")
"""
Test 1: Golden Spiral Cylindrical Manifold Visualization
Tests the fundamental geometric structure of GoldenFieldGate
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import os

# Create output directory
os.makedirs('output', exist_ok=True)

# Set style and random seed
plt.style.use('dark_background')
torch.manual_seed(1618)
np.random.seed(1618)

# Golden ratio constants
PHI = (1 + np.sqrt(5)) / 2
GAP = 1 - 1/PHI  # 0.381966...

class GoldenFieldGate:
    """Core implementation for manifold testing"""
    
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
                r  # z = r creates cone within cylinder
            ])
        return coords

# Initialize system
print("="*60)
print("TEST 1: Golden Spiral Cylindrical Manifold")
print("="*60)

gfg = GoldenFieldGate(vocab_size=500)

fig = plt.figure(figsize=(18, 6))
gs = GridSpec(1, 3, figure=fig)

# 3D cylinder view
ax1 = fig.add_subplot(gs[0], projection='3d')
embeddings = gfg.embeddings.numpy()

# Color by radial position
colors = embeddings[:, 2]
scatter = ax1.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], 
                     c=colors, cmap='plasma', s=20, alpha=0.6)

# Draw cylinder boundaries
theta = np.linspace(0, 2*np.pi, 100)
z_bottom = np.full_like(theta, GAP)
z_top = np.full_like(theta, 1.0)

for z, r in [(z_bottom, GAP), (z_top, 1.0)]:
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)
    ax1.plot(x_circle, y_circle, z, 'w-', alpha=0.3)

# Draw vertical lines
for i in range(0, 360, 45):
    theta_line = np.radians(i)
    x_line = [GAP * np.cos(theta_line), np.cos(theta_line)]
    y_line = [GAP * np.sin(theta_line), np.sin(theta_line)]
    z_line = [GAP, 1.0]
    ax1.plot(x_line, y_line, z_line, 'w-', alpha=0.2)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z (radius)')
ax1.set_title('Cylindrical Semantic Manifold')
plt.colorbar(scatter, ax=ax1, pad=0.1)

# Top-down view showing golden spiral
ax2 = fig.add_subplot(gs[1])
ax2.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, cmap='plasma', s=10, alpha=0.6)

# Draw spiral path
spiral_idx = np.arange(0, 200, 5)
ax2.plot(embeddings[spiral_idx, 0], embeddings[spiral_idx, 1], 'w-', alpha=0.5, linewidth=1)

# Add circles
circle_gap = Circle((0, 0), GAP, fill=False, edgecolor='cyan', linewidth=2, alpha=0.5)
circle_max = Circle((0, 0), 1.0, fill=False, edgecolor='magenta', linewidth=2, alpha=0.5)
ax2.add_patch(circle_gap)
ax2.add_patch(circle_max)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Golden Spiral Distribution (Top View)')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.2)

# Radial distribution
ax3 = fig.add_subplot(gs[2])
radii = np.sqrt(embeddings[:, 0]**2 + embeddings[:, 1]**2)
ax3.hist(radii, bins=50, color='gold', alpha=0.7, edgecolor='white', linewidth=0.5)
ax3.axvline(x=GAP, color='cyan', linestyle='--', linewidth=2, label=f'GAP = {GAP:.3f}')
ax3.axvline(x=1.0, color='magenta', linestyle='--', linewidth=2, label='r = 1.0')
ax3.set_xlabel('Radius')
ax3.set_ylabel('Token Count')
ax3.set_title('Radial Distribution')
ax3.legend()
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('output/test1_manifold_structure.png', dpi=200, bbox_inches='tight')
plt.close()

# Additional analysis
print(f"\nManifold Properties:")
print(f"- Vocabulary size: {gfg.vocab_size}")
print(f"- Golden ratio (φ): {PHI:.6f}")
print(f"- GAP boundary: {GAP:.6f}")
print(f"- Radial range: [{GAP:.3f}, 1.000]")
print(f"- Angular distribution: Golden angle = 2π/φ² ≈ {2*np.pi/PHI**2:.6f} rad")

# Check golden spiral properties
angles = []
for i in range(10):
    theta = 2 * np.pi * ((i * PHI) % 1.0)
    angles.append(theta)

print(f"\nFirst 10 angles (radians):")
for i, angle in enumerate(angles):
    print(f"  Token {i}: θ = {angle:.4f} ({np.degrees(angle):.1f}°)")

# Verify uniform distribution on cylinder
print(f"\nDistribution validation:")
print(f"- Min radius: {radii.min():.6f}")
print(f"- Max radius: {radii.max():.6f}")
print(f"- Mean radius: {radii.mean():.6f}")
print(f"- Std radius: {radii.std():.6f}")

# Angular distribution check
thetas = np.arctan2(embeddings[:, 1], embeddings[:, 0])
thetas[thetas < 0] += 2 * np.pi
print(f"- Angular coverage: {np.degrees(thetas.min()):.1f}° to {np.degrees(thetas.max()):.1f}°")

print("\nTest 1 complete. Output saved to: output/test1_manifold_structure.png")
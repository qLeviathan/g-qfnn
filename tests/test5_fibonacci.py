"""
Test 5: Fibonacci Resonance and Convergence Analysis
Tests multi-scale resonance patterns and golden ratio convergence
"""

import torch
import torch.nn.functional as F
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
    
    def compute_coherence(self, field):
        """Compute field coherence metric"""
        centered = field - field.mean(dim=0)
        cov = torch.matmul(centered.T, centered) / len(field)
        eigvals = torch.linalg.eigvals(cov).real
        coherence = eigvals.max() / eigvals.sum()
        return coherence.item()
    
    def compute_entropy(self, field):
        """Compute field entropy"""
        energy = 0.5 * torch.sum(field**2, dim=1)
        probs = F.softmax(-energy / self.PHI, dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return entropy.item()

print("="*60)
print("TEST 5: Fibonacci Resonance Levels")
print("="*60)

# Initialize system
gfg = GoldenFieldGate(vocab_size=500)

# Generate Fibonacci sequence
fib_scales = [1, 1]
for i in range(15):
    fib_scales.append(fib_scales[-1] + fib_scales[-2])

print(f"Fibonacci sequence: {fib_scales[:10]}")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Multi-scale resonance analysis
print("\nAnalyzing resonance at Fibonacci scales...")
resonance_patterns = []
for scale in fib_scales[:10]:
    # Create scaled field
    scaled_field = gfg.embeddings * scale
    coherence = gfg.compute_coherence(scaled_field)
    entropy = gfg.compute_entropy(scaled_field)
    
    # Compute spectral properties
    cov = torch.matmul(scaled_field.T, scaled_field) / len(scaled_field)
    eigvals = torch.linalg.eigvals(cov).real
    spectral_gap = eigvals[0] - eigvals[1] if len(eigvals) > 1 else 0
    
    resonance_patterns.append((scale, coherence, entropy, spectral_gap.item()))

scales, coherences, entropies, spectral_gaps = zip(*resonance_patterns)

# Plot coherence vs scale
axes[0, 0].plot(scales, coherences, 'o-', color='gold', markersize=10, linewidth=2)
axes[0, 0].set_title('Coherence vs Fibonacci Scale')
axes[0, 0].set_xlabel('Fibonacci Scale')
axes[0, 0].set_ylabel('Field Coherence')
axes[0, 0].set_xscale('log')
axes[0, 0].grid(True, alpha=0.2)

# Plot entropy vs scale
axes[0, 1].plot(scales, entropies, 'o-', color='cyan', markersize=10, linewidth=2)
axes[0, 1].set_title('Entropy vs Fibonacci Scale')
axes[0, 1].set_xlabel('Fibonacci Scale')
axes[0, 1].set_ylabel('Field Entropy')
axes[0, 1].set_xscale('log')
axes[0, 1].grid(True, alpha=0.2)

# Golden ratio convergence
fib_ratios = [fib_scales[i+1] / fib_scales[i] for i in range(len(fib_scales)-1)]
axes[1, 0].plot(range(len(fib_ratios)), fib_ratios, 'o-', color='magenta', markersize=10)
axes[1, 0].axhline(y=PHI, color='gold', linestyle='--', linewidth=2, label=f'φ = {PHI:.6f}')
axes[1, 0].set_title('Fibonacci Ratio Convergence')
axes[1, 0].set_xlabel('Fibonacci Index')
axes[1, 0].set_ylabel('Ratio F(n+1)/F(n)')
axes[1, 0].set_ylim(1.4, 1.8)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.2)

# Energy wells at Fibonacci scales
x = np.linspace(-2, 2, 200)
energy_wells = np.zeros_like(x)

colors = plt.cm.viridis(np.linspace(0, 1, 6))
for i, scale in enumerate(fib_scales[:6]):
    well = -scale * np.exp(-(x**2) / (2 * (scale/PHI)**2))
    energy_wells += well
    axes[1, 1].plot(x, well, color=colors[i], alpha=0.6, label=f'F_{i} = {scale}')

axes[1, 1].plot(x, energy_wells, 'w-', linewidth=3, label='Total')
axes[1, 1].set_title('Fibonacci Energy Wells')
axes[1, 1].set_xlabel('Position')
axes[1, 1].set_ylabel('Potential Energy')
axes[1, 1].legend(loc='lower right')
axes[1, 1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('output/test5_fibonacci_resonance.png', dpi=200, bbox_inches='tight')
plt.close()

# Additional analysis: Spectral properties
print("\nSpectral analysis at Fibonacci scales...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Spectral gap evolution
ax1.plot(scales[:8], spectral_gaps[:8], 'o-', color='orange', markersize=10, linewidth=2)
ax1.set_title('Spectral Gap vs Fibonacci Scale')
ax1.set_xlabel('Fibonacci Scale')
ax1.set_ylabel('Spectral Gap (λ₁ - λ₂)')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.2)

# Resonance quality factor Q = coherence/entropy
Q_factors = [c/e for c, e in zip(coherences, entropies)]
ax2.plot(scales, Q_factors, 'o-', color='lime', markersize=10, linewidth=2)
ax2.set_title('Resonance Quality Factor')
ax2.set_xlabel('Fibonacci Scale')
ax2.set_ylabel('Q = Coherence/Entropy')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('output/test5_spectral_analysis.png', dpi=200, bbox_inches='tight')
plt.close()

# Multi-frequency interference pattern
print("\nGenerating multi-frequency interference patterns...")
t = torch.linspace(0, 10, 1000)
frequencies = [1/f for f in fib_scales[2:7]]  # Use inverse Fibonacci as frequencies
amplitudes = [1/np.sqrt(i+1) for i in range(len(frequencies))]  # Decay with index

# Create composite wave
composite_signal = torch.zeros_like(t)
individual_signals = []

for freq, amp in zip(frequencies, amplitudes):
    signal = amp * torch.sin(2 * np.pi * freq * t)
    composite_signal += signal
    individual_signals.append(signal)

# Plot interference pattern
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Individual components
for i, signal in enumerate(individual_signals[:3]):
    axes[0].plot(t[:200], signal[:200], alpha=0.7, 
                 label=f'F_{i+2} component')
axes[0].set_title('Individual Fibonacci-Frequency Components')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Amplitude')
axes[0].legend()
axes[0].grid(True, alpha=0.2)

# Composite signal
axes[1].plot(t[:500], composite_signal[:500], 'gold', linewidth=2)
axes[1].set_title('Composite Multi-Fibonacci Interference')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.2)

# Fourier transform to show frequency content
fft_result = torch.fft.rfft(composite_signal)
freqs = torch.fft.rfftfreq(len(composite_signal), 0.01)
axes[2].plot(freqs[:100], torch.abs(fft_result[:100]), 'cyan', linewidth=2)
axes[2].set_title('Frequency Spectrum (FFT)')
axes[2].set_xlabel('Frequency')
axes[2].set_ylabel('Magnitude')
axes[2].grid(True, alpha=0.2)

# Mark Fibonacci frequencies
for freq in frequencies:
    axes[2].axvline(x=freq, color='red', alpha=0.5, linestyle='--')

plt.tight_layout()
plt.savefig('output/test5_interference_patterns.png', dpi=200, bbox_inches='tight')
plt.close()

# Print analysis
print(f"\nFibonacci Resonance Analysis:")
print(f"- Golden ratio convergence achieved at n={np.where(np.abs(np.array(fib_ratios) - PHI) < 0.001)[0][0]}")
print(f"- Maximum coherence: {max(coherences):.4f} at scale {scales[coherences.index(max(coherences))]}")
print(f"- Minimum entropy: {min(entropies):.4f} at scale {scales[entropies.index(min(entropies))]}")
print(f"- Maximum Q-factor: {max(Q_factors):.4f} at scale {scales[Q_factors.index(max(Q_factors))]}")
print(f"\nSpectral Properties:")
print(f"- Maximum spectral gap: {max(spectral_gaps):.4f}")
print(f"- Spectral gap exhibits power-law scaling with exponent ≈ {np.polyfit(np.log(scales[:5]), np.log(spectral_gaps[:5]), 1)[0]:.2f}")

# Verify golden ratio relationships
print(f"\nGolden Ratio Verification:")
print(f"- φ² = φ + 1: {PHI**2:.6f} ≈ {PHI + 1:.6f} (error: {abs(PHI**2 - (PHI + 1)):.2e})")
print(f"- 1/φ = φ - 1: {1/PHI:.6f} ≈ {PHI - 1:.6f} (error: {abs(1/PHI - (PHI - 1)):.2e})")
print(f"- GAP = 1 - 1/φ: {GAP:.6f} ≈ {1 - 1/PHI:.6f} (verified)")

print("\nTest 5 complete. Outputs saved to:")
print("- output/test5_fibonacci_resonance.png")
print("- output/test5_spectral_analysis.png")
print("- output/test5_interference_patterns.png")
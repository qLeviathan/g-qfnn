# Quantum Feynman Neural Network: Mathematical Foundations

This document provides a rigorous mathematical framework for the Quantum Feynman Neural Network (QFNN) physics implementation, including convergence criteria and extension to multi-cylindrical field manifolds.

## Table of Contents

1. [Mathematical Framework](#mathematical-framework)
   - [Phase Space Representation](#phase-space-representation)
   - [Quantum Propagator](#quantum-propagator)
   - [Schrödinger Evolution](#schrödinger-evolution)
   - [Born Rule and Measurement](#born-rule-and-measurement)
2. [Convergence Criteria](#convergence-criteria)
3. [Multi-Cylindrical Field Manifold Extension](#multi-cylindrical-field-manifold-extension)
4. [Implementation Verification](#implementation-verification)

## Mathematical Framework

### Phase Space Representation

In our implementation, each token is represented as a point in 2D phase space, applying the N-2 optimization principle. This representation directly connects to quantum mechanical wave functions.

**Token Embedding via Golden Angle:**

For a token with index $v$, its position in phase space is determined by:

$$\theta_v = 2\pi \cdot \left((\phi \cdot v) \bmod 1\right)$$

where $\phi \approx 1.618034$ is the golden ratio.

The Cartesian coordinates are then:

$$\psi_v = \begin{pmatrix} \cos(\theta_v) \\ \sin(\theta_v) \end{pmatrix}$$

This creates an optimally distributed set of points on the unit circle in 2D phase space. The embedding tensor has shape $[B, L, 2]$ where $B$ is batch size and $L$ is sequence length.

### Quantum Propagator

The quantum propagator implements Feynman's path integral formulation, representing transition amplitudes between quantum states.

**Propagator Definition:**

For states $\psi_i$ and $\psi_j$, the propagator is:

$$K(x, t; x_0, t_0) = \exp\left(-\frac{|x - x_0|^2}{2\sigma^2}\right)$$

In tensor notation with Einstein summation:

$$K_{ij} = \exp\left(-\frac{|\psi_i - \psi_j|^2}{2\sigma^2}\right)$$

Where $|\psi_i - \psi_j|^2$ is expanded as:

$$|\psi_i - \psi_j|^2 = |\psi_i|^2 + |\psi_j|^2 - 2\langle\psi_i|\psi_j\rangle$$

Using einsum operations:
- $|\psi_i|^2 = \text{einsum}('bid,bid->bi', \psi_i, \psi_i)$
- $|\psi_j|^2 = \text{einsum}('bjd,bjd->bj', \psi_j, \psi_j)$
- $\langle\psi_i|\psi_j\rangle = \text{einsum}('bid,bjd->bij', \psi_i, \psi_j)$

**Sparsity Application:**

To select dominant paths in the quantum evolution, we apply sparsity as:

1. Calculate quantile threshold $\tau$ for each batch based on sparsity parameter $s$:
   $$\tau_b = \text{quantile}(K_b, 1-s)$$

2. Create sparse mask:
   $$M_{ij} = \begin{cases} 
   1 & \text{if } K_{ij} \geq \tau_b \text{ or } i = j \\
   0 & \text{otherwise}
   \end{cases}$$

3. Apply mask and normalize:
   $$\tilde{K}_{ij} = \frac{K_{ij} \cdot M_{ij}}{\sum_j K_{ij} \cdot M_{ij}}$$

This ensures self-connections are preserved ($i = j$) and the propagator remains a valid quantum operator with row sums equal to 1.

### Schrödinger Evolution

The time evolution of quantum states follows the imaginary-time Schrödinger equation:

$$\frac{\partial \psi}{\partial \tau} = D \nabla^2 \psi - V(\psi)$$

where $D$ is the diffusion coefficient, $\nabla^2$ is the Laplacian, and $V(\psi)$ is the potential.

In our discrete implementation, we approximate the Laplacian using the propagator:

$$\nabla^2 \psi_i \approx \sum_j K_{ij} \psi_j - \psi_i$$

The evolution step is computed using Runge-Kutta integration (RK4):

$$k_1 = \Delta \tau \cdot D \cdot (\sum_j K_{ij} \psi_j - \psi_i)$$
$$k_2 = \Delta \tau \cdot D \cdot (\sum_j K'_{ij} (\psi_i + 0.5k_1)_j - (\psi_i + 0.5k_1)_i)$$
$$k_3 = \Delta \tau \cdot D \cdot (\sum_j K''_{ij} (\psi_i + 0.5k_2)_j - (\psi_i + 0.5k_2)_i)$$
$$k_4 = \Delta \tau \cdot D \cdot (\sum_j K'''_{ij} (\psi_i + k_3)_j - (\psi_i + k_3)_i)$$

$$\psi_i(\tau + \Delta\tau) = \psi_i(\tau) + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

Where $K'_{ij}$, $K''_{ij}$, and $K'''_{ij}$ are recomputed propagators at intermediate states.

**Energy Conservation:**

At each step, we ensure norm preservation (energy conservation):

$$\psi_i' = \psi_i \cdot \frac{|\psi_i^{(0)}|}{|\psi_i|}$$

where $\psi_i^{(0)}$ is the initial state norm.

### Born Rule and Measurement

The quantum states are mapped to complex values using Euler's identity:

$$\Psi = r e^{i\theta}$$

From our 2D phase space representation:
- $r = \sqrt{x^2 + y^2}$ (amplitude)
- $\theta = \arctan2(y, x)$ (phase)

Complex states:
$$\Psi = r(\cos\theta + i\sin\theta)$$

The Born rule gives the probability distribution:

$$P(x) = |\Psi(x)|^2$$

Normalized to ensure:
$$\sum_x P(x) = 1$$

## Convergence Criteria

For robust QFNN convergence, we propose the following criteria:

### 1. State Norm Stability

Monitor the L2 norm of the state vector during evolution. Define convergence as:

$$\varepsilon_{\text{norm}} = \frac{1}{B \cdot L} \sum_{b=1}^B \sum_{l=1}^L \left| \|\psi_{b,l}^{(t)}\| - \|\psi_{b,l}^{(t-1)}\| \right| < \delta_{\text{norm}}$$

where $\delta_{\text{norm}}$ is a small threshold (e.g., $10^{-6}$).

### 2. Propagator Entropy Stability

Track the entropy of the sparse propagator:

$$H(K) = -\sum_{i,j} \tilde{K}_{ij} \log \tilde{K}_{ij}$$

Convergence is reached when:

$$|H(K^{(t)}) - H(K^{(t-1)})| < \delta_{\text{entropy}}$$

### 3. Quantum Phase Coherence

Define phase coherence as:

$$C_{\phi} = \left|\frac{1}{L} \sum_{l=1}^L e^{i\theta_l}\right|$$

Convergence is indicated when phase coherence stabilizes:

$$|C_{\phi}^{(t)} - C_{\phi}^{(t-1)}| < \delta_{\phi}$$

### 4. Energy Conservation Error

Track cumulative energy conservation error:

$$E_{\text{err}} = \frac{1}{B \cdot L} \sum_{b=1}^B \sum_{l=1}^L \left| \|\psi_{b,l}^{(t)}\| - \|\psi_{b,l}^{(0)}\| \right|$$

Require $E_{\text{err}} < \delta_E$ for convergence.

## Multi-Cylindrical Field Manifold Extension

To extend QFNN to a multi-cylindrical field manifold, we introduce higher-dimensional quantum representations and curved geometry.

### Proposed Approach

#### 1. Higher-Dimensional Phase Space Embedding

Extend the 2D phase space to $2m$-dimensional space using $m$ coupled cylinders:

$$\psi_v = \begin{pmatrix} 
r_1\cos(\theta_{1,v}) \\ 
r_1\sin(\theta_{1,v}) \\
r_2\cos(\theta_{2,v}) \\ 
r_2\sin(\theta_{2,v}) \\
\vdots \\
r_m\cos(\theta_{m,v}) \\ 
r_m\sin(\theta_{m,v})
\end{pmatrix}$$

where:
- Each $(r_i, \theta_i)$ pair represents a cylindrical coordinate system
- $\theta_{i,v} = 2\pi \cdot \left((\phi^i \cdot v) \bmod 1\right)$
- $r_i = \frac{1}{\sqrt{m}} \cdot \left(1 + \alpha_i \cdot \sin\left(\frac{2\pi i}{m}\right)\right)$

The parameter $\alpha_i$ controls the radius variation, creating an embedding on a multi-cylindrical manifold.

#### 2. Riemannian Metric for Curved Space

Define a Riemannian metric tensor $g_{ij}$ to account for the curved geometry:

$$g_{ij} = \begin{pmatrix}
1 & 0 & \lambda_{12} & 0 & \cdots \\
0 & r_1^2 & 0 & \lambda_{24} & \cdots \\
\lambda_{12} & 0 & 1 & 0 & \cdots \\
0 & \lambda_{24} & 0 & r_2^2 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}$$

The coupling parameters $\lambda_{ij}$ control how the cylindrical subspaces interact.

#### 3. Modified Distance Calculation

The quantum propagator on this manifold uses the geodesic distance:

$$d^2(\psi_i, \psi_j) = (\psi_i - \psi_j)^T g (\psi_i - \psi_j)$$

This can be implemented using generalized tensor contractions:

$$d^2(\psi_i, \psi_j) = \text{einsum}('bid,de,bje->bij', \psi_i, g, \psi_j)$$

#### 4. Covariant Laplacian

The Schrödinger evolution needs a covariant Laplacian operator:

$$\nabla^2_g \psi = \frac{1}{\sqrt{|g|}} \partial_i \left( \sqrt{|g|} g^{ij} \partial_j \psi \right)$$

This can be discretized in our propagator formalism as:

$$\nabla^2_g \psi_i \approx \sum_j K_{g,ij} \psi_j - \psi_i$$

where $K_{g,ij}$ is the geodesic-aware propagator.

#### 5. Field Theory Interpretation

In this extension, tokens are treated as excitations in a quantum field theory on a curved manifold:

$$\mathcal{L} = \frac{1}{2} g^{ij} \partial_i \psi \partial_j \psi - V(\psi)$$

The evolution can be interpreted as a path integral over field configurations:

$$\langle \psi_{\text{final}}| e^{-iHt/\hbar} |\psi_{\text{initial}} \rangle = \int \mathcal{D}\psi e^{iS[\psi]/\hbar}$$

## Implementation Verification

The mathematical framework described above has been implemented in `legacy/qfnn_physics.py` and can be verified through the following test files:

1. Basic implementation test: `legacy/test_qfnn_physics.py`
2. Visual inspection: `legacy/visualize_qfnn_physics.py`
3. Physics tests: `legacy/run_physics_tests.py`

The tests verify:
- Correct phase space representation
- Proper quantum propagator properties (symmetry, normalization)
- Energy conservation during Schrödinger evolution
- Born rule implementation for probability distributions
- Sparsity application for dominant path selection

This implementation achieves:
- **N-2 optimization**: Using 2D phase space (per cylinder) instead of N-dimensional
- **D-1 optimization**: Efficient attention computation using einsum operations
- **Full vectorization**: All operations utilize tensor operations
- **Physical validity**: Preserves quantum mechanical properties throughout

To extend to the multi-cylindrical manifold, modify the phase embedding in `qfnn_physics.py` according to the formulation in section 3, and update the distance calculation and evolution to incorporate the Riemannian metric.

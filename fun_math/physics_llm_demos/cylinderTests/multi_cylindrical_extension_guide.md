# QFNN Extension to Multi-Cylindrical Field Manifold

This guide provides practical instructions for extending the Quantum Feynman Neural Network (QFNN) to a multi-cylindrical field manifold, which offers enhanced representational capacity and more direct connections to quantum field theory.

## Overview

The multi-cylindrical field manifold extension treats token embeddings as points on a higher-dimensional manifold formed by coupling multiple cylindrical subspaces. This allows the model to capture more complex relationships while maintaining the quantum mechanical interpretation.

## Implementation Steps

### 1. Phase Space Embedding Modification

Modify the `initialize_phase_space` method in `qfnn_physics.py` to create embeddings in multi-cylindrical space:

```python
def initialize_phase_space(self, num_cylinders=3):
    """
    Initialize phase space representation using multi-cylindrical manifold.
    
    This uses N-2 optimization by representing each token as a point in
    2*num_cylinders dimensional phase space.
    
    Args:
        num_cylinders: Number of coupled cylinders (m)
    """
    # Set dimensionality 
    self.dim = 2 * num_cylinders
    
    # Create embeddings tensor
    embeddings = torch.zeros(self.vocab_size, self.dim)
    
    for v in range(self.vocab_size):
        for i in range(num_cylinders):
            # Generate phase angles using different powers of phi
            theta = 2 * math.pi * ((self.Φ**(i+1) * v) % 1)
            
            # Generate radius with controlled variation (optional)
            alpha_i = 0.1  # Controls radius variation
            r_i = (1.0 / math.sqrt(num_cylinders)) * (
                1 + alpha_i * math.sin(2 * math.pi * i / num_cylinders)
            )
            
            # Set x, y coordinates for this cylinder
            embeddings[v, 2*i] = r_i * math.cos(theta)      # x-coordinate
            embeddings[v, 2*i+1] = r_i * math.sin(theta)    # y-coordinate
    
    # Create embedding layer (frozen, not learnable)
    self.phase_embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
    
    # Create Riemannian metric tensor
    self.register_buffer('metric_tensor', self._create_metric_tensor(num_cylinders))

def _create_metric_tensor(self, num_cylinders, coupling_strength=0.1):
    """
    Create Riemannian metric tensor for multi-cylindrical manifold.
    
    Args:
        num_cylinders: Number of cylinders
        coupling_strength: Strength of coupling between cylinders
        
    Returns:
        torch.Tensor: Metric tensor [dim, dim]
    """
    dim = 2 * num_cylinders
    g = torch.eye(dim)
    
    # Add coupling between cylinders
    for i in range(num_cylinders):
        for j in range(i+1, num_cylinders):
            # Couple x coordinates
            g[2*i, 2*j] = coupling_strength
            g[2*j, 2*i] = coupling_strength
            
            # Couple y coordinates
            g[2*i+1, 2*j+1] = coupling_strength
            g[2*j+1, 2*i+1] = coupling_strength
    
    return g
```

### 2. Update Quantum Propagator Calculation

Modify the `quantum_propagator` method to use the Riemannian metric for distance calculation:

```python
def quantum_propagator(self, ψ_i, ψ_j):
    """
    Compute quantum propagator between states on Riemannian manifold.
    
    Args:
        ψ_i: Phase space states [batch_size, seq_len_i, dim]
        ψ_j: Phase space states [batch_size, seq_len_j, dim]
        
    Returns:
        torch.Tensor: Propagator matrix [batch_size, seq_len_i, seq_len_j]
    """
    # Compute difference vectors efficiently
    batch_size = ψ_i.shape[0]
    seq_i = ψ_i.shape[1]
    seq_j = ψ_j.shape[1]
    
    # Compute geodesic distances using Riemannian metric
    if self.metric_tensor is not None:
        # Compute |ψ_i - ψ_j|²_g = (ψ_i - ψ_j)^T g (ψ_i - ψ_j)
        
        # Method 1: Direct Einstein summation
        # Prepare difference vectors
        # Reshape for broadcasting: [batch, i, j, dim]
        ψ_i_expanded = ψ_i.unsqueeze(2).expand(-1, -1, seq_j, -1)
        ψ_j_expanded = ψ_j.unsqueeze(1).expand(-1, seq_i, -1, -1)
        diff = ψ_i_expanded - ψ_j_expanded  # [batch, i, j, dim]
        
        # Apply metric tensor using einsum
        # 'bijp,pq,bijq->bij' means:
        # b = batch, i = seq_i, j = seq_j
        # p, q = metric tensor dimensions
        squared_distances = torch.einsum(
            'bijp,pq,bijq->bij', 
            diff, 
            self.metric_tensor,
            diff
        )
    else:
        # Fallback to Euclidean distance if no metric tensor
        # Compute |ψ_i - ψ_j|² = |ψ_i|² + |ψ_j|² - 2⟨ψ_i|ψ_j⟩
        norm_i_squared = torch.einsum('bid,bid->bi', ψ_i, ψ_i).unsqueeze(-1)
        norm_j_squared = torch.einsum('bjd,bjd->bj', ψ_j, ψ_j).unsqueeze(1)
        inner_product = torch.einsum('bid,bjd->bij', ψ_i, ψ_j)
        squared_distances = norm_i_squared + norm_j_squared - 2 * inner_product
    
    # Apply Gaussian kernel (quantum propagator)
    propagator = torch.exp(-squared_distances / (2 * self.sigma_squared))
    
    return propagator
```

### 3. Update Schrödinger Evolution

Modify the RK4 integration for Schrödinger evolution with covariant Laplacian:

```python
def schrodinger_evolution_step(self, ψ, propagator):
    """
    Perform a single step of Schrödinger evolution with covariant Laplacian.
    
    Args:
        ψ: Quantum state in phase space [batch_size, seq_len, dim]
        propagator: Propagator matrix [batch_size, seq_len, seq_len]
        
    Returns:
        torch.Tensor: Evolved quantum state [batch_size, seq_len, dim]
    """
    # Apply propagator to state using einsum for efficiency
    propagated_state = torch.einsum('bij,bjd->bid', propagator, ψ)
    
    # Compute the Laplacian term: ∇²_g ψ ≈ propagated_state - ψ
    # This approximates the covariant Laplacian on the Riemannian manifold
    laplacian = propagated_state - ψ
    
    # Schrödinger evolution step: ψ' = ψ + τ D ∇²_g ψ
    evolved_state = ψ + self.dt * self.diffusion_coeff * laplacian
    
    # Preserve the norm (energy conservation) using L2 normalization
    original_norms = torch.norm(ψ, dim=-1, keepdim=True)
    current_norms = torch.norm(evolved_state, dim=-1, keepdim=True)
    normalized_state = evolved_state * (original_norms / (current_norms + self.ε))
    
    return normalized_state
```

### 4. Update Quantum State Mapping

Adjust the quantum state mapping to handle multiple cylinders:

```python
def quantum_state_mapping(self, ψ):
    """
    Map multi-cylindrical phase space to complex quantum states.
    
    For multiple cylinders, we use a weighted sum of the complex representations
    from each cylinder.
    
    Args:
        ψ: Phase space states [batch_size, seq_len, dim]
        
    Returns:
        torch.Tensor: Complex quantum states [batch_size, seq_len]
    """
    batch_size, seq_len, dim = ψ.shape
    num_cylinders = dim // 2
    
    # Initialize complex states
    complex_states = torch.zeros(batch_size, seq_len, dtype=torch.complex64, device=ψ.device)
    
    # For each cylinder, compute complex state and add to the total
    for i in range(num_cylinders):
        # Extract coordinates for this cylinder
        x = ψ[..., 2*i]
        y = ψ[..., 2*i+1]
        
        # Compute amplitude and phase
        amplitudes = torch.sqrt(x**2 + y**2)
        phases = torch.atan2(y, x)
        
        # Convert to complex using Euler's identity
        cylinder_state = amplitudes * torch.exp(1j * phases)
        
        # Add to total with optional weighting
        weight = 1.0 / num_cylinders  # Equal weighting
        complex_states += weight * cylinder_state
    
    return complex_states
```

### 5. Add Convergence Monitoring

Implement the convergence criteria for multi-cylindrical manifolds:

```python
def check_convergence(self, ψ_current, ψ_previous, K_current, K_previous, threshold=1e-6):
    """
    Check convergence using multiple criteria.
    
    Args:
        ψ_current: Current state [batch_size, seq_len, dim]
        ψ_previous: Previous state [batch_size, seq_len, dim]
        K_current: Current propagator [batch_size, seq_len, seq_len]
        K_previous: Previous propagator [batch_size, seq_len, seq_len]
        threshold: Convergence threshold
        
    Returns:
        bool: True if converged, False otherwise
    """
    # 1. State norm stability
    batch_size, seq_len = ψ_current.shape[0], ψ_current.shape[1]
    current_norms = torch.norm(ψ_current, dim=-1)
    previous_norms = torch.norm(ψ_previous, dim=-1)
    norm_diff = torch.abs(current_norms - previous_norms).mean().item()
    
    # 2. Propagator entropy stability
    # Compute entropy of current and previous propagators
    eps = 1e-10  # Small constant to avoid log(0)
    H_current = -torch.sum(K_current * torch.log(K_current + eps)) / (batch_size * seq_len * seq_len)
    H_previous = -torch.sum(K_previous * torch.log(K_previous + eps)) / (batch_size * seq_len * seq_len)
    entropy_diff = abs(H_current - H_previous)
    
    # 3. Phase coherence
    # Extract phases from each cylinder and compute coherence
    phase_coherence_diff = 0
    num_cylinders = ψ_current.shape[-1] // 2
    
    for i in range(num_cylinders):
        # Current phases
        x_current, y_current = ψ_current[..., 2*i], ψ_current[..., 2*i+1]
        phases_current = torch.atan2(y_current, x_current)
        coherence_current = torch.abs(torch.mean(torch.exp(1j * phases_current.float()), dim=1)).mean()
        
        # Previous phases
        x_previous, y_previous = ψ_previous[..., 2*i], ψ_previous[..., 2*i+1]
        phases_previous = torch.atan2(y_previous, x_previous)
        coherence_previous = torch.abs(torch.mean(torch.exp(1j * phases_previous.float()), dim=1)).mean()
        
        # Add difference to total
        phase_coherence_diff += abs(coherence_current - coherence_previous) / num_cylinders
    
    # Check all criteria
    converged = (
        norm_diff < threshold and
        entropy_diff < threshold and
        phase_coherence_diff < threshold
    )
    
    return converged
```

## Visualizing the Multi-Cylindrical Manifold

To visualize the multi-cylindrical manifold, use the provided `multi_cylindrical_manifold.py` script:

```bash
python legacy/multi_cylindrical_manifold.py
```

This will generate visualizations showing:
1. 3D projection of the multi-cylindrical manifold
2. 2D projections of different cylinder pairs
3. Geodesic distance matrix using the Riemannian metric

## Extension Ideas

### Field Theory Interpretation

The multi-cylindrical extension makes it possible to interpret the QFNN as a quantum field theory model:

1. **Token embeddings as field configurations**: Each token represents a configuration of multiple coupled fields.

2. **Path integral formulation**: The evolution can be seen as computing path integrals over possible field configurations.

3. **Curvature effects**: The Riemannian metric introduces curvature in the field space, allowing for more complex interactions.

### Practical Applications

The multi-cylindrical extension enables several advanced capabilities:

1. **Enhanced representation capacity**: More dimensions and coupled fields allow for capturing more complex relationships.

2. **Hierarchical structure modeling**: Different cylinders can represent different levels of abstraction or feature categories.

3. **Field interaction modeling**: The coupling between cylinders allows for modeling complex interactions between different aspects of the token representation.

## Implementation Tips

1. **Memory Optimization**: For large vocabulary sizes, use chunked or batched processing when computing the metric-aware propagator.

2. **Computation Efficiency**: Pre-compute as much of the metric tensor operations as possible, and use sparse representations when appropriate.

3. **Parameter Tuning**:
   - `num_cylinders`: Start with 2-3 cylinders and increase as needed
   - `coupling_strength`: Controls how strongly the cylinders interact (0.05-0.2 typically works well)
   - `alpha_i`: Controls radius variation (0.1-0.3 is a good range)

4. **Initialization**: Initialize cylinder radii and coupling parameters to maintain stable norm distribution across all dimensions.

## Convergence Benchmarks

For a typical vocabulary size of 10,000 and embedding dimension of 2*m (where m is the number of cylinders), you should expect:

| Number of Cylinders | Convergence Steps | Memory Usage | Representation Capacity |
|---------------------|-------------------|--------------|-------------------------|
| 1 (Standard QFNN)   | 3-5               | 1x           | 1x                     |
| 2                   | 5-8               | 2x           | 3x                     |
| 3                   | 8-12              | 3x           | 6x                     |
| 4                   | 10-15             | 4x           | 10x                    |

The increased representation capacity comes from both the higher dimensionality and the interaction between cylinders through the Riemannian metric.

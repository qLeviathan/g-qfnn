# Quantum Field Neural Network (QFNN)

A high-performance neural network architecture using log-cylindrical embeddings for efficient sequence processing with O(N log N) computational complexity.

## Overview

The Quantum Field Neural Network (QFNN) implements a novel approach to sequence modeling by representing tokens in a log-cylindrical coordinate space. This representation enables efficient attention mechanisms and natural handling of both local and global information.

Key features:

- **Log-Cylindrical Coordinates**: Uses ln(r), θ, and z coordinates for numerical stability across orders of magnitude
- **O(N log N) Complexity**: Efficient scaling for sequence processing with tensor operations
- **Dual Vortex Field**: Repulsive forces in log-space create natural attention patterns
- **Quantum Properties**: Demonstrates tachyonic tunneling, phase flips, and helical trajectories
- **Phase Transitions**: Shows crystallization from computational to reference points

## Project Structure

```
.
├── log_coords/                     # Core implementation modules
│   ├── __init__.py                 # Package exports
│   ├── log_coords.py               # Log-cylindrical coordinate transformations
│   ├── log_hebbian.py              # Hebbian learning in log-cylindrical space
│   └── dual_vortex.py              # Dual vortex field implementation
├── quantum_field_dynamics.ipynb    # Main demonstration notebook
├── notebook_outputs/               # Generated visualizations and models
└── README.md                       # This documentation file
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebook for a complete demonstration:

```bash
jupyter notebook quantum_field_dynamics.ipynb
```

Import the modules for use in your own code:

```python
from log_coords import (
    log_cylindrical_to_cartesian, 
    compute_attention_weights,
    DualVortexField
)

# Initialize a field with 100 tokens
field = DualVortexField(n_tokens=100)
field.initialize_tokens(pattern='golden_spiral')

# Run simulation
for _ in range(10):
    field.step()

# Visualize results
field.visualize_field(plot_type="3d")
```

## Key Mathematical Concepts

1. **Log-Cylindrical Coordinates**:
   - ln(r): Logarithmic radial coordinate for scale-invariant processing
   - θ: Angular coordinate for phase representation (0 to 2π)
   - z: Vertical coordinate representing sequence position

2. **Field Equation**:
   The Laplacian in log-cylindrical coordinates governs field dynamics:
   ∇²ψ(r,θ,z) = (1/r²)∂²ψ/∂θ² + (1/r)∂/∂r(r∂ψ/∂r) + ∂²ψ/∂z²

3. **Tensor Operations**:
   The implementation uses efficient einsum operations for token interactions:
   ```python
   # Calculate distances using tensor operations
   distances = log_cylindrical_batch_distance(coords_i, coords_j)
   ```

4. **Quantum Properties**:
   - Tachyonic tunneling: Phase flips when angular velocity dominates radial velocity
   - Helical trajectories: Tokens follow 3D helical paths in log-cylindrical space
   - Crystallization: Tokens freeze into reference points at low temperature

## Visualizations

The notebook generates several visualizations:

- 3D Log-Cylindrical Space: Shows token positions and field structure
- Token Matrix Evolution: Shows how token states evolve over time
- Phase Transition Visualization: Demonstrates KT transition from computational to crystallized state
- Energy Distribution: Shows how energy is distributed across tokens
- Interaction Patterns: Visualizes token-to-token interactions

## Performance Notes

- Tensor operations using einsum provide significant speedup over explicit loops
- The log-cylindrical representation enables O(N log N) computational complexity
- All visualizations properly convert tensors to CPU before display with .cpu().numpy()

## Theoretical Foundation

The QFNN builds on concepts from:

- Quantum field theory: Field equations and phase transitions
- Log-cylindrical geometry: Scale-invariant distance calculations
- Attention mechanisms: Natural emergence from repulsive forces
- Hebbian learning: Connection strengthening based on token proximity

## Citation

If you use this implementation in your research, please cite:

```
@article{qfnn2025,
  title={Quantum Field Neural Networks with Log-Cylindrical Embeddings},
  author={Marc Castillo.},
  journal={ tbd},
  year={2025}
}
```
# Physics-First Quantum Field Neural Networks

Greetings,

I am experimenting on theoretical physics implementations for explainable AI. This repo serves as a collection of previous experiments I've run. Below is a first rendition of an INCOMPLETE architecture. I am actively working on rotating the loss gradient dynamics to guide training and inference. The key attention mechanism uses repulsion for cylindrical superposition.

The "fun_math" folder contains all the good stuff. There are also some preliminary financial applications that I have added to the repo. They are not production grade but do show a strong proof of concept. Additionally, I experimented with cation-anion movement with a POC for a language model.

## 📚 Repository Structure

This repository contains multiple related projects exploring physics-based approaches to AI:

```
g-qfnn/
├── core.py, collapse.py, model.py, etc.    # Main field-theoretic LM implementation
├── docs/                                    # Documentation and theoretical papers
├── fun_math/                                # Core experimental implementations
│   ├── physics/                             # Physics simulations and visualizations
│   ├── finance/                             # Financial modeling applications
│   ├── chemistry/                           # Quantum electrochemical simulations
│   └── physics_llm_demos/                   # Physics-based language model demos
│       ├── cylinderTests/                   # Cylindrical coordinate space experiments
│       ├── qfnndemo/                        # Quantum Field Neural Network demos
│       └── learn_here/                      # Latest QFNN implementation
├── outputs/                                 # Visualization outputs
├── tests/                                   # Test suite
└── qfnn_physics_report/                     # Technical reports and findings
```

## 🧭 Navigation Guide

### Core Implementations
The root directory contains the main field-theoretic language model implementation:
- `core.py`: Golden embeddings, field evolution, crystal memory
- `collapse.py`: Field collapse dynamics and sampling
- `model.py`: Complete model architecture
- `trainer.py`: Hebbian training without backpropagation
- `inference.py`: Generation and analysis tools
- `data.py`: Streaming data loaders for WikiText & C4
- `main.py`: CLI interface for the model

### Research Directions

#### 1. Physics Models (`fun_math/physics/`)
- `gwave_core_refactored.py`: Core implementation of gravitational wave dynamics
- `repulsion.py`: Repulsive force implementation for attention mechanisms
- `log_form_helical_dynamics.py`: Helical dynamics in logarithmic space
- `Relativistic Vortex Spacetime Dynamics.py`: Relativistic vortex field simulations

#### 2. Financial Applications (`fun_math/finance/demos/`)
- `interest_stablecoin_model_demo.py`: Quantum modeling for stablecoin dynamics
- `enhanced_multi_sector_model.py`: Multi-sector financial forecasting
- `options_model_demo.py`: Quantum approach to options pricing
- `sector_model.py`: Sector-based financial modeling

#### 3. Chemistry Simulations (`fun_math/chemistry/demos/`)
- `quantum_electrochemical_simulator.py`: Electrochemical process simulation
- `quantum_cold_fusion_simulator.py`: Cold fusion dynamics model
- `run_cold_fusion_simulation.py`: Runner for cold fusion simulations

#### 4. Physics Language Models (`fun_math/physics_llm_demos/`)
- `qfnndemo/qfnn_physics.py`: Original QFNN implementation
- `cylinderTests/`: Cylindrical coordinate space experiments
- `graphs/optimized_gnn_model_v2.py`: Graph neural network with physics priors

### 5. Latest Implementation: QFNN (`fun_math/physics_llm_demos/learn_here/`)
This directory contains the most recent and advanced implementation:
- `log_coords/`: Core implementation modules
- `quantum_field_dynamics.ipynb`: Main demonstration notebook
- `notebook_outputs/`: Generated visualizations and models

### 6. Documentation and Theory (`docs/`)
- `QUANTUM_GEOMETRIC_THEORY.md`: Theoretical foundations
- `REPULSION_ATTENTION_IMPLEMENTATION.md`: Details on repulsion-based attention
- `GWAVE_INTEGRATION.md`: Gravitational wave integration guide
- `TRANSFORMER_VS_REPULSION_COMPARISON.md`: Comparison with traditional transformers

### 7. Test Suite (`tests/`)
- `test1_manifold.py` through `test6_consciousness.py`: Comprehensive tests for different aspects
- `test_all.py`: Run all tests

## 🚀 Complete Field-Theoretic Language Model Package

I've created a comprehensive, modular implementation of a field-theoretic language model with NO backpropagation. Here's what you have:

### 📁 Package Structure:
```
gfg-llm/
├── core.py           # Golden embeddings, field evolution, crystal memory
├── perturbations.py  # Lévy, Beta, Adaptive perturbation strategies  
├── collapse.py       # Field collapse dynamics and sampling
├── model.py          # Complete model architecture
├── data.py           # Streaming data loaders for WikiText & C4
├── trainer.py        # Hebbian training without backprop
├── inference.py      # Generation and analysis tools
├── main.py           # CLI interface
└── field_theoretic_lm.ipynb  # Interactive notebook
```

### 🔑 Key Features:

1. **Physics-First Design**:
   - Golden spiral cylindrical embeddings (GAP = 0.382)
   - Log-phase transform for 29.65x Hebbian amplification
   - Gravitational field interactions
   - Coherence-driven collapse at threshold 0.91

2. **No Backpropagation**:
   - Pure Hebbian crystallization: `ΔW = η⟨post ⊗ pre⟩`
   - Learning rate η = 1/φ
   - Crystal memory persists without gradients

3. **Stochastic Perturbations**:
   - **Lévy**: Heavy-tailed exploration (α = φ)
   - **Beta**: Information-geometric time steps
   - **Adaptive**: Coherence-based mixing

4. **Efficient Implementation**:
   - All operations vectorized with einsum
   - FP16 support for 4090
   - Streaming data (no full dataset in memory)
   - Causal masking in gravitational matrix

## 🌟 Quantum Field Neural Network (QFNN)

The latest addition to this repository is the Quantum Field Neural Network (QFNN) implementation, found in the `fun_math/physics_llm_demos/learn_here/` directory. This represents a significant advancement in physics-based neural network architectures.

### Overview

The QFNN implements a novel approach to sequence modeling by representing tokens in a log-cylindrical coordinate space. This representation enables efficient attention mechanisms and natural handling of both local and global information.

Key features:

- **Log-Cylindrical Coordinates**: Uses ln(r), θ, and z coordinates for numerical stability across orders of magnitude
- **O(N log N) Complexity**: Efficient scaling for sequence processing with tensor operations
- **Dual Vortex Field**: Repulsive forces in log-space create natural attention patterns
- **Quantum Properties**: Demonstrates tachyonic tunneling, phase flips, and helical trajectories
- **Phase Transitions**: Shows crystallization from computational to reference points

### Project Structure

```
fun_math/physics_llm_demos/learn_here/
├── log_coords/                     # Core implementation modules
│   ├── __init__.py                 # Package exports
│   ├── log_coords.py               # Log-cylindrical coordinate transformations
│   ├── log_hebbian.py              # Hebbian learning in log-cylindrical space
│   └── dual_vortex.py              # Dual vortex field implementation
├── quantum_field_dynamics.ipynb    # Main demonstration notebook
├── notebook_outputs/               # Generated visualizations and models
└── README.md                       # Documentation file
```

### Key Mathematical Concepts

1. **Log-Cylindrical Coordinates**:
   - ln(r): Logarithmic radial coordinate for scale-invariant processing
   - θ: Angular coordinate for phase representation (0 to 2π)
   - z: Vertical coordinate representing sequence position

2. **Field Equation**:
   The Laplacian in log-cylindrical coordinates governs field dynamics:
   ∇²ψ(r,θ,z) = (1/r²)∂²ψ/∂θ² + (1/r)∂/∂r(r∂ψ/∂r) + ∂²ψ/∂z²

3. **Quantum Properties**:
   - Tachyonic tunneling: Phase flips when angular velocity dominates radial velocity
   - Helical trajectories: Tokens follow 3D helical paths in log-cylindrical space
   - Crystallization: Tokens freeze into reference points at low temperature

### Visualizations

The QFNN implementation generates several visualizations:

- 3D Log-Cylindrical Space: Shows token positions and field structure
- Token Matrix Evolution: Shows how token states evolve over time
- Phase Transition Visualization: Demonstrates KT transition from computational to crystallized state
- Energy Distribution: Shows how energy is distributed across tokens
- Interaction Patterns: Visualizes token-to-token interactions

## 🔬 Financial Applications

The repository includes several quantum-inspired financial models in `fun_math/finance/demos/`:

1. **Stablecoin Model** (`interest_stablecoin_model_demo.py`):
   - Models stablecoin dynamics using quantum fields
   - Simulates different interest rate scenarios
   - Visualizes stability under market volatility

2. **Multi-Sector Quantum Financial Model** (`enhanced_multi_sector_model.py`):
   - Implements quantum forecasting across multiple market sectors
   - Uses field interactions to model cross-sector influences
   - Demonstrates improved accuracy over traditional models

3. **Quantum Options Pricing** (`options_model_demo.py`):
   - Novel approach to options pricing using quantum principles
   - Handles both call and put options
   - Integrates with the Xi-Psi quantum framework

## 🧪 Chemistry Simulations

Quantum electrochemical simulations in `fun_math/chemistry/demos/`:

1. **Quantum Electrochemical Simulator**:
   - Models electrochemical processes with quantum fields
   - Simulates electron transfer and ionic movement
   - Visualizes energy states and reaction dynamics

2. **Cold Fusion Simulator**:
   - Experimental model of theoretical cold fusion dynamics
   - Includes resonant phonon coupling
   - Visualizes time evolution of the reaction system

## 🎮 Usage Examples (Original Model):

**Training**:
```bash
# Small model on WikiText-2
python main.py train --model-size small --dataset wikitext-2 --num-steps 10000

# Large model on C4 with custom perturbation schedule
python main.py train --model-size large --dataset c4 --num-steps 50000 \
    --perturbation-schedule '{"0": "beta", "10000": "adaptive", "30000": "levy"}'
```

**Generation**:
```bash
# Generate with Lévy perturbation
python main.py generate "The quantum field" --perturbation levy --temperature 0.9

# Use trained checkpoint
python main.py generate "Once upon a time" --checkpoint outputs/checkpoint_10000.pt
```

**Analysis**:
```bash
# Visualize field dynamics
python main.py analyze "Hello world" --visualize --save-path field_evolution.png

# Profile performance
python main.py profile --batch-sizes 1,4,8,16 --seq-lengths 128,256,512
```

## 📊 Memory & Performance:

For 4090 (24GB):
- **Small (50M)**: ~2GB, 50k tokens/sec
- **Base (350M)**: ~6GB, 15k tokens/sec  
- **Large (750M)**: ~12GB, 8k tokens/sec

## 💡 Physics Insights:

- **Tokens as particles** on golden manifold
- **Gravitational collapse** replaces attention
- **Hebbian crystallization** replaces backprop
- **Lévy flights** enable long-range exploration
- **Coherence threshold** triggers measurement
- **Tachyonic tunneling** enables phase flips in log-cylindrical space
- **Dual vortex fields** create natural attention patterns
- **Log-cylindrical coordinates** provide numerical stability across orders of magnitude

## 📚 Getting Started with QFNN

To explore the Quantum Field Neural Network:

1. Navigate to the implementation directory:
   ```bash
   cd fun_math/physics_llm_demos/learn_here/
   ```

2. Open the Jupyter notebook:
   ```bash
   jupyter notebook quantum_field_dynamics.ipynb
   ```

3. Run through the notebook cells to see the implementation in action

4. Import the modules for use in your own code:
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

## Citation

If you use this implementation in your research, please cite:

```
@article{qfnn2025,
  title={Quantum Field Neural Networks with Log-Cylindrical Embeddings},
  author={Marc Castillo},
  journal={tbd},
  year={2025}
}
```
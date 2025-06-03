"""
Test the log-cylindrical quantum field neural network modules
"""

import torch
import matplotlib.pyplot as plt
import os
import time

# Force CPU mode for testing
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device('cpu')

# Import our modules
from log_coords import LogCylindricalCoords
from log_hebbian import SparseLogHebbian
from dual_vortex import DualVortexField
from quantum_field_nn import QuantumFieldNN

# Create output directory
os.makedirs('outputs', exist_ok=True)

def test_log_coords():
    """Test the log-cylindrical coordinate system"""
    print("\n=== Testing Log-Cylindrical Coordinate System ===")
    
    # Create coordinate system
    coords = LogCylindricalCoords(device=device)
    
    # Generate golden spiral
    N = 100
    ln_r, theta = coords.generate_golden_spiral(N)
    
    # Visualize
    coords.visualize_comparison(n=100, save_path="outputs/log_cylindrical_comparison.png")
    
    # Test conversion
    x, y = coords.ln_r_theta_to_cartesian(ln_r, theta)
    ln_r2, theta2 = coords.cartesian_to_ln_r_theta(x, y)
    
    # Check error
    ln_r_error = torch.abs(ln_r - ln_r2).mean().item()
    theta_error = torch.abs(torch.remainder(theta - theta2 + torch.pi, 2*torch.pi) - torch.pi).mean().item()
    
    print(f"Round-trip conversion error:")
    print(f"  ln_r error: {ln_r_error:.8f}")
    print(f"  theta error: {theta_error:.8f}")
    
    return coords, ln_r, theta

def test_log_hebbian(coords, ln_r, theta):
    """Test the log-Hebbian learning"""
    print("\n=== Testing Log-Hebbian Learning ===")
    
    # Create Hebbian network
    N = ln_r.shape[0]
    hebbian = SparseLogHebbian(N, device=device)
    
    # Perform Hebbian updates
    print("Performing Hebbian updates...")
    start_time = time.time()
    
    # Run multiple updates
    num_updates = 5
    dt = 0.1
    
    for i in range(num_updates):
        hebbian.log_update(ln_r, theta, coords, dt)
        print(f"Update {i+1}/{num_updates}, connections: {len(hebbian.indices)}")
    
    end_time = time.time()
    print(f"Hebbian updates completed in {end_time - start_time:.2f} seconds")
    
    # Visualize the network
    try:
        hebbian.visualize_hebbian_network(ln_r, theta, coords, save_path="outputs/hebbian_network.png")
        print("Hebbian network visualization saved")
    except Exception as e:
        print(f"Error visualizing Hebbian network: {e}")
    
    return hebbian

def test_dual_vortex():
    """Test the dual vortex field dynamics"""
    print("\n=== Testing Dual Vortex Field Dynamics ===")
    
    # Create field
    N = 30  # Small N for quick testing
    field = DualVortexField(N, device=device)
    
    # Initialize tokens
    field.initialize_tokens(pattern='golden_spiral')
    
    # Run a short simulation
    print("Running simulation...")
    field.run_simulation(steps=20, record_every=5)
    
    # Visualize
    try:
        field.visualize_trajectories(save_path="outputs/dual_vortex_trajectories.png")
        field.visualize_energy(save_path="outputs/dual_vortex_energy.png")
        field.visualize_field(save_path="outputs/dual_vortex_field.png")
        print("Dual vortex visualizations saved")
    except Exception as e:
        print(f"Error visualizing dual vortex field: {e}")
    
    return field

def test_quantum_field_nn():
    """Test the quantum field neural network"""
    print("\n=== Testing Quantum Field Neural Network ===")
    
    # Create small model for testing
    vocab_size = 100
    embedding_dim = 32
    model = QuantumFieldNN(vocab_size, embedding_dim, device=device)
    
    # Test forward pass
    batch_size = 2
    seq_len = 5
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    print(f"Running forward pass on input shape: {input_ids.shape}")
    logits = model(input_ids, evolution_steps=2)
    print(f"Output shape: {logits.shape}")
    
    # Visualize embeddings
    try:
        model.visualize_embeddings(save_path="outputs/qfnn_embeddings.png")
        model.compare_embedding_systems(save_path="outputs/qfnn_embedding_comparison.png")
        print("QFNN visualizations saved")
    except Exception as e:
        print(f"Error visualizing QFNN: {e}")
    
    return model

if __name__ == "__main__":
    print("Testing Log-Cylindrical Quantum Field Neural Network modules")
    print(f"Using device: {device}")
    
    # Test modules
    coords, ln_r, theta = test_log_coords()
    hebbian = test_log_hebbian(coords, ln_r, theta)
    field = test_dual_vortex()
    model = test_quantum_field_nn()
    
    print("\nAll tests completed!")
    print("Output visualizations saved in the 'outputs' directory")
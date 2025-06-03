#!/usr/bin/env python3
"""
Runner for Quantum Electrochemical Simulator

This script runs the electrochemical simulation examples from the quantum_electrochemical module
with additional debug information and learning rate visualization
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

# Ensure imports can be found regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function for running electrochemical simulations"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Quantum Electrochemical Simulator")
    parser.add_argument('--sim_type', type=str, default='diffusion', 
                        choices=['diffusion', 'reaction', 'migration'],
                        help='Type of simulation to run')
    parser.add_argument('--steps', type=int, default=200,
                        help='Number of simulation steps')
    parser.add_argument('--save_plot', action='store_true',
                        help='Save the simulation plots')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Hebbian learning rate (default: 0.005)')
    parser.add_argument('--decay_rate', type=float, default=0.999,
                        help='Weight decay rate (default: 0.999)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    args = parser.parse_args()
    
    print(f"Starting {args.sim_type} simulation with {args.steps} steps")
    print(f"Learning rate: {args.learning_rate}, Decay rate: {args.decay_rate}")
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Import the simulator
    try:
        from quantum_electrochemical.quantum_electrochemical_simulator import (
            QuantumElectrochemicalSimulator, plot_simulation_results
        )
        print("Successfully imported quantum electrochemical simulator")
    except ImportError as e:
        print(f"Error importing quantum electrochemical simulator: {e}")
        print("Make sure you're running this script from the project root directory")
        return 1
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Run the appropriate simulation
    try:
        if args.sim_type == 'diffusion':
            print("\nRunning diffusion simulation example...")
            
            # Create model with custom learning rate
            model = QuantumElectrochemicalSimulator(
                grid_dim=1,
                hidden_dim=64,
                num_species=2,  # Cation and anion
                hebbian_lr=args.learning_rate,
                decay_rate=args.decay_rate
            ).to(device)
            
            # Grid setup - 1D for simplicity
            grid_size = [100]
            
            # Diffusion coefficients for each species
            diffusion_coeffs = [0.1, 0.05]  # Cation diffuses faster than anion
            
            # No reactions in this simple example
            reaction_rates = [0.0]
            
            # Initial conditions - Gaussian concentration peaks at different positions
            def cation_initial(pos):
                return 1.0 * torch.exp(-((pos[0] - 0.25) / 0.05)**2)
            
            def anion_initial(pos):
                return 0.8 * torch.exp(-((pos[0] - 0.75) / 0.05)**2)
            
            initial_conditions = {
                0: cation_initial,  # Cation
                1: anion_initial    # Anion
            }
            
            print(f"Starting simulation with {args.steps} steps...")
            # Run simulation with our custom parameters
            results = model.run_diffusion_reaction_simulation(
                grid_size=grid_size,
                diffusion_coeffs=diffusion_coeffs,
                reaction_rates=reaction_rates,
                initial_conditions=initial_conditions,
                num_steps=args.steps,
                dt=0.01
            )
            
            if args.debug:
                print("\nSimulation Results:")
                print(f"  Phase Coherence: {len(results['phase_coherence'])} values")
                print(f"  Energy History: {len(results['energy_history'])} values")
                print(f"  Final State Shape: {results['final_state'].shape}")
            
            # Create standard visualization
            species_names = ["Cation", "Anion"]
            fig = plot_simulation_results(results, grid_size, species_names)
            
            # Save the simulation results
            if args.save_plot:
                # Main visualization
                plot_path = os.path.join(output_dir, f"quantum_electrochemical_diffusion_{timestamp}.png")
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {plot_path}")
                
                # Learning progress visualization (separate figure)
                fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
                
                # Plot phase coherence
                coherence = results['phase_coherence']
                ax1.plot(coherence)
                ax1.set_title('Phase Coherence During Training')
                ax1.set_xlabel('Step')
                ax1.set_ylabel('Coherence')
                ax1.grid(True)
                
                # Plot energy
                energy = results['energy_history']
                ax2.plot(energy)
                ax2.set_title('System Energy')
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Energy')
                ax2.grid(True)
                
                # Plot effective learning rate (coherence * learning rate)
                effective_lr = [c * args.learning_rate for c in coherence]
                ax3.plot(effective_lr)
                ax3.set_title(f'Effective Learning Rate (Coherence × {args.learning_rate})')
                ax3.set_xlabel('Step')
                ax3.set_ylabel('Effective Rate')
                ax3.grid(True)
                
                plt.tight_layout()
                lr_plot_path = os.path.join(output_dir, f"learning_rate_analysis_{timestamp}.png")
                fig2.savefig(lr_plot_path, dpi=300)
                print(f"Learning rate analysis saved to {lr_plot_path}")
            
            # Show the plots
            plt.show()
            
        # Add other simulation types here in the future
        else:
            print(f"Simulation type {args.sim_type} not implemented yet")
            return 1
            
        print("\nSimulation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

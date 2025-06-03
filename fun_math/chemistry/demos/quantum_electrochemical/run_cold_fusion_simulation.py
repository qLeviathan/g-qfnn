#!/usr/bin/env python3
"""
Run a quantum cold fusion simulation with timeline visualization

This script executes a simulation of cold fusion processes using quantum field modeling
and generates slide-based visualizations of the reaction dynamics
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Make sure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from quantum_electrochemical.quantum_cold_fusion_simulator import (
    QuantumColdFusionSimulator, device
)
from quantum_electrochemical.plot_cold_fusion_timeline import (
    plot_cold_fusion_timeline, plot_cold_fusion_3d
)

# Create dedicated output directory for cold fusion simulations
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"cold_fusion_{TIMESTAMP}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create subdirectories for different types of outputs
TIMELINE_DIR = os.path.join(OUTPUT_DIR, "timeline")
EVOLUTION_DIR = os.path.join(OUTPUT_DIR, "evolution")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")
os.makedirs(TIMELINE_DIR, exist_ok=True)
os.makedirs(EVOLUTION_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def run_cold_fusion_simulation():
    """Run the primary cold fusion simulation and generate visualizations"""
    
    print("Starting Quantum Cold Fusion Simulation...")
    print(f"Outputs will be saved to: {OUTPUT_DIR}")
    
    # Set up simulation parameters
    # Using 2D simulation for better visualization
    grid_size = [30, 30]  # 30x30 grid for detailed but reasonable simulation
    num_steps = 500       # Increased number of time steps for better evolution tracking
    
    # Create time evolution plots function
    def plot_time_evolution(results, config_name, save_dir):
        """Create detailed time series plots of the simulation evolution"""
        
        # Create figure with a grid of subplots
        fig, axs = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [1, 1, 1, 1.5]})
        
        # Extract time series data
        tunneling_history = np.array(results["tunneling_history"])
        screening_history = np.array(results["screening_history"])
        coherence_history = np.array(results["phase_coherence"])
        
        # Handle different array lengths by truncating to the shortest length
        min_length = min(len(tunneling_history), len(screening_history), len(coherence_history))
        tunneling_history = tunneling_history[:min_length]
        screening_history = screening_history[:min_length]
        coherence_history = coherence_history[:min_length]
        
        # If available, extract other time series data
        if "phonon_coupling_history" in results:
            phonon_history = np.array(results["phonon_coupling_history"])
            if len(phonon_history) > min_length:
                phonon_history = phonon_history[:min_length]
        else:
            phonon_history = np.zeros_like(tunneling_history)
        
        fusion_events = np.array(results["fusion_events"])
        if len(fusion_events) > min_length:
            fusion_events = fusion_events[:min_length]
        
        fusion_energy = np.array(results["fusion_energy"])
        if len(fusion_energy) > min_length:
            fusion_energy = fusion_energy[:min_length]
        
        # Time steps
        steps = np.arange(min_length)
        
        # Plot 1: Quantum parameters
        axs[0].plot(steps, tunneling_history, 'r-', label='Tunneling Probability', linewidth=2)
        axs[0].plot(steps, screening_history, 'g-', label='Electron Screening', linewidth=2)
        axs[0].set_title(f"Quantum Parameters Evolution - {config_name}", fontsize=14)
        axs[0].set_xlabel("Simulation Steps")
        axs[0].set_ylabel("Parameter Value")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Plot 2: Coherence and phonon coupling
        axs[1].plot(steps, coherence_history, 'b-', label='Quantum Coherence', linewidth=2)
        
        if len(phonon_history) > 0:
            axs[1].plot(steps, phonon_history, 'm-', label='Phonon Coupling', linewidth=2)
        
        axs[1].set_title("Coherence and Energy Transfer", fontsize=14)
        axs[1].set_xlabel("Simulation Steps")
        axs[1].set_ylabel("Parameter Value")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        # Plot 3: Fusion events and energy over time
        fusion_events_cumsum = np.cumsum(fusion_events)
        
        # Primary y-axis: Events
        axs[2].bar(steps, fusion_events, color='orange', alpha=0.6, label='Fusion Events')
        axs[2].set_xlabel("Simulation Steps")
        axs[2].set_ylabel("Events per Step")
        
        # Secondary y-axis: Cumulative events and energy
        ax2 = axs[2].twinx()
        ax2.plot(steps, fusion_events_cumsum, 'r-', label='Cumulative Events', linewidth=2)
        
        # Add fusion energy if available
        if "fusion_energy" in results and len(results["fusion_energy"]) > 0:
            # Make sure energy data is the right length
            energy_data = np.array(results["fusion_energy"])
            if len(energy_data) > len(steps):
                energy_data = energy_data[:len(steps)]
            elif len(energy_data) < len(steps):
                energy_data = np.pad(energy_data, (0, len(steps) - len(energy_data)), 'edge')
                
            ax2.plot(steps, energy_data, 'm--', label='Fusion Energy (MeV)', linewidth=2)
            
        ax2.set_ylabel("Cumulative Events / Energy", color='r')
        
        axs[2].set_title("Fusion Events & Energy", fontsize=14)
        
        # Combine legends
        lines1, labels1 = axs[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        axs[2].grid(True, alpha=0.3)
        
        # Plot 4: Energy control analysis OR correlation analysis
        # Check if we have energy control data
        if hasattr(model, "energy_control_efficiency") and len(model.energy_control_efficiency) > 0:
            # We have energy control data - plot it
            control_steps = np.arange(len(model.energy_control_efficiency))
            
            # Primary y-axis: Control efficiency
            axs[3].plot(control_steps, model.energy_control_efficiency, 'g-', 
                       label='Control Efficiency', linewidth=2)
            axs[3].set_ylabel("Efficiency", color='g')
            axs[3].tick_params(axis='y', labelcolor='g')
            axs[3].set_ylim(0, 1.1)
            
            # Secondary y-axis: Damping factors
            ax3_2 = axs[3].twinx()
            ax3_2.plot(control_steps, model.energy_damping_factors, 'b-', 
                     label='Damping Factor', linewidth=2)
            ax3_2.set_ylabel("Damping Factor", color='b')
            ax3_2.tick_params(axis='y', labelcolor='b')
            
            # Set titles and labels
            axs[3].set_title("Energy Control Metrics", fontsize=14)
            axs[3].set_xlabel("Control Events")
            
            # Combine legends
            lines1, labels1 = axs[3].get_legend_handles_labels()
            lines2, labels2 = ax3_2.get_legend_handles_labels()
            axs[3].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
        else:
            # No energy control data - show parameter correlation
            axs[3].scatter(tunneling_history, screening_history, 
                         c=coherence_history, cmap='viridis', 
                         s=50, alpha=0.7, edgecolor='k', linewidth=0.5)
            axs[3].set_title("Parameter Correlation Analysis", fontsize=14)
            axs[3].set_xlabel("Tunneling Probability")
            axs[3].set_ylabel("Electron Screening")
            axs[3].grid(True, alpha=0.3)
            
            # Add colorbar for coherence
            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(min(coherence_history), max(coherence_history)), 
                                                   cmap='viridis'), ax=axs[3])
            cbar.set_label('Quantum Coherence')
        
        # Tight layout and save
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"time_evolution_{config_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    # Create our simulator model
    model = QuantumColdFusionSimulator(
        grid_dim=2,               # 2D simulation
        hidden_dim=64,            # Neural net hidden dimension
        num_species=3,            # Deuterium, electrons, palladium
        temperature=293.0,        # Room temperature in K
        hebbian_lr=0.01           # Learning rate for dynamics
    ).to(device)
    
    # Define a function to create coherence sweep simulation
    def run_coherence_sweep(base_config, coherence_values):
        """Run simulations with varying coherence parameters"""
        sweep_results = []
        
        for coherence_value in coherence_values:
            # Clone the base config
            sweep_config = base_config.copy()
            
            # Update with coherence value modifier
            sweep_config["name"] = f"{base_config['name']}_coherence_{coherence_value:.2f}"
            sweep_config["coherence_scale"] = coherence_value
            
            # Adjust electron density which impacts coherence
            sweep_config["electron_density_scale"] = coherence_value
            
            print(f"\nRunning Coherence Sweep - Value: {coherence_value:.2f}...")
            
            # Run the simulation
            results = model.run_cold_fusion_simulation(
                grid_size=grid_size,
                deuterium_loading_ratio=sweep_config["deuterium_loading_ratio"],
                electron_density_scale=sweep_config["electron_density_scale"],
                palladium_structure=sweep_config["palladium_structure"],
                temperature=sweep_config["temperature"],
                electrolysis_current=sweep_config["electrolysis_current"],
                num_steps=num_steps,
                dt=0.005  # Smaller time step for more precision
            )
            
            # Extract key metrics
            fusion_events = np.sum(results["fusion_events"])
            coherence_avg = np.mean(results["phase_coherence"])
            tunneling_avg = np.mean(results["tunneling_history"])
            
            # Store results
            sweep_results.append({
                "coherence_value": coherence_value,
                "fusion_events": fusion_events,
                "coherence_avg": coherence_avg,
                "tunneling_avg": tunneling_avg,
                "config": sweep_config,
                "results": results
            })
            
            # Create evolution plot if fusion events occurred
            if fusion_events > 0:
                plot_time_evolution(results, sweep_config["name"], EVOLUTION_DIR)
        
        return sweep_results
    
    # Run simulation with different parameters to explore resonance conditions
    simulation_configs = [
        {
            "name": "standard",
            "deuterium_loading_ratio": 0.8,    # Standard loading
            "electron_density_scale": 1.0,     # Normal electron density
            "palladium_structure": "uniform",  # Uniform palladium
            "temperature": 293.0,              # Room temperature (K)
            "electrolysis_current": 0.0,       # No external current
        },
        {
            "name": "high_loading",
            "deuterium_loading_ratio": 1.5,    # Even higher loading (increased from 1.2)
            "electron_density_scale": 1.1,     # Slightly higher electron density
            "palladium_structure": "lattice",  # Lattice structure
            "temperature": 293.0,
            "electrolysis_current": 0.0,
        },
        {
            "name": "resonant_phonon",
            "deuterium_loading_ratio": 1.2,    # High loading
            "electron_density_scale": 1.3,     # Higher electron density
            "palladium_structure": "lattice",  # Lattice structure
            "temperature": 320.0,              # Slightly elevated temperature
            "electrolysis_current": 0.7,       # Higher applied current
        },
        {
            "name": "energy_control_demo",
            "deuterium_loading_ratio": 2.0,    # Extremely high loading for fusion
            "electron_density_scale": 2.1,     # High electron screening 
            "palladium_structure": "lattice",  # Lattice structure
            "temperature": 350.0,              # Elevated temperature
            "electrolysis_current": 1.5,       # High current
        }
    ]
    
    # First run each base configuration
    base_results = []
    for i, config in enumerate(simulation_configs):
        print(f"\n\nRunning Base Configuration {i+1}/{len(simulation_configs)}: {config['name']}...")
        print(f"Parameters: {config}")
        
        # Start timing
        start_time = time.time()
        
        # Run the simulation
        results = model.run_cold_fusion_simulation(
            grid_size=grid_size,
            deuterium_loading_ratio=config["deuterium_loading_ratio"],
            electron_density_scale=config["electron_density_scale"],
            palladium_structure=config["palladium_structure"],
            temperature=config["temperature"],
            electrolysis_current=config["electrolysis_current"],
            num_steps=num_steps,
            dt=0.01
        )
        
        base_results.append({
            "config": config,
            "results": results
        })
        
        # Calculate runtime
        runtime = time.time() - start_time
        print(f"Simulation completed in {runtime:.2f} seconds")
        
        # Analyze results
        fusion_events = np.sum(results["fusion_events"])
        final_energy = results["fusion_energy"][-1] if results["fusion_energy"] else 0
        
        print(f"Total Fusion Events: {fusion_events}")
        print(f"Total Fusion Energy: {final_energy:.2f} MeV")
        
        # Create visualization filenames
        timeline_filename = f"timeline_{config['name']}.png"
        timeline_path = os.path.join(TIMELINE_DIR, timeline_filename)
        
        # Create and save timeline visualization
        print("Generating timeline visualization...")
        fig_timeline = plot_cold_fusion_timeline(results, num_frames=6, save_path=timeline_path)
        
        # Create 3D visualization
        view3d_filename = f"3d_view_{config['name']}.png"
        view3d_path = os.path.join(TIMELINE_DIR, view3d_filename)
        
        print("Generating 3D visualization...")
        fig_3d = plot_cold_fusion_3d(results, frame_idx=-1, save_path=view3d_path)
        
        # Create time evolution visualization
        evolution_path = plot_time_evolution(results, config["name"], EVOLUTION_DIR)
        print(f"Time evolution visualization saved to: {evolution_path}")
        
        # Close figures to free memory
        plt.close(fig_timeline)
        plt.close(fig_3d)
        
        # If no fusion events detected, run coherence sweep for this configuration
        if fusion_events == 0:
            print(f"\nNo fusion events detected for {config['name']}. Running coherence sweep...")
            
            # Define coherence values to sweep
            coherence_values = np.linspace(1.0, 2.5, 6)  # Test 6 coherence values from 1.0 to 2.5
            
            # Run coherence sweep
            sweep_results = run_coherence_sweep(config, coherence_values)
            
            # Analyze sweep results
            fusion_by_coherence = [r["fusion_events"] for r in sweep_results]
            best_idx = np.argmax(fusion_by_coherence)
            
            if fusion_by_coherence[best_idx] > 0:
                print(f"Optimal coherence value found: {sweep_results[best_idx]['coherence_value']:.2f}")
                print(f"Fusion events at optimal value: {fusion_by_coherence[best_idx]}")
                
                # Get the best results for detailed visualization
                best_result = sweep_results[best_idx]["results"]
                best_config_name = sweep_results[best_idx]["config"]["name"]
                
                # Create detailed visualizations for the best result
                best_timeline_path = os.path.join(TIMELINE_DIR, f"timeline_{best_config_name}.png")
                plot_cold_fusion_timeline(best_result, num_frames=8, save_path=best_timeline_path)
            else:
                print("No fusion events detected in coherence sweep.")
            
            # Create coherence sweep analysis plot
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # Plot fusion events
            ax1.plot([r["coherence_value"] for r in sweep_results], 
                    [r["fusion_events"] for r in sweep_results], 
                    'ro-', linewidth=2, markersize=8, label='Fusion Events')
            ax1.set_xlabel('Coherence Scale Factor')
            ax1.set_ylabel('Fusion Events', color='r')
            ax1.tick_params(axis='y', labelcolor='r')
            
            # Add second y-axis for quantum parameters
            ax2 = ax1.twinx()
            ax2.plot([r["coherence_value"] for r in sweep_results], 
                    [r["coherence_avg"] for r in sweep_results], 
                    'b^-', linewidth=2, markersize=8, label='Avg. Coherence')
            ax2.plot([r["coherence_value"] for r in sweep_results], 
                    [r["tunneling_avg"] for r in sweep_results], 
                    'gD-', linewidth=2, markersize=8, label='Avg. Tunneling')
            ax2.set_ylabel('Quantum Parameter Values', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Add title and grid
            plt.title(f'Coherence Sweep Analysis for {config["name"]}', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            # Save plot
            sweep_plot_path = os.path.join(ANALYSIS_DIR, f"coherence_sweep_{config['name']}.png")
            plt.tight_layout()
            plt.savefig(sweep_plot_path, dpi=300)
            plt.close(fig)
            
            print(f"Coherence sweep analysis saved to: {sweep_plot_path}")
    
    # Create comparative analysis plot for all base configurations
    fig, axs = plt.subplots(3, 1, figsize=(14, 18))
    
    # Colors for each configuration
    colors = ['blue', 'green', 'red', 'purple']
    
    # Plot 1: Tunneling probability comparison
    for i, res in enumerate(base_results):
        config_name = res["config"]["name"]
        tunneling = res["results"]["tunneling_history"]
        axs[0].plot(np.arange(len(tunneling)), tunneling, 
                  color=colors[i % len(colors)], linewidth=2, label=config_name)
    
    axs[0].set_title("Tunneling Probability Comparison", fontsize=14)
    axs[0].set_xlabel("Simulation Steps")
    axs[0].set_ylabel("Tunneling Probability")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Coherence comparison
    for i, res in enumerate(base_results):
        config_name = res["config"]["name"]
        coherence = res["results"]["phase_coherence"]
        axs[1].plot(np.arange(len(coherence)), coherence, 
                  color=colors[i % len(colors)], linewidth=2, label=config_name)
    
    axs[1].set_title("Quantum Coherence Comparison", fontsize=14)
    axs[1].set_xlabel("Simulation Steps")
    axs[1].set_ylabel("Coherence")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Screening comparison
    for i, res in enumerate(base_results):
        config_name = res["config"]["name"]
        screening = res["results"]["screening_history"]
        axs[2].plot(np.arange(len(screening)), screening, 
                  color=colors[i % len(colors)], linewidth=2, label=config_name)
    
    axs[2].set_title("Electron Screening Comparison", fontsize=14)
    axs[2].set_xlabel("Simulation Steps")
    axs[2].set_ylabel("Screening Factor")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    # Save comparative analysis
    comp_analysis_path = os.path.join(ANALYSIS_DIR, "configurations_comparison.png")
    plt.tight_layout()
    plt.savefig(comp_analysis_path, dpi=300)
    plt.close(fig)
    
    print("\nAll simulations and analyses complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"- Timeline visualizations: {TIMELINE_DIR}")
    print(f"- Time evolution graphics: {EVOLUTION_DIR}")
    print(f"- Comparative analyses: {ANALYSIS_DIR}")


if __name__ == "__main__":
    run_cold_fusion_simulation()

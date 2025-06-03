#!/usr/bin/env python3
"""
Generate journal-grade visualizations for Gwave Quantum Field Dynamics.

This script runs the tachyonic experiment and generates high-quality
visualizations showing wave mechanics, tachyonic trajectories, field dynamics,
Lévy flights and loss landscapes.
"""

import os
import numpy as np
from gwave_core_fixed import GwaveCore, GwaveConfig, PHI, place_tokens_at_critical
from gwave_advanced_viz import GwaveAdvancedViz

# Output directory
OUTPUT_DIR = "outputs/gwave/physics/journal_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_tachyonic_experiment():
    """
    Run enhanced tachyonic experiment optimized for striking visuals.
    Places tokens in patterns that create interesting dynamics.
    """
    print("Running enhanced tachyonic experiment for journal visualizations...")
    
    # Create config optimized for visuals
    cfg = GwaveConfig(
        max_tokens=128,
        levy_alpha=PHI,
        track_phi_n=8,
        track_tachyonic=True,
        # Reduced light speed to make tachyonic events more common
        c=0.7,
        # Higher omega_z for more interesting helical motion
        omega_z=2 * np.pi / PHI,
        # Higher resonance temperature for more dynamic interactions
        resonance_temp=0.2,
        # Stability parameters
        mass_min=0.01,
        energy_clip=1e8
    )
    
    # Initialize model
    gw = GwaveCore(cfg)
    
    # Calculate critical radius
    r_critical = cfg.c * PHI**2 / np.pi
    ell_critical = np.log(r_critical)
    
    # Place tokens in a more interesting pattern with multiple layers
    
    # 1. Tokens at critical radius in a circular pattern
    n_critical = 16
    for i in range(n_critical):
        theta = 2*np.pi * i / n_critical
        z = 2*np.pi * i / n_critical
        gw.add_token(ell_critical, theta, z, 1.0)
    
    # 2. Tokens at phi^n radii to demonstrate stratification
    for n in range(1, 5):  # Phi^1 through Phi^4
        phi_n = PHI**n
        ell_phi_n = np.log(phi_n)
        
        n_tokens = 8
        for i in range(n_tokens):
            theta = 2*np.pi * (i + 0.5) / n_tokens
            z = 2*np.pi * (i + 0.25*n) / n_tokens
            
            # Add slight variation for more dynamic behavior
            variation = 0.03 * (np.random.random() - 0.5)
            gw.add_token(ell_phi_n + variation, theta, z, 0.8)
    
    # 3. Tokens in a spiral pattern to create vortex fields
    n_spiral = 12
    for i in range(n_spiral):
        # Log-spiral pattern
        t = i / n_spiral * 3 * np.pi
        r = 0.5 * np.exp(t / (PHI * 8))
        theta = t
        
        # Convert radius to log-radius
        ell = np.log(r)
        
        # z coordinate
        z = 2 * np.pi * i / n_spiral
        
        # Add token
        gw.add_token(ell, theta, z, 1.2)
    
    # 4. Some outer tokens to show boundary effects
    n_outer = 8
    for i in range(n_outer):
        theta = 2*np.pi * i / n_outer
        z = 2*np.pi * (i + 0.5) / n_outer
        gw.add_token(4.0 + 0.1*np.random.random(), theta, z, 0.7)
    
    print(f"Placed {gw.N_act} tokens in optimized patterns")
    
    # Evolve system with more steps for richer dynamics
    steps = 300
    print(f"Evolving system for {steps} steps...")
    gw.evolve(steps)
    
    # Statistics
    crystallized = np.sum(gw.froz[:gw.N_act])
    tachyonic_events = len(gw.tachyonic_events)
    
    print("\nResults:")
    print(f"- Active tokens: {gw.N_act}")
    print(f"- Crystallized tokens: {crystallized}")
    print(f"- Tachyonic events: {tachyonic_events}")
    
    return gw

def run_levy_flight_experiment():
    """
    Run enhanced experiment optimized for Lévy flights and inversions.
    """
    print("Running enhanced Lévy flight experiment for journal visualizations...")
    
    # Create config optimized for Lévy flights
    cfg = GwaveConfig(
        max_tokens=64,
        # Higher levy_alpha for more extreme Lévy flights
        levy_alpha=PHI*1.2,
        track_phi_n=8,
        track_tachyonic=True,
        # Other parameters optimized for interesting dynamics
        resonance_temp=0.3,
        k_bound=0.3,
        # Stability parameters
        mass_min=0.01,
        energy_clip=1e8
    )
    
    # Initialize model
    gw = GwaveCore(cfg)
    
    # Place tokens in concentric rings to encourage tunneling
    rings = [1/PHI, 1.0, PHI, PHI**2]
    
    for i, radius in enumerate(rings):
        ell = np.log(radius)
        n_tokens = 12
        
        for j in range(n_tokens):
            theta = 2*np.pi * j / n_tokens
            z = 2*np.pi * (j + 0.5*i) / n_tokens
            
            # Add slight variation
            variation = 0.05 * (np.random.random() - 0.5)
            gw.add_token(ell + variation, theta, z, 1.0)
    
    print(f"Placed {gw.N_act} tokens in concentric rings")
    
    # Evolve system with more steps for richer dynamics
    steps = 400
    print(f"Evolving system for {steps} steps...")
    gw.evolve(steps)
    
    # Statistics
    crystallized = np.sum(gw.froz[:gw.N_act])
    tachyonic_events = len(gw.tachyonic_events)
    
    print("\nResults:")
    print(f"- Active tokens: {gw.N_act}")
    print(f"- Crystallized tokens: {crystallized}")
    print(f"- Tachyonic events: {tachyonic_events}")
    
    return gw

def generate_journal_visualizations():
    """
    Generate the full set of journal-grade visualizations.
    """
    # Run optimized experiments
    gw_tachyonic = run_tachyonic_experiment()
    
    # Save experiment state
    dump_path = os.path.join(OUTPUT_DIR, "tachyonic_experiment.npz")
    gw_tachyonic.dump(dump_path)
    print(f"Saved experiment state to {dump_path}")
    
    # Generate advanced visualizations
    viz = GwaveAdvancedViz(gw_tachyonic)
    
    # Change output directory
    viz.output_dir = OUTPUT_DIR
    
    # Generate all visualizations
    viz.visualize_all()
    
    # Run Lévy flight experiment
    gw_levy = run_levy_flight_experiment()
    
    # Save experiment state
    dump_path = os.path.join(OUTPUT_DIR, "levy_flight_experiment.npz")
    gw_levy.dump(dump_path)
    print(f"Saved experiment state to {dump_path}")
    
    # Generate advanced visualizations
    viz_levy = GwaveAdvancedViz(gw_levy)
    
    # Change output directory
    viz_levy.output_dir = OUTPUT_DIR
    
    # Generate visualizations with different filenames
    viz_levy.visualize_wave_mechanics("wave_mechanics_levy.png")
    viz_levy.visualize_loss_landscape("loss_landscape_levy.png")
    
    print("\nAll journal-grade visualizations generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_journal_visualizations()
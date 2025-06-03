#!/usr/bin/env python3
"""
Generate dual vortex and phase-locked evolution visualizations.

This script demonstrates:
1. Dual counter-rotating vortices at the golden ratio positions
2. Phase-locked evolution showing how the loss field evolves until sequence completion
"""

import os
import numpy as np
from gwave_core_fixed import GwaveCore, GwaveConfig, PHI
from gwave_advanced_viz import GwaveAdvancedViz

# Output directory
OUTPUT_DIR = "outputs/gwave/physics/dual_vortex_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_dual_vortex_experiment():
    """
    Run experiment optimized for showing dual vortices.
    """
    print("Running dual vortex experiment...")
    
    # Create config optimized for dual vortices
    cfg = GwaveConfig(
        max_tokens=64,
        levy_alpha=PHI,
        track_phi_n=8,
        track_tachyonic=True,
        # Higher resonance temperature for more dynamic interactions
        resonance_temp=0.3,
        # Stability parameters
        mass_min=0.01,
        energy_clip=1e8
    )
    
    # Initialize model
    gw = GwaveCore(cfg)
    
    # Place tokens in specific patterns to highlight dual vortices
    
    # 1. Tokens around first vortex center at (φ-1, 0)
    vortex1_center_r = PHI - 1
    vortex1_center_ell = np.log(vortex1_center_r)
    
    n_vortex1 = 12
    for i in range(n_vortex1):
        # Arrange in a circle around center
        theta = 2*np.pi * i / n_vortex1
        r_offset = 0.2 * np.random.random()
        ell = vortex1_center_ell + r_offset
        z = 2*np.pi * i / n_vortex1
        
        # Add token
        gw.add_token(ell, theta, z, 1.0)
    
    # 2. Tokens around second vortex center at (-1/φ, 0)
    vortex2_center_r = 1/PHI
    vortex2_center_ell = np.log(vortex2_center_r)
    
    n_vortex2 = 12
    for i in range(n_vortex2):
        # Arrange in a circle around center
        theta = 2*np.pi * i / n_vortex2
        r_offset = 0.15 * np.random.random()
        ell = vortex2_center_ell + r_offset
        z = 2*np.pi * (i + 0.5) / n_vortex2  # Offset for variety
        
        # Add token
        gw.add_token(ell, theta, z, 1.0)
    
    # 3. Tokens in between vortices to show interaction
    n_between = 8
    for i in range(n_between):
        # Interpolate between vortex centers
        alpha = (i + 0.5) / n_between
        ell = vortex1_center_ell * alpha + vortex2_center_ell * (1 - alpha)
        theta = np.pi * i / n_between  # Vary angle
        z = np.pi * (1 - i / n_between)
        
        # Add token
        gw.add_token(ell, theta, z, 1.0)
    
    # 4. Some tokens at other phi^n radii
    for n in range(2, 5):  # Phi^2 through Phi^4
        phi_n = PHI**n
        ell_phi_n = np.log(phi_n)
        
        n_tokens = 8
        for i in range(n_tokens):
            theta = 2*np.pi * (i + 0.5) / n_tokens
            z = 2*np.pi * (i + 0.25*n) / n_tokens
            
            # Add slight variation
            variation = 0.05 * (np.random.random() - 0.5)
            gw.add_token(ell_phi_n + variation, theta, z, 0.8)
    
    print(f"Placed {gw.N_act} tokens to highlight dual vortices")
    
    # Evolve system
    steps = 200
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

def generate_visualizations():
    """
    Generate dual vortex and phase-locked evolution visualizations.
    """
    # Run experiment
    gw = run_dual_vortex_experiment()
    
    # Save experiment state
    dump_path = os.path.join(OUTPUT_DIR, "dual_vortex_experiment.npz")
    gw.dump(dump_path)
    print(f"Saved experiment state to {dump_path}")
    
    # Create visualizer
    viz = GwaveAdvancedViz(gw)
    
    # Set output directory
    viz.output_dir = OUTPUT_DIR
    
    # Generate dual vortex visualization
    print("Generating dual vortex visualization...")
    viz.visualize_dual_vortices()
    
    # Generate phase-locked evolution
    print("Generating phase-locked evolution visualizations...")
    viz.visualize_phase_locked_evolution(steps=5)
    
    print("\nAll visualizations generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_visualizations()
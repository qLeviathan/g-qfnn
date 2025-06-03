#!/usr/bin/env python3
"""
Gwave Quantum Field Experiments

This script runs multiple experiments with the Gwave framework to demonstrate:
1. Tachyonic helical trajectories
2. Phi^n layer stratification
3. Vortex field dynamics

Usage:
  python gwave_experiments.py [--experiment EXPERIMENT_NAME]
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil  # For copying files
import tempfile  # For temp directory
from gwave_core_fixed import GwaveCore, GwaveConfig, PHI, place_tokens_at_critical, text_to_tokens, safe_log, OUTPUT_DIR as CORE_OUTPUT_DIR

# Experiment output directory
OUTPUT_DIR = "outputs/gwave/physics/experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def experiment_tachyonic_helical(steps=300):
    """
    Experiment to demonstrate tachyonic helical trajectories.
    Places tokens at and near the critical radius for tachyonic events.
    """
    print("Running Tachyonic Helical Trajectory Experiment")
    
    # Create config optimized for tachyonic events
    cfg = GwaveConfig(
        max_tokens=64,
        levy_alpha=PHI,
        track_phi_n=8,
        track_tachyonic=True,
        # Reduced light speed to make tachyonic events more common
        c=0.8,
        # Higher omega_z for more interesting helical motion
        omega_z=2 * np.pi / PHI,
        # Stability parameters
        mass_min=0.01,
        energy_clip=1e8
    )
    
    # Initialize model
    gw = GwaveCore(cfg)
    
    # Calculate critical radius
    r_critical = cfg.c * PHI**2 / np.pi
    ell_critical = safe_log(r_critical)
    
    # Place tokens in a pattern designed to produce tachyonic events
    # Some exactly at critical radius
    n_exact = 8
    for i in range(n_exact):
        theta = 2*np.pi * i / n_exact
        z = 2*np.pi * i / n_exact
        gw.add_token(ell_critical, theta, z, 1.0)
    
    # Some just inside critical radius
    n_inside = 8
    for i in range(n_inside):
        theta = 2*np.pi * (i + 0.5) / n_inside
        z = 2*np.pi * (i + 0.25) / n_inside
        gw.add_token(ell_critical * 0.9, theta, z, 1.0)
    
    # Some just outside critical radius
    n_outside = 8
    for i in range(n_outside):
        theta = 2*np.pi * (i + 0.25) / n_outside
        z = 2*np.pi * (i + 0.5) / n_outside
        gw.add_token(ell_critical * 1.1, theta, z, 1.0)
        
    print(f"Placed {n_exact + n_inside + n_outside} tokens at and near critical radius ell = {ell_critical}")
    
    # Evolve system
    print(f"Evolving system for {steps} steps...")
    gw.evolve(steps)
    
    # Statistics
    crystallized = np.sum(gw.froz[:gw.N_act])
    tachyonic_events = len(gw.tachyonic_events)
    
    print("\nResults:")
    print(f"- Active tokens: {gw.N_act}")
    print(f"- Crystallized tokens: {crystallized}")
    print(f"- Tachyonic events: {tachyonic_events}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Custom visualization of tachyonic helical trajectories
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique tokens that experienced tachyonic events
    tachyonic_tokens = set(event['token'] for event in gw.tachyonic_events)
    
    # Track all trajectories but highlight tachyonic ones
    for i in range(gw.N_act):
        # Extract trajectory
        traj = np.array([pos[i] for pos in gw.traj])
        
        # Filter out invalid positions
        valid_mask = np.all(np.isfinite(traj), axis=1)
        if not np.any(valid_mask):
            continue
            
        traj = traj[valid_mask]
        
        # Convert from log-cylindrical to Cartesian
        ell = traj[:, 0]
        theta = traj[:, 1]
        z = traj[:, 2]
        
        r = np.exp(ell)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Determine color and style based on whether token is tachyonic
        if i in tachyonic_tokens:
            color = 'red'
            linewidth = 2
            label = f'Tachyonic token {i}'
            
            # Find tachyonic events for this token
            token_events = [event for event in gw.tachyonic_events if event['token'] == i]
            
            # Mark tachyonic events
            for event in token_events:
                # Find closest trajectory point
                t_idx = int(event['time'] / (PHI ** -2))
                if t_idx < len(x) and np.isfinite(x[t_idx]) and np.isfinite(y[t_idx]) and np.isfinite(z[t_idx]):
                    # Mark with sphere
                    ax.scatter([x[t_idx]], [y[t_idx]], [z[t_idx]], 
                              color='purple', s=100, alpha=0.7)
                    
                    # Add velocity vector
                    v_phase = event['velocity']
                    scale = 0.2 * v_phase / gw.cfg.c
                    dx = -scale * np.sin(theta[t_idx])
                    dy = scale * np.cos(theta[t_idx])
                    dz = 0  # Assuming z-direction is not affected
                    
                    # Check for valid vector
                    if np.isfinite(dx) and np.isfinite(dy) and np.isfinite(dz):
                        ax.quiver(x[t_idx], y[t_idx], z[t_idx], 
                                 dx, dy, dz, 
                                 color='purple', alpha=0.7, 
                                 arrow_length_ratio=0.3)
            
            # Plot tachyonic trajectory
            ax.plot(x, y, z, color=color, linewidth=linewidth, label=label)
            
            # Mark start and end
            if len(x) > 0 and np.isfinite(x[0]) and np.isfinite(y[0]) and np.isfinite(z[0]):
                ax.scatter(x[0], y[0], z[0], color='green', s=50)
            if len(x) > 0 and np.isfinite(x[-1]) and np.isfinite(y[-1]) and np.isfinite(z[-1]):
                ax.scatter(x[-1], y[-1], z[-1], color='orange', s=50)
    
    # Plot cylinder at critical radius
    theta_circle = np.linspace(0, 2*np.pi, 100)
    z_levels = np.linspace(0, 2*np.pi, 10)
    
    for z_level in z_levels:
        # Critical radius for tachyonic behavior
        r_critical = gw.cfg.c * PHI**2 / np.pi
        x_crit = r_critical * np.cos(theta_circle)
        y_crit = r_critical * np.sin(theta_circle)
        ax.plot(x_crit, y_crit, [z_level]*len(theta_circle), 
               'red', alpha=0.3, linewidth=1, linestyle='--')
    
    # Add reference helix
    t_ref = np.linspace(0, 4*np.pi, 200)
    r_ref = r_critical
    x_ref = r_ref * np.cos(PHI * t_ref)
    y_ref = r_ref * np.sin(PHI * t_ref)
    z_ref = t_ref
    ax.plot(x_ref, y_ref, z_ref, 'k--', alpha=0.5, linewidth=1,
           label='Reference helix (φ frequency)')
    
    # Add labels
    ax.set_xlabel('X = r·cos(θ)')
    ax.set_ylabel('Y = r·sin(θ)')
    ax.set_zlabel('Z (rotor phase)')
    ax.set_title('Tachyonic Helical Trajectories')
    
    # Add legend for unique entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    # Save figure
    filename = "tachyonic_helical_trajectories.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    
    # Generate standard visualizations to default location then copy
    gw.visualize_tokens()
    gw.visualize_vortex_field()
    gw.visualize_energy_history()
    
    # Copy generated files to experiment directory with new names
    shutil.copy(os.path.join(CORE_OUTPUT_DIR, "tokens.png"), 
               os.path.join(OUTPUT_DIR, "tachyonic_tokens.png"))
    shutil.copy(os.path.join(CORE_OUTPUT_DIR, "vortex_field.png"), 
               os.path.join(OUTPUT_DIR, "tachyonic_vortex_field.png"))
    shutil.copy(os.path.join(CORE_OUTPUT_DIR, "energy_history.png"), 
               os.path.join(OUTPUT_DIR, "tachyonic_energy_history.png"))
    
    print(f"Visualizations saved to {OUTPUT_DIR}/")
    
    return gw

def experiment_phi_n_layers(steps=200):
    """
    Experiment to demonstrate phi^n layer stratification.
    Places tokens at phi^n radii to observe natural stratification.
    """
    print("Running Phi^n Layer Stratification Experiment")
    
    # Create config optimized for phi^n tracking
    cfg = GwaveConfig(
        max_tokens=64,
        levy_alpha=PHI,
        track_phi_n=8,
        track_tachyonic=True,
        # Higher sigma_theta to increase Hebbian coupling at phi^n radii
        sigma_theta=np.pi / (PHI * 0.5),
        # Stability parameters
        mass_min=0.01,
        energy_clip=1e8
    )
    
    # Initialize model
    gw = GwaveCore(cfg)
    
    # Place tokens at phi^n radii
    phi_n_layers = [PHI**i for i in range(1, cfg.track_phi_n + 1)]
    
    # Number of tokens per layer
    tokens_per_layer = 8
    
    for i, phi_n in enumerate(phi_n_layers):
        for j in range(tokens_per_layer):
            # Convert radius to log-radius
            ell = np.log(phi_n)
            
            # Calculate angle - distribute evenly with offset per layer
            theta = 2*np.pi * (j + 0.1*i) / tokens_per_layer
            
            # z coordinate - distribute in range [0, 2π)
            z = 2*np.pi * j / tokens_per_layer
            
            # Add token with slightly varying radius to encourage interaction
            variation = 0.05 * (np.random.random() - 0.5)
            gw.add_token(ell + variation, theta, z, 1.0)
    
    print(f"Placed {tokens_per_layer * len(phi_n_layers)} tokens at phi^n radii")
    
    # Evolve system
    print(f"Evolving system for {steps} steps...")
    gw.evolve(steps)
    
    # Statistics
    crystallized = np.sum(gw.froz[:gw.N_act])
    tachyonic_events = len(gw.tachyonic_events)
    
    print("\nResults:")
    print(f"- Active tokens: {gw.N_act}")
    print(f"- Crystallized tokens: {crystallized}")
    print(f"- Tachyonic events: {tachyonic_events}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Custom visualization of phi^n layers
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Convert to Cartesian
    r = np.exp(gw.pos[:gw.N_act, 0])
    theta = gw.pos[:gw.N_act, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Filter out invalid positions
    valid_mask = np.isfinite(x) & np.isfinite(y)
    
    # Plot tokens with color based on phi^n layer
    cmap = plt.cm.viridis
    colors = []
    sizes = []
    
    for i in range(gw.N_act):
        if not valid_mask[i]:
            continue
            
        # Find closest phi^n layer
        token_r = r[i]
        distances = [abs(token_r - phi_n) for phi_n in phi_n_layers]
        closest_layer = np.argmin(distances)
        min_distance = distances[closest_layer]
        
        # Normalize distance for color
        norm_distance = min_distance / phi_n_layers[closest_layer]
        
        # Color based on distance to nearest phi^n layer
        if norm_distance < 0.05:  # Very close to a phi^n layer
            color = cmap(closest_layer / (len(phi_n_layers)-1))
            size = 100
        else:
            color = 'gray'
            size = 30
            
        colors.append(color)
        sizes.append(size)
    
    # Plot tokens
    scatter = ax.scatter(
        x[valid_mask], 
        y[valid_mask], 
        c=colors, 
        s=sizes,
        alpha=0.7
    )
    
    # Add phi^n layers
    theta_circle = np.linspace(0, 2*np.pi, 100)
    
    for i, phi_n in enumerate(phi_n_layers):
        x_phi_n = phi_n * np.cos(theta_circle)
        y_phi_n = phi_n * np.sin(theta_circle)
        ax.plot(x_phi_n, y_phi_n, color=cmap(i/(len(phi_n_layers)-1)), 
               alpha=0.7, linewidth=2, label=f'φ^{i+1} = {phi_n:.3f}')
    
    # Add vortex field indicators
    vorticity = gw.vortex_field[:gw.N_act, 2]
    # Normalize for better visualization
    if np.any(vorticity != 0):
        max_vorticity = np.max(np.abs(vorticity[np.isfinite(vorticity)]))
        if max_vorticity > 0:
            vorticity_norm = np.clip(vorticity / max_vorticity, -1.0, 1.0)
            
            # Draw arrows proportional to vorticity
            for i in range(gw.N_act):
                if gw.froz[i] or not valid_mask[i] or not np.isfinite(vorticity[i]):
                    continue
                    
                # Create tangential vector to show rotation
                v_scale = vorticity_norm[i] * 0.2 * r[i]
                dx = -v_scale * np.sin(theta[i])
                dy = v_scale * np.cos(theta[i])
                
                # Check for valid vector
                if np.isfinite(dx) and np.isfinite(dy):
                    ax.arrow(x[i], y[i], dx, dy, 
                           head_width=0.05, head_length=0.08, 
                           fc='green', ec='green', alpha=0.6)
    
    # Labels and title
    ax.set_xlabel('X = r·cos(θ)')
    ax.set_ylabel('Y = r·sin(θ)')
    ax.set_title('Phi^n Layer Stratification')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save figure
    filename = "phi_n_stratification.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    
    # Generate standard visualizations to default location then copy
    gw.visualize_phi_n_tracking()
    gw.visualize_vortex_field()
    
    # Copy generated files to experiment directory with new names
    shutil.copy(os.path.join(CORE_OUTPUT_DIR, "phi_n_tracking.png"), 
               os.path.join(OUTPUT_DIR, "phi_n_tracking_detailed.png"))
    shutil.copy(os.path.join(CORE_OUTPUT_DIR, "vortex_field.png"), 
               os.path.join(OUTPUT_DIR, "phi_n_vortex_field.png"))
    
    print(f"Visualizations saved to {OUTPUT_DIR}/")
    
    return gw

def experiment_vortex_field(steps=200):
    """
    Experiment to demonstrate vortex field dynamics.
    Places tokens in a pattern designed to create strong vortex fields.
    """
    print("Running Vortex Field Dynamics Experiment")
    
    # Create config optimized for vortex field visualization
    cfg = GwaveConfig(
        max_tokens=64,
        levy_alpha=PHI,
        track_phi_n=8,
        track_tachyonic=True,
        # Higher omega_z for more interesting vortex dynamics
        omega_z=2 * np.pi / (PHI * 0.5),
        # Stability parameters
        mass_min=0.01,
        energy_clip=1e8
    )
    
    # Initialize model
    gw = GwaveCore(cfg)
    
    # Place tokens in a spiral pattern to create vortex fields
    n_tokens = 24
    for i in range(n_tokens):
        # Log-spiral pattern
        t = i / n_tokens * 4 * np.pi
        r = PHI * np.exp(t / (PHI * 10))
        theta = t
        
        # Convert radius to log-radius
        ell = np.log(r)
        
        # z coordinate
        z = 2 * np.pi * i / n_tokens
        
        # Add token
        gw.add_token(ell, theta, z, 1.0)
    
    print(f"Placed {n_tokens} tokens in a spiral pattern")
    
    # Evolve system
    print(f"Evolving system for {steps} steps...")
    gw.evolve(steps)
    
    # Statistics
    crystallized = np.sum(gw.froz[:gw.N_act])
    tachyonic_events = len(gw.tachyonic_events)
    
    print("\nResults:")
    print(f"- Active tokens: {gw.N_act}")
    print(f"- Crystallized tokens: {crystallized}")
    print(f"- Tachyonic events: {tachyonic_events}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Custom visualization of vortex field
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Convert to Cartesian
    r = np.exp(gw.pos[:gw.N_act, 0])
    theta = gw.pos[:gw.N_act, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Filter out invalid positions
    valid_mask = np.isfinite(x) & np.isfinite(y)
    
    # Plot tokens
    scatter = ax.scatter(x[valid_mask], y[valid_mask], 
                       c=['blue' if f else 'red' for f in gw.froz[:gw.N_act] if valid_mask[i]], 
                       s=50, alpha=0.7)
    
    # Enhanced vortex field visualization
    vorticity = gw.vortex_field[:gw.N_act, 2]
    # Create a dense grid for vortex field interpolation
    grid_size = 20
    x_grid = np.linspace(np.min(x[valid_mask])-0.5, np.max(x[valid_mask])+0.5, grid_size)
    y_grid = np.linspace(np.min(y[valid_mask])-0.5, np.max(y[valid_mask])+0.5, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Initialize vortex field grid
    Z = np.zeros_like(X)
    
    # Simple inverse distance weighting interpolation
    for i in range(grid_size):
        for j in range(grid_size):
            weights = 0
            total = 0
            for k in range(gw.N_act):
                if not valid_mask[k] or not np.isfinite(vorticity[k]):
                    continue
                    
                # Calculate distance to grid point
                dist = np.sqrt((X[i,j] - x[k])**2 + (Y[i,j] - y[k])**2)
                if dist < 0.01:  # Avoid division by zero
                    dist = 0.01
                    
                # Inverse distance weighting
                weight = 1 / (dist**2)
                weights += weight
                total += vorticity[k] * weight
            
            if weights > 0:
                Z[i,j] = total / weights
    
    # Plot interpolated vortex field
    vortex_contour = ax.contourf(X, Y, Z, 15, cmap='coolwarm', alpha=0.5)
    plt.colorbar(vortex_contour, ax=ax, label='Vorticity')
    
    # Add streamplot to show field lines
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    # Calculate streamplot vectors
    for i in range(grid_size):
        for j in range(grid_size):
            # For 2D vorticity field in z direction
            U[i,j] = -Z[i,j] * (Y[i,j] - np.mean(y[valid_mask]))
            V[i,j] = Z[i,j] * (X[i,j] - np.mean(x[valid_mask]))
    
    # Normalize to prevent extreme vectors
    magnitude = np.sqrt(U**2 + V**2)
    max_mag = np.max(magnitude)
    if max_mag > 0:
        U = U / max_mag
        V = V / max_mag
    
    # Plot streamlines
    ax.streamplot(X, Y, U, V, color='black', linewidth=1, density=1, arrowsize=1)
    
    # Add preferred radii
    theta_circle = np.linspace(0, 2*np.pi, 100)
    
    # Critical radius for tachyonic behavior
    r_critical = gw.cfg.c * PHI**2 / np.pi
    x_crit = r_critical * np.cos(theta_circle)
    y_crit = r_critical * np.sin(theta_circle)
    ax.plot(x_crit, y_crit, 'red', alpha=0.5, linewidth=1, linestyle='--',
           label=f'r_crit = {r_critical:.3f}')
    
    # Labels and title
    ax.set_xlabel('X = r·cos(θ)')
    ax.set_ylabel('Y = r·sin(θ)')
    ax.set_title('Vortex Field Dynamics')
    ax.set_aspect('equal')
    ax.legend()
    
    # Save figure
    filename = "vortex_field_dynamics.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    
    # Create a 3D visualization of vortex field evolution
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Select tokens with significant vorticity
    significant_tokens = []
    for i in range(gw.N_act):
        if np.abs(vorticity[i]) > 0.2 * np.max(np.abs(vorticity[np.isfinite(vorticity)])):
            significant_tokens.append(i)
    
    # Plot trajectories colored by vorticity
    for i in significant_tokens:
        # Extract trajectory
        traj = np.array([pos[i] for pos in gw.traj])
        
        # Filter out invalid positions
        valid_mask = np.all(np.isfinite(traj), axis=1)
        if not np.any(valid_mask):
            continue
            
        traj = traj[valid_mask]
        
        # Convert from log-cylindrical to Cartesian
        ell = traj[:, 0]
        theta = traj[:, 1]
        z = traj[:, 2]
        
        r = np.exp(ell)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Color by vorticity
        points = ax.scatter(x, y, z, c=range(len(x)), cmap='plasma', 
                         s=30, alpha=0.8)
        
        # Connect with lines
        ax.plot(x, y, z, alpha=0.5)
    
    # Add labels
    ax.set_xlabel('X = r·cos(θ)')
    ax.set_ylabel('Y = r·sin(θ)')
    ax.set_zlabel('Z (rotor phase)')
    ax.set_title('Vortex Field Evolution in 3D')
    
    # Add colorbar to show time progression
    cbar = plt.colorbar(points, ax=ax)
    cbar.set_label('Time progression')
    
    # Save figure
    filename = "vortex_field_evolution_3d.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to {OUTPUT_DIR}/")
    
    return gw

def create_experiment_report(gw_tachyonic, gw_phi_n, gw_vortex):
    """Create a markdown report summarizing the experiments."""
    report_path = os.path.join(OUTPUT_DIR, "EXPERIMENT_REPORT.md")
    
    with open(report_path, 'w') as f:
        f.write("# Gwave Quantum Field Experiments Report\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report summarizes the results of three experiments with the Gwave framework:\n\n")
        f.write("1. **Tachyonic Helical Trajectories**: Investigating superluminal information propagation\n")
        f.write("2. **Phi^n Layer Stratification**: Observing natural stratification at φⁿ radii\n")
        f.write("3. **Vortex Field Dynamics**: Analyzing emergent vorticity patterns\n\n")
        
        f.write("## Experiment 1: Tachyonic Helical Trajectories\n\n")
        f.write(f"- **Tokens**: {gw_tachyonic.N_act}\n")
        f.write(f"- **Crystallized**: {np.sum(gw_tachyonic.froz[:gw_tachyonic.N_act])}\n")
        f.write(f"- **Tachyonic Events**: {len(gw_tachyonic.tachyonic_events)}\n\n")
        
        f.write("Tachyonic events occur when the phase velocity of a token exceeds the semantic speed of light. ")
        f.write("In log-cylindrical coordinates, this happens when the angular velocity at a given radius produces ")
        f.write("a tangential velocity greater than c.\n\n")
        
        f.write("The critical radius for tachyonic events is given by: r_crit = c · φ² / π ≈ ")
        f.write(f"{gw_tachyonic.cfg.c * PHI**2 / np.pi:.3f}\n\n")
        
        f.write("![Tachyonic Helical Trajectories](tachyonic_helical_trajectories.png)\n\n")
        
        f.write("## Experiment 2: Phi^n Layer Stratification\n\n")
        f.write(f"- **Tokens**: {gw_phi_n.N_act}\n")
        f.write(f"- **Crystallized**: {np.sum(gw_phi_n.froz[:gw_phi_n.N_act])}\n")
        f.write(f"- **Phi^n Layers Tracked**: {gw_phi_n.cfg.track_phi_n}\n\n")
        
        f.write("Phi^n layer tracking monitors how tokens naturally stratify at radii corresponding to powers of the golden ratio (φ). ")
        f.write("This demonstrates the geometric structure imposed by the log-cylindrical coordinate system.\n\n")
        
        f.write("Layer distribution:\n\n")
        for i, phi_n in enumerate(gw_phi_n.phi_n_layers):
            total = gw_phi_n.phi_n_counts[i, 0]
            crystallized = gw_phi_n.phi_n_counts[i, 1]
            if total > 0:
                f.write(f"- φ^{i+1} = {phi_n:.3f}: {total} tokens ({crystallized} crystallized)\n")
        f.write("\n")
        
        f.write("![Phi^n Layer Stratification](phi_n_stratification.png)\n\n")
        
        f.write("## Experiment 3: Vortex Field Dynamics\n\n")
        f.write(f"- **Tokens**: {gw_vortex.N_act}\n")
        f.write(f"- **Crystallized**: {np.sum(gw_vortex.froz[:gw_vortex.N_act])}\n")
        f.write(f"- **Tachyonic Events**: {len(gw_vortex.tachyonic_events)}\n\n")
        
        f.write("The vortex field represents the curl of the velocity field in phase space. ")
        f.write("It captures rotational motion and identifies regions of phase space with coherent circular patterns.\n\n")
        
        f.write("![Vortex Field Dynamics](vortex_field_dynamics.png)\n\n")
        f.write("![Vortex Field Evolution 3D](vortex_field_evolution_3d.png)\n\n")
        
        f.write("## Conclusions\n\n")
        f.write("These experiments demonstrate the rich dynamics of the Gwave framework:\n\n")
        f.write("1. Tachyonic events create helical trajectories in phase space, allowing information to tunnel across the manifold\n")
        f.write("2. The golden ratio (φ) provides natural stratification layers, consistent with optimal information geometry\n")
        f.write("3. Vortex fields emerge from token interactions, creating coherent patterns of circular motion\n\n")
        
        f.write("The log-cylindrical coordinate system (ℓ, θ, z) proves to be a powerful framework for modeling quantum-like ")
        f.write("information dynamics, where the geometry of the space itself guides token evolution in a way that ")
        f.write("naturally incorporates key quantum principles.\n")
    
    print(f"Report generated at {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Gwave Quantum Field Experiments")
    parser.add_argument("--experiment", type=str, default="all",
                      choices=["all", "tachyonic", "phi_n", "vortex"],
                      help="Which experiment to run")
    args = parser.parse_args()
    
    # Track experiment results
    results = {}
    
    if args.experiment in ["all", "tachyonic"]:
        results["tachyonic"] = experiment_tachyonic_helical()
        
    if args.experiment in ["all", "phi_n"]:
        results["phi_n"] = experiment_phi_n_layers()
        
    if args.experiment in ["all", "vortex"]:
        results["vortex"] = experiment_vortex_field()
    
    # Create report if all experiments were run
    if args.experiment == "all":
        create_experiment_report(
            results["tachyonic"], 
            results["phi_n"], 
            results["vortex"]
        )
        
    print("All experiments completed successfully!")

if __name__ == "__main__":
    main()
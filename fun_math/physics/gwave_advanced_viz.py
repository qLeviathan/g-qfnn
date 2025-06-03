#!/usr/bin/env python3
"""
Advanced Visualizations for Gwave Quantum Field Dynamics

Creates publication-quality visualizations that highlight:
- Wave mechanics in phase space
- Tachyonic helical trajectories with escape velocities
- Field potentials and loss landscapes
- Lévy flight inversions and tunneling events
- Phi^n stratification with heat maps

Uses a consistent visual language to tell the story of quantum field dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path
import colorsys
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from gwave_core_fixed import GwaveCore, PHI, safe_log, safe_sqrt

# Constant for detecting Lévy flights
GOLDEN_THR = PHI  # Same as in gwave_core_fixed.py

# Set up high-quality visualization defaults
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is installed
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Custom color maps
# Golden color scheme inspired by phi ratio
golden_colors = [(0.1, 0.1, 0.1), (0.4, 0.2, 0.0), (0.8, 0.5, 0.0), (1.0, 0.84, 0.0), (1.0, 1.0, 0.9)]
GOLDEN_CMAP = LinearSegmentedColormap.from_list('golden', golden_colors)

# Quantum blue-purple-pink colormap for wave functions
quantum_colors = [(0.0, 0.0, 0.2), (0.2, 0.2, 0.5), (0.5, 0.0, 0.5), (0.8, 0.0, 0.3), (1.0, 0.5, 0.0)]
QUANTUM_CMAP = LinearSegmentedColormap.from_list('quantum', quantum_colors)

# Vortex field colormap
vortex_colors = [(0.0, 0.0, 0.5), (0.0, 0.4, 0.7), (1.0, 1.0, 1.0), (0.7, 0.0, 0.0), (0.5, 0.0, 0.0)]
VORTEX_CMAP = LinearSegmentedColormap.from_list('vortex', vortex_colors)

# Output directory
OUTPUT_DIR = "outputs/gwave/physics/advanced_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class GwaveAdvancedViz:
    """Advanced visualization class for Gwave quantum field dynamics."""
    
    def __init__(self, gwave_core: GwaveCore):
        """Initialize with a GwaveCore instance."""
        self.gw = gwave_core
        self.output_dir = OUTPUT_DIR
        
        # Analysis metrics
        self._compute_trajectory_metrics()
        
    def _compute_trajectory_metrics(self):
        """Compute additional metrics for trajectory analysis."""
        if not self.gw.traj or self.gw.N_act == 0:
            return
            
        n_steps = len(self.gw.traj)
        n_tokens = self.gw.N_act
        
        # Compute displacement magnitudes for each token across time
        self.displacement = np.zeros((n_tokens, n_steps-1))
        self.angular_velocity = np.zeros((n_tokens, n_steps-1))
        self.radial_velocity = np.zeros((n_tokens, n_steps-1))
        self.levy_flight_events = []
        
        for i in range(n_tokens):
            for t in range(1, n_steps):
                # Skip if either position is invalid
                if not (np.all(np.isfinite(self.gw.traj[t-1][i])) and 
                        np.all(np.isfinite(self.gw.traj[t][i]))):
                    continue
                
                # Get positions
                prev_pos = self.gw.traj[t-1][i]
                curr_pos = self.gw.traj[t][i]
                
                # Calculate log-radial displacement
                d_ell = curr_pos[0] - prev_pos[0]
                
                # Calculate angular displacement (with wrapping)
                d_theta = np.abs((curr_pos[1] - prev_pos[1] + np.pi) % (2*np.pi) - np.pi)
                
                # Store displacement magnitude
                self.displacement[i, t-1] = np.sqrt(d_ell**2 + d_theta**2)
                
                # Calculate velocities
                dt = PHI**-2  # Time step
                self.radial_velocity[i, t-1] = d_ell / dt
                self.angular_velocity[i, t-1] = d_theta / dt
                
                # Detect Lévy flight events (large sudden displacements)
                if self.displacement[i, t-1] > GOLDEN_THR and not self.gw.froz[i]:
                    self.levy_flight_events.append({
                        'token': i,
                        'time_step': t,
                        'time': t * dt,
                        'displacement': self.displacement[i, t-1],
                        'position_before': prev_pos.copy(),
                        'position_after': curr_pos.copy()
                    })
    
    def visualize_wave_mechanics(self, filename="wave_mechanics.png"):
        """
        Create a comprehensive visualization of wave mechanics in phase space.
        Shows token positions, velocities, and wave function intensity.
        """
        fig = plt.figure(figsize=(16, 14))
        
        # Create 3D and 2D subplots
        ax1 = fig.add_subplot(221, projection='3d')  # 3D trajectory view
        ax2 = fig.add_subplot(222)                  # 2D phase space with velocity
        ax3 = fig.add_subplot(223)                  # Intensity heat map
        ax4 = fig.add_subplot(224)                  # Vortex field
        
        # Get current token positions
        r = np.exp(self.gw.pos[:self.gw.N_act, 0])
        theta = self.gw.pos[:self.gw.N_act, 1]
        z = self.gw.pos[:self.gw.N_act, 2]
        
        # Convert to Cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Filter out invalid positions
        valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        
        # Color tokens by their crystallization state
        token_colors = []
        for i in range(self.gw.N_act):
            if not valid_mask[i]:
                continue
                
            if self.gw.froz[i]:
                # Crystallized - use golden colors
                token_colors.append((0.8, 0.6, 0.0, 0.8))
            else:
                # Active - use blue/purple colors
                token_colors.append((0.2, 0.3, 0.8, 0.8))
        
        # Size tokens by their field strength
        max_s = np.max(self.gw.s[:self.gw.N_act]) if self.gw.N_act > 0 else 1.0
        token_sizes = 100 * (0.5 + 0.5 * self.gw.s[:self.gw.N_act] / (max_s + 1e-10))
        
        # 1. 3D Trajectory View
        # ====================
        # Only plot the last 50 steps of trajectory for clarity
        traj_length = min(50, len(self.gw.traj))
        if traj_length > 0:
            for i in range(self.gw.N_act):
                if not valid_mask[i]:
                    continue
                
                # Extract recent trajectory
                traj = np.array([pos[i] for pos in self.gw.traj[-traj_length:]])
                
                # Filter out invalid positions
                traj_valid = np.all(np.isfinite(traj), axis=1)
                if not np.any(traj_valid):
                    continue
                
                traj = traj[traj_valid]
                
                # Convert to Cartesian
                r_traj = np.exp(traj[:, 0])
                theta_traj = traj[:, 1]
                z_traj = traj[:, 2]
                
                x_traj = r_traj * np.cos(theta_traj)
                y_traj = r_traj * np.sin(theta_traj)
                
                # Color trajectory by time (newer = brighter)
                n_points = len(x_traj)
                colors = []
                for j in range(n_points):
                    alpha = 0.3 + 0.7 * j / (n_points - 1 + 1e-10) if n_points > 1 else 0.7
                    if self.gw.froz[i]:
                        colors.append((0.8, 0.6, 0.0, alpha))  # Gold for crystallized
                    else:
                        colors.append((0.2, 0.3, 0.8, alpha))  # Blue for active
                
                # Plot trajectory with color gradient
                for j in range(1, n_points):
                    ax1.plot(x_traj[j-1:j+1], y_traj[j-1:j+1], z_traj[j-1:j+1], 
                            color=colors[j], linewidth=1.5)
                
                # Mark latest position
                if n_points > 0:
                    ax1.scatter([x_traj[-1]], [y_traj[-1]], [z_traj[-1]], 
                              color=colors[-1], s=token_sizes[i], 
                              edgecolors='white', linewidth=0.5)
        
        # Add critical surfaces for tachyonic events
        u = np.linspace(0, 2*np.pi, 32)
        v = np.linspace(0, 2*np.pi, 16)
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Critical radius for tachyonic events
        r_critical = self.gw.cfg.c * PHI**2 / np.pi
        
        # Create critical surface
        x_crit = r_critical * np.cos(u_grid) * np.cos(v_grid)
        y_crit = r_critical * np.sin(u_grid) * np.cos(v_grid)
        z_crit = r_critical * np.sin(v_grid)
        
        # Plot critical surface
        ax1.plot_surface(x_crit, y_crit, z_crit, color='r', alpha=0.15, 
                       rstride=1, cstride=1, linewidth=0.1, edgecolor='r')
        
        # Annotate tachyonic events in 3D
        for event in self.gw.tachyonic_events[-10:]:  # Show only the last 10 events
            token_idx = event['token']
            pos = event['position']
            
            # Convert to Cartesian
            r_event = np.exp(pos[0])
            theta_event = pos[1]
            z_event = pos[2]
            
            x_event = r_event * np.cos(theta_event)
            y_event = r_event * np.sin(theta_event)
            
            # Plot event
            ax1.scatter([x_event], [y_event], [z_event], color='red', s=100, 
                      marker='*', edgecolors='white', linewidth=0.5)
        
        # Add phi^n layers
        for n in range(1, 4):
            r_phi_n = PHI**n
            x_phi = r_phi_n * np.cos(u_grid) * np.cos(v_grid)
            y_phi = r_phi_n * np.sin(u_grid) * np.cos(v_grid)
            z_phi = r_phi_n * np.sin(v_grid)
            
            ax1.plot_surface(x_phi, y_phi, z_phi, color='gold', alpha=0.1, 
                           rstride=1, cstride=1, linewidth=0.1, edgecolor='gold')
        
        # Axes settings
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.set_zlabel('Z', fontsize=12)
        ax1.set_title('Wave Dynamics in 3D Phase Space', fontsize=14)
        ax1.grid(False)
        
        # 2. 2D Phase Space with Velocity Vectors
        # =======================================
        # Plot tokens
        scatter = ax2.scatter(x[valid_mask], y[valid_mask], 
                            c=token_colors, s=token_sizes[valid_mask], 
                            alpha=0.9, zorder=10)
        
        # Add velocity vectors if we have trajectory data
        if len(self.gw.traj) > 1:
            # Calculate velocities from last two trajectory points
            last_pos = self.gw.traj[-1]
            prev_pos = self.gw.traj[-2]
            
            for i in range(self.gw.N_act):
                if not valid_mask[i] or self.gw.froz[i]:
                    continue
                
                # Skip if positions are invalid
                if not (np.all(np.isfinite(last_pos[i])) and 
                        np.all(np.isfinite(prev_pos[i]))):
                    continue
                
                # Calculate displacement
                d_ell = last_pos[i, 0] - prev_pos[i, 0]
                d_theta = (last_pos[i, 1] - prev_pos[i, 1] + np.pi) % (2*np.pi) - np.pi
                
                # Convert to Cartesian velocity
                dr = d_ell * r[i]  # Approximate radial change
                dx = dr * np.cos(theta[i]) - r[i] * d_theta * np.sin(theta[i])
                dy = dr * np.sin(theta[i]) + r[i] * d_theta * np.cos(theta[i])
                
                # Scale for visibility
                scale = 5.0
                dx *= scale
                dy *= scale
                
                # Draw velocity vector
                ax2.arrow(x[i], y[i], dx, dy, 
                        head_width=0.1, head_length=0.15, 
                        fc='white', ec='white', alpha=0.7)
        
        # Add phi^n circles
        theta_circle = np.linspace(0, 2*np.pi, 100)
        for n in range(1, 6):
            r_phi_n = PHI**n
            x_phi = r_phi_n * np.cos(theta_circle)
            y_phi = r_phi_n * np.sin(theta_circle)
            ax2.plot(x_phi, y_phi, 'gold', alpha=0.5, linewidth=1, 
                   label=f'$\\phi^{n}$' if n == 1 else "")
        
        # Add critical radius
        x_crit = r_critical * np.cos(theta_circle)
        y_crit = r_critical * np.sin(theta_circle)
        ax2.plot(x_crit, y_crit, 'r--', alpha=0.7, linewidth=1.5, 
               label='Tachyonic Threshold')
        
        # Axes settings
        ax2.set_xlabel('X = r·cos(θ)', fontsize=12)
        ax2.set_ylabel('Y = r·sin(θ)', fontsize=12)
        ax2.set_title('Phase Space with Velocity Vectors', fontsize=14)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        # 3. Intensity Heat Map
        # ===================
        # Create a grid for the heat map
        grid_size = 200
        x_grid = np.linspace(-5, 5, grid_size)
        y_grid = np.linspace(-5, 5, grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Initialize intensity field
        intensity = np.zeros((grid_size, grid_size))
        
        # Generate intensity field from token positions
        for i in range(self.gw.N_act):
            if not valid_mask[i]:
                continue
            
            # Convert to polar coordinates for grid
            R_grid = np.sqrt(X_grid**2 + Y_grid**2)
            Theta_grid = np.arctan2(Y_grid, X_grid)
            
            # Token log-radial and angular position
            ell_i = self.gw.pos[i, 0]
            theta_i = self.gw.pos[i, 1]
            
            # Calculate log-radial and angular differences
            d_ell = np.log(R_grid + 1e-10) - ell_i
            d_theta = np.minimum(
                np.abs(Theta_grid - theta_i),
                2*np.pi - np.abs(Theta_grid - theta_i)
            )
            
            # Calculate intensity contribution (Gaussian)
            sigma_r = 0.5
            sigma_theta = np.pi/4
            intensity_i = (
                self.gw.s[i] * 
                np.exp(-d_ell**2/(2*sigma_r**2)) * 
                np.exp(-d_theta**2/(2*sigma_theta**2))
            )
            
            # Add to total intensity
            intensity += intensity_i
        
        # Apply Gaussian blur for smoother field
        intensity = gaussian_filter(intensity, sigma=1.0)
        
        # Create heat map
        im = ax3.pcolormesh(X_grid, Y_grid, intensity, cmap=QUANTUM_CMAP, 
                         shading='gouraud', alpha=0.8)
        
        # Add contour lines
        contour_levels = np.linspace(0, np.max(intensity), 10)
        ax3.contour(X_grid, Y_grid, intensity, levels=contour_levels, 
                  colors='white', alpha=0.3, linewidths=0.5)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax3)
        cbar.set_label('Field Intensity')
        
        # Plot token positions on heat map
        ax3.scatter(x[valid_mask], y[valid_mask], c='white', s=10, alpha=0.7)
        
        # Add phi^n circles
        for n in range(1, 6):
            r_phi_n = PHI**n
            x_phi = r_phi_n * np.cos(theta_circle)
            y_phi = r_phi_n * np.sin(theta_circle)
            ax3.plot(x_phi, y_phi, 'white', alpha=0.5, linewidth=1)
        
        # Add critical radius
        ax3.plot(x_crit, y_crit, 'r--', alpha=0.7, linewidth=1.5)
        
        # Axes settings
        ax3.set_xlabel('X = r·cos(θ)', fontsize=12)
        ax3.set_ylabel('Y = r·sin(θ)', fontsize=12)
        ax3.set_title('Quantum Field Intensity', fontsize=14)
        ax3.set_aspect('equal')
        
        # 4. Vortex Field
        # ==============
        # Create a grid for the vortex field
        grid_size = 50
        x_grid = np.linspace(-5, 5, grid_size)
        y_grid = np.linspace(-5, 5, grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Initialize vortex field
        vorticity = np.zeros((grid_size, grid_size))
        
        # Get vorticity from tokens
        token_vorticity = self.gw.vortex_field[:self.gw.N_act, 2]
        
        # Interpolate vorticity to grid using inverse distance weighting
        for i in range(grid_size):
            for j in range(grid_size):
                weights = 0
                total = 0
                for k in range(self.gw.N_act):
                    if not valid_mask[k] or not np.isfinite(token_vorticity[k]):
                        continue
                    
                    # Calculate distance to grid point
                    dist = np.sqrt((X_grid[i,j] - x[k])**2 + (Y_grid[i,j] - y[k])**2)
                    if dist < 0.01:  # Avoid division by zero
                        dist = 0.01
                    
                    # Inverse distance weighting
                    weight = 1 / (dist**2)
                    weights += weight
                    total += token_vorticity[k] * weight
                
                if weights > 0:
                    vorticity[i,j] = total / weights
        
        # Apply Gaussian blur for smoother field
        vorticity = gaussian_filter(vorticity, sigma=1.0)
        
        # Create vorticity heat map with diverging colormap
        vmax = np.max(np.abs(vorticity))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax4.pcolormesh(X_grid, Y_grid, vorticity, cmap=VORTEX_CMAP, 
                         norm=norm, shading='gouraud', alpha=0.8)
        
        # Add streamplot to show vortex field lines
        # Calculate streamplot vectors (perpendicular to vorticity gradient)
        U = np.zeros_like(X_grid)
        V = np.zeros_like(Y_grid)
        
        # Calculate simplified curl vectors for streamplot
        for i in range(grid_size):
            for j in range(grid_size):
                U[i,j] = -vorticity[i,j] * (Y_grid[i,j] - 0)
                V[i,j] = vorticity[i,j] * (X_grid[i,j] - 0)
        
        # Normalize
        magnitude = np.sqrt(U**2 + V**2)
        max_mag = np.max(magnitude)
        if max_mag > 0:
            U = U / max_mag
            V = V / max_mag
        
        # Plot streamlines
        ax4.streamplot(x_grid, y_grid, U.T, V.T, color='white', linewidth=0.7, 
                     density=1.5, arrowsize=0.8)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax4)
        cbar.set_label('Vorticity')
        
        # Plot token positions on vorticity map
        ax4.scatter(x[valid_mask], y[valid_mask], c='black', s=10, alpha=0.7)
        
        # Add critical radius
        ax4.plot(x_crit, y_crit, 'r--', alpha=0.7, linewidth=1.5)
        
        # Axes settings
        ax4.set_xlabel('X = r·cos(θ)', fontsize=12)
        ax4.set_ylabel('Y = r·sin(θ)', fontsize=12)
        ax4.set_title('Vortex Field with Flow Lines', fontsize=14)
        ax4.set_aspect('equal')
        
        # Global figure settings
        plt.tight_layout()
        fig.suptitle('Gwave Quantum Field Dynamics', fontsize=16)
        plt.subplots_adjust(top=0.92)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {os.path.join(self.output_dir, filename)}")
        
        return fig
    
    def visualize_tachyonic_helical(self, filename="tachyonic_helical_advanced.png"):
        """
        Create striking visualization of tachyonic helical trajectories.
        Highlights escape velocities and Lévy flights.
        """
        if not self.gw.tachyonic_events:
            print("No tachyonic events to visualize")
            return None
        
        fig = plt.figure(figsize=(15, 12))
        
        # Create 3D and 2D subplots
        ax1 = fig.add_subplot(221, projection='3d')  # 3D helical trajectories
        ax2 = fig.add_subplot(222)                  # Velocity phase portrait
        ax3 = fig.add_subplot(223, projection='3d')  # Space-time cone
        ax4 = fig.add_subplot(224)                  # Energy-momentum profile
        
        # Get unique tokens that experienced tachyonic events
        tachyonic_tokens = set(event['token'] for event in self.gw.tachyonic_events)
        
        # 1. 3D Helical Trajectories
        # =========================
        # Setup custom lighting for better 3D visualization
        ax1.view_init(elev=30, azim=45)
        
        # Track all trajectories but highlight tachyonic ones
        for i in range(self.gw.N_act):
            # Extract trajectory
            traj = np.array([pos[i] for pos in self.gw.traj])
            
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
            
            # Skip non-tachyonic tokens for clarity
            if i not in tachyonic_tokens:
                continue
            
            # Plot trajectory with gradient color by time
            points = np.array([x, y, z]).T
            segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
            
            # Create gradient line collection
            norm = plt.Normalize(0, len(segments))
            line_colors = [plt.cm.plasma(norm(i)) for i in range(len(segments))]
            
            for j in range(len(segments)):
                ax1.plot(segments[j,:,0], segments[j,:,1], segments[j,:,2], 
                      color=line_colors[j], linewidth=2, alpha=0.8)
            
            # Find tachyonic events for this token
            token_events = [event for event in self.gw.tachyonic_events if event['token'] == i]
            
            # Mark tachyonic events with explosion-like markers
            for event in token_events:
                pos = event['position']
                
                # Convert to Cartesian
                r_event = np.exp(pos[0])
                theta_event = pos[1]
                z_event = pos[2]
                
                x_event = r_event * np.cos(theta_event)
                y_event = r_event * np.sin(theta_event)
                
                # Add explosion marker
                ax1.scatter([x_event], [y_event], [z_event], 
                         color='yellow', s=200, marker='*', 
                         edgecolors='red', linewidth=1.5, alpha=0.9)
                
                # Add light cone at tachyonic event
                u = np.linspace(0, 2*np.pi, 20)
                v = np.linspace(0, np.pi/3, 10)
                u_grid, v_grid = np.meshgrid(u, v)
                
                cone_height = 0.5
                
                x_cone = x_event + cone_height * v_grid * np.cos(u_grid)
                y_cone = y_event + cone_height * v_grid * np.sin(u_grid)
                z_cone = z_event + cone_height * v_grid
                
                ax1.plot_surface(x_cone, y_cone, z_cone, color='red', alpha=0.2)
        
        # Add critical surface
        u = np.linspace(0, 2*np.pi, 32)
        v = np.linspace(0, 2*np.pi, 16)
        u_grid, v_grid = np.meshgrid(u, v)
        
        # Critical radius for tachyonic events
        r_critical = self.gw.cfg.c * PHI**2 / np.pi
        
        # Create critical torus
        theta_torus = u_grid
        phi_torus = v_grid
        
        x_torus = r_critical * np.cos(theta_torus)
        y_torus = r_critical * np.sin(theta_torus)
        z_torus = 0.2 * np.sin(phi_torus)
        
        # Plot critical surface
        ax1.plot_surface(x_torus, y_torus, z_torus, color='red', alpha=0.15, 
                      linewidth=0.1, antialiased=True)
        
        # Add reference helix
        t_ref = np.linspace(0, 4*np.pi, 200)
        r_ref = 1/PHI
        x_ref = r_ref * np.cos(PHI * t_ref)
        y_ref = r_ref * np.sin(PHI * t_ref)
        z_ref = 0.2 * t_ref
        
        ax1.plot(x_ref, y_ref, z_ref, color='white', linestyle='--', 
              linewidth=1, alpha=0.5, label='Φ-frequency helix')
        
        # Axes settings
        ax1.set_xlabel('X', fontsize=12, labelpad=10)
        ax1.set_ylabel('Y', fontsize=12, labelpad=10)
        ax1.set_zlabel('Z', fontsize=12, labelpad=10)
        ax1.set_title('Tachyonic Helical Trajectories', fontsize=14)
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        
        # 2. Velocity Phase Portrait
        # ========================
        # Create a scatter plot of angular vs. radial velocity
        v_r_all = []
        v_theta_all = []
        colors = []
        sizes = []
        
        for i in range(self.gw.N_act):
            for t in range(len(self.gw.traj) - 1):
                # Skip if data is not valid
                if not (t < self.radial_velocity.shape[1] and 
                        np.isfinite(self.radial_velocity[i, t]) and 
                        np.isfinite(self.angular_velocity[i, t])):
                    continue
                
                v_r = self.radial_velocity[i, t]
                v_theta = self.angular_velocity[i, t]
                
                v_r_all.append(v_r)
                v_theta_all.append(v_theta)
                
                # Color by whether this token is tachyonic
                if i in tachyonic_tokens:
                    # Calculate time-dependent color (newer = brighter)
                    time_factor = t / (len(self.gw.traj) - 2)
                    colors.append((1.0, 0.5*time_factor, 0, 0.5 + 0.5*time_factor))
                    sizes.append(20 + 30*time_factor)
                else:
                    colors.append((0.3, 0.3, 0.7, 0.3))
                    sizes.append(10)
        
        # Plot velocity phase portrait
        ax2.scatter(v_r_all, v_theta_all, c=colors, s=sizes, alpha=0.7)
        
        # Mark tachyonic threshold
        if v_theta_all:
            v_theta_max = max(max(v_theta_all), 10)
            v_r_vals = np.linspace(-5, 5, 100)
            v_theta_threshold = self.gw.cfg.c / np.array([np.exp(v) for v in v_r_vals])
            ax2.plot(v_r_vals, v_theta_threshold, 'r--', linewidth=2, 
                   label='Tachyonic Threshold')
        
        # Add diagonal guidelines for phi-ratio slopes
        v_r_range = np.linspace(-5, 5, 100)
        ax2.plot(v_r_range, PHI * v_r_range, 'g--', alpha=0.5, 
               label=f'Slope = φ')
        ax2.plot(v_r_range, v_r_range / PHI, 'g:', alpha=0.5, 
               label=f'Slope = 1/φ')
        
        # Axes settings
        ax2.set_xlabel('Radial Velocity (dℓ/dt)', fontsize=12)
        ax2.set_ylabel('Angular Velocity (dθ/dt)', fontsize=12)
        ax2.set_title('Velocity Phase Portrait', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        # 3. Space-Time Cone for Tachyonic Events
        # =======================================
        # Set up for light cone visualization
        ax3.view_init(elev=20, azim=30)
        
        # Plot a space-time diagram showing light cones and tachyonic events
        time_axis = np.linspace(0, self.gw.t, min(20, len(self.gw.traj)))
        
        # Plot world lines for tachyonic tokens
        for i in tachyonic_tokens:
            # Get trajectory timepoints
            trajectory = []
            for t, traj_point in enumerate(self.gw.traj):
                if t >= len(time_axis):
                    break
                    
                if np.all(np.isfinite(traj_point[i])):
                    pos = traj_point[i]
                    r = np.exp(pos[0])
                    theta = pos[1]
                    
                    x_pos = r * np.cos(theta)
                    y_pos = r * np.sin(theta)
                    
                    trajectory.append((x_pos, y_pos, time_axis[t]))
            
            # Plot world line
            if trajectory:
                traj_array = np.array(trajectory)
                ax3.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 
                       color='blue', alpha=0.7, linewidth=1.5)
        
        # Add light cones at tachyonic events
        for event in self.gw.tachyonic_events:
            pos = event['position']
            time = event['time']
            
            # Convert to Cartesian
            r_event = np.exp(pos[0])
            theta_event = pos[1]
            
            x_event = r_event * np.cos(theta_event)
            y_event = r_event * np.sin(theta_event)
            
            # Create light cone
            u = np.linspace(0, 2*np.pi, 20)
            h = np.linspace(0, 1, 10)
            u_grid, h_grid = np.meshgrid(u, h)
            
            cone_radius = h_grid * self.gw.cfg.c
            
            x_cone = x_event + cone_radius * np.cos(u_grid)
            y_cone = y_event + cone_radius * np.sin(u_grid)
            z_cone = time + h_grid
            
            # Plot future light cone
            ax3.plot_surface(x_cone, y_cone, z_cone, color='red', alpha=0.2, 
                          rstride=1, cstride=2, linewidth=0)
            
            # Mark event point
            ax3.scatter([x_event], [y_event], [time], color='yellow', s=100, 
                      marker='*', edgecolors='red', linewidth=1)
        
        # Add coordinate axes in space-time
        ax3.plot([0, 5], [0, 0], [0, 0], 'k-', linewidth=1)  # x-axis
        ax3.plot([0, 0], [0, 5], [0, 0], 'k-', linewidth=1)  # y-axis
        ax3.plot([0, 0], [0, 0], [0, max(time_axis)], 'k-', linewidth=1)  # t-axis
        
        # Axes settings
        ax3.set_xlabel('X', fontsize=12, labelpad=10)
        ax3.set_ylabel('Y', fontsize=12, labelpad=10)
        ax3.set_zlabel('Time', fontsize=12, labelpad=10)
        ax3.set_title('Space-Time Diagram with Light Cones', fontsize=14)
        ax3.xaxis.pane.fill = False
        ax3.yaxis.pane.fill = False
        ax3.zaxis.pane.fill = False
        
        # 4. Energy-Momentum Profile
        # ========================
        # Plot energy history with tachyonic events marked
        energy_history = np.array(self.gw.energy_history)
        valid_energy = np.isfinite(energy_history)
        
        if np.any(valid_energy):
            time_steps = np.arange(len(energy_history))[valid_energy]
            energy_values = energy_history[valid_energy]
            
            ax4.plot(time_steps, energy_values, 'b-', linewidth=1.5)
            
            # Add markers for tachyonic events
            for event in self.gw.tachyonic_events:
                t_idx = int(event['time'] / (PHI**-2))
                if t_idx < len(energy_history) and np.isfinite(energy_history[t_idx]):
                    ax4.scatter(t_idx, energy_history[t_idx], color='red', s=100, 
                             marker='*', zorder=10, edgecolors='yellow', linewidth=1)
            
            # Add markers for Lévy flights
            for event in self.levy_flight_events:
                t_idx = event['time_step']
                if t_idx < len(energy_history) and np.isfinite(energy_history[t_idx]):
                    ax4.scatter(t_idx, energy_history[t_idx], color='green', s=80, 
                             marker='o', zorder=9, edgecolors='white', linewidth=1)
            
            # Add regions with background color for tachyonic events
            for event in self.gw.tachyonic_events:
                t_idx = int(event['time'] / (PHI**-2))
                if t_idx < len(energy_history):
                    ax4.axvspan(t_idx-5, t_idx+5, color='red', alpha=0.1)
            
            # Add annotations for key events
            if self.gw.tachyonic_events:
                event = self.gw.tachyonic_events[-1]  # Get the last event
                t_idx = int(event['time'] / (PHI**-2))
                if t_idx < len(energy_history) and np.isfinite(energy_history[t_idx]):
                    ax4.annotate('Tachyonic\nEvent', 
                              xy=(t_idx, energy_history[t_idx]),
                              xytext=(t_idx+10, energy_history[t_idx]*1.2),
                              arrowprops=dict(facecolor='red', shrink=0.05),
                              ha='center')
            
            if self.levy_flight_events:
                event = self.levy_flight_events[-1]  # Get the last event
                t_idx = event['time_step']
                if t_idx < len(energy_history) and np.isfinite(energy_history[t_idx]):
                    ax4.annotate('Lévy Flight', 
                              xy=(t_idx, energy_history[t_idx]),
                              xytext=(t_idx-10, energy_history[t_idx]*1.3),
                              arrowprops=dict(facecolor='green', shrink=0.05),
                              ha='center')
        
        # Axes settings
        ax4.set_xlabel('Time Steps', fontsize=12)
        ax4.set_ylabel('Energy', fontsize=12)
        ax4.set_title('Energy Profile with Tachyonic Events', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Add legend
        ax4.plot([], [], 'b-', linewidth=1.5, label='Energy')
        ax4.scatter([], [], color='red', s=100, marker='*', 
                 edgecolors='yellow', linewidth=1, label='Tachyonic Event')
        ax4.scatter([], [], color='green', s=80, marker='o', 
                 edgecolors='white', linewidth=1, label='Lévy Flight')
        ax4.legend(loc='upper right')
        
        # Global figure settings
        plt.tight_layout()
        fig.suptitle('Tachyonic Phenomena in Quantum Phase Space', fontsize=16)
        plt.subplots_adjust(top=0.92)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {os.path.join(self.output_dir, filename)}")
        
        return fig
    
    def visualize_loss_landscape(self, filename="loss_landscape.png"):
        """
        Visualize the loss landscape with tokens navigating through it.
        Shows the valleys and ridges that guide token movement.
        """
        fig = plt.figure(figsize=(15, 12))
        
        # Create subplots for different views
        ax1 = fig.add_subplot(221, projection='3d')  # 3D loss landscape
        ax2 = fig.add_subplot(222)                  # 2D contour with trajectories
        ax3 = fig.add_subplot(223)                  # Gradient field
        ax4 = fig.add_subplot(224)                  # Loss profile along trajectories
        
        # Calculate token positions in Cartesian
        r = np.exp(self.gw.pos[:self.gw.N_act, 0])
        theta = self.gw.pos[:self.gw.N_act, 1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Filter out invalid positions
        valid_mask = np.isfinite(x) & np.isfinite(y)
        
        # Create a grid for the loss landscape
        grid_size = 100
        x_grid = np.linspace(-5, 5, grid_size)
        y_grid = np.linspace(-5, 5, grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Convert to log-cylindrical
        R_grid = np.sqrt(X_grid**2 + Y_grid**2)
        Theta_grid = np.arctan2(Y_grid, X_grid)
        Ell_grid = np.log(R_grid + 1e-10)
        
        # Calculate loss landscape based on token positions and forces
        loss = np.zeros((grid_size, grid_size))
        
        # Generate repulsive potential from token positions
        for i in range(self.gw.N_act):
            if not valid_mask[i]:
                continue
                
            # Token parameters
            ell_i = self.gw.pos[i, 0]
            theta_i = self.gw.pos[i, 1]
            strength = self.gw.s[i]
            
            # Distance in log-cylindrical coordinates
            d_ell = Ell_grid - ell_i
            d_theta = np.minimum(
                np.abs(Theta_grid - theta_i),
                2*np.pi - np.abs(Theta_grid - theta_i)
            )
            
            # Combined squared distance
            d2 = d_ell**2 + d_theta**2
            
            # Repulsive potential (inverse of distance)
            potential = strength / (d2 + 0.1)
            
            # Add to loss landscape
            loss += potential
        
        # Add phi^n attractors to the loss landscape
        for n in range(1, 6):
            r_phi_n = PHI**n
            ell_phi_n = np.log(r_phi_n)
            
            # Attractive potential at phi^n radius
            d_ell = np.abs(Ell_grid - ell_phi_n)
            potential = -0.5 * np.exp(-d_ell**2 / 0.1)
            
            # Add to loss landscape
            loss += potential
        
        # Add critical radius barrier
        r_critical = self.gw.cfg.c * PHI**2 / np.pi
        ell_critical = np.log(r_critical)
        
        # Repulsive barrier at critical radius
        d_ell = np.abs(Ell_grid - ell_critical)
        barrier = 0.8 * np.exp(-d_ell**2 / 0.05)
        
        # Add to loss landscape
        loss += barrier
        
        # Apply Gaussian blur for smoother landscape
        loss = gaussian_filter(loss, sigma=1.0)
        
        # Normalize for better visualization
        loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss) + 1e-10)
        
        # 1. 3D Loss Landscape
        # ===================
        # Plot 3D surface
        surf = ax1.plot_surface(X_grid, Y_grid, loss, cmap=GOLDEN_CMAP, 
                             linewidth=0, antialiased=True, alpha=0.8)
        
        # Plot token positions on surface
        for i in range(self.gw.N_act):
            if not valid_mask[i]:
                continue
                
            # Find the loss value at token position
            x_idx = np.argmin(np.abs(x_grid - x[i]))
            y_idx = np.argmin(np.abs(y_grid - y[i]))
            z_val = loss[y_idx, x_idx]
            
            # Plot token with color based on crystallization
            if self.gw.froz[i]:
                color = 'gold'
            else:
                color = 'blue'
                
            ax1.scatter([x[i]], [y[i]], [z_val + 0.05], color=color, s=50, 
                      edgecolors='white', linewidth=0.5)
        
        # Add trajectory paths on surface for a few selected tokens
        if len(self.gw.traj) > 10:
            # Select a few interesting tokens
            selected_tokens = []
            
            # Include some tachyonic tokens
            tachyonic_tokens = set(event['token'] for event in self.gw.tachyonic_events)
            selected_tokens.extend(list(tachyonic_tokens)[:3])
            
            # Include some levy flight tokens
            levy_tokens = set(event['token'] for event in self.levy_flight_events)
            selected_tokens.extend(list(levy_tokens - set(selected_tokens))[:2])
            
            # Include some other tokens if needed
            while len(selected_tokens) < 5 and len(selected_tokens) < self.gw.N_act:
                token = np.random.randint(0, self.gw.N_act)
                if token not in selected_tokens:
                    selected_tokens.append(token)
            
            # Plot trajectories for selected tokens
            for i in selected_tokens:
                # Get trajectory
                traj = np.array([pos[i] for pos in self.gw.traj[-50:]])  # Last 50 steps
                
                # Filter out invalid positions
                valid_traj = np.all(np.isfinite(traj), axis=1)
                if not np.any(valid_traj):
                    continue
                    
                traj = traj[valid_traj]
                
                # Convert to Cartesian
                r_traj = np.exp(traj[:, 0])
                theta_traj = traj[:, 1]
                
                x_traj = r_traj * np.cos(theta_traj)
                y_traj = r_traj * np.sin(theta_traj)
                
                # Find loss values along trajectory
                z_traj = []
                for j in range(len(x_traj)):
                    x_idx = np.argmin(np.abs(x_grid - x_traj[j]))
                    y_idx = np.argmin(np.abs(y_grid - y_traj[j]))
                    z_traj.append(loss[y_idx, x_idx] + 0.02)  # Small offset for visibility
                
                # Plot trajectory
                ax1.plot(x_traj, y_traj, z_traj, 'r-', linewidth=1.5, alpha=0.7)
        
        # Axes settings
        ax1.set_xlabel('X', fontsize=12, labelpad=5)
        ax1.set_ylabel('Y', fontsize=12, labelpad=5)
        ax1.set_zlabel('Loss', fontsize=12, labelpad=5)
        ax1.set_title('3D Loss Landscape', fontsize=14)
        ax1.view_init(elev=30, azim=45)
        
        # 2. 2D Contour with Trajectories
        # ==============================
        # Create contour plot
        contour = ax2.contourf(X_grid, Y_grid, loss, 20, cmap=GOLDEN_CMAP, alpha=0.7)
        cbar = fig.colorbar(contour, ax=ax2)
        cbar.set_label('Loss')
        
        # Add contour lines
        ax2.contour(X_grid, Y_grid, loss, 20, colors='white', linewidths=0.5, alpha=0.7)
        
        # Plot token positions
        token_colors = ['gold' if f else 'blue' for f in self.gw.froz[:self.gw.N_act]]
        ax2.scatter(x[valid_mask], y[valid_mask], 
                 c=[token_colors[i] for i in range(len(token_colors)) if valid_mask[i]], 
                 s=50, edgecolors='white', linewidth=0.5, zorder=10)
        
        # Plot phi^n circles
        theta_circle = np.linspace(0, 2*np.pi, 100)
        for n in range(1, 6):
            r_phi_n = PHI**n
            x_phi = r_phi_n * np.cos(theta_circle)
            y_phi = r_phi_n * np.sin(theta_circle)
            ax2.plot(x_phi, y_phi, 'white', alpha=0.7, linewidth=1, 
                   linestyle='--' if n % 2 == 0 else '-')
        
        # Add critical radius
        r_critical = self.gw.cfg.c * PHI**2 / np.pi
        x_crit = r_critical * np.cos(theta_circle)
        y_crit = r_critical * np.sin(theta_circle)
        ax2.plot(x_crit, y_crit, 'r--', alpha=0.7, linewidth=1.5)
        
        # Plot trajectories for all tokens
        if len(self.gw.traj) > 10:
            for i in range(self.gw.N_act):
                # Get trajectory
                traj = np.array([pos[i] for pos in self.gw.traj[-20:]])  # Last 20 steps
                
                # Filter out invalid positions
                valid_traj = np.all(np.isfinite(traj), axis=1)
                if not np.any(valid_traj):
                    continue
                    
                traj = traj[valid_traj]
                
                # Convert to Cartesian
                r_traj = np.exp(traj[:, 0])
                theta_traj = traj[:, 1]
                
                x_traj = r_traj * np.cos(theta_traj)
                y_traj = r_traj * np.sin(theta_traj)
                
                # Get color based on token type
                if i in set(event['token'] for event in self.gw.tachyonic_events):
                    color = 'red'
                    alpha = 0.8
                    width = 1.5
                elif i in set(event['token'] for event in self.levy_flight_events):
                    color = 'green'
                    alpha = 0.7
                    width = 1.2
                else:
                    color = 'white'
                    alpha = 0.5
                    width = 0.8
                
                # Plot trajectory
                ax2.plot(x_traj, y_traj, color=color, linewidth=width, alpha=alpha)
        
        # Axes settings
        ax2.set_xlabel('X', fontsize=12)
        ax2.set_ylabel('Y', fontsize=12)
        ax2.set_title('Loss Contours with Token Trajectories', fontsize=14)
        ax2.set_aspect('equal')
        
        # 3. Gradient Field
        # ===============
        # Calculate loss gradient
        dy, dx = np.gradient(loss)
        
        # Normalize gradient for better visualization
        gradient_mag = np.sqrt(dx**2 + dy**2)
        max_grad = np.max(gradient_mag)
        
        if max_grad > 0:
            dx = dx / max_grad
            dy = dy / max_grad
        
        # Plot gradient field as streamlines
        ax3.streamplot(x_grid, y_grid, -dx, -dy, color='white', 
                     density=1.5, linewidth=0.8, arrowsize=1.0)
        
        # Add the loss heatmap
        ax3.imshow(loss, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], 
                 origin='lower', cmap=GOLDEN_CMAP, alpha=0.7, aspect='auto')
        
        # Plot token positions
        ax3.scatter(x[valid_mask], y[valid_mask], 
                 c=[token_colors[i] for i in range(len(token_colors)) if valid_mask[i]], 
                 s=50, edgecolors='white', linewidth=0.5, zorder=10)
        
        # Mark Lévy flight events
        for event in self.levy_flight_events:
            pos_before = event['position_before']
            pos_after = event['position_after']
            
            # Convert to Cartesian
            r_before = np.exp(pos_before[0])
            theta_before = pos_before[1]
            x_before = r_before * np.cos(theta_before)
            y_before = r_before * np.sin(theta_before)
            
            r_after = np.exp(pos_after[0])
            theta_after = pos_after[1]
            x_after = r_after * np.cos(theta_after)
            y_after = r_after * np.sin(theta_after)
            
            # Plot the flight
            ax3.plot([x_before, x_after], [y_before, y_after], 
                   'g-', linewidth=1.5, alpha=0.7)
            
            # Add arrow for direction
            dx = x_after - x_before
            dy = y_after - y_before
            ax3.arrow(x_before, y_before, dx*0.8, dy*0.8, 
                    head_width=0.2, head_length=0.3, 
                    fc='green', ec='green', alpha=0.7)
        
        # Axes settings
        ax3.set_xlabel('X', fontsize=12)
        ax3.set_ylabel('Y', fontsize=12)
        ax3.set_title('Loss Gradient Field with Lévy Flights', fontsize=14)
        ax3.set_aspect('equal')
        
        # 4. Loss Profile Along Trajectories
        # ================================
        # Plot loss profile along trajectories for selected tokens
        if len(self.gw.traj) > 10:
            # Get a few interesting tokens
            tachyonic_tokens = list(set(event['token'] for event in self.gw.tachyonic_events))
            levy_tokens = list(set(event['token'] for event in self.levy_flight_events))
            
            # Select up to 5 interesting tokens
            selected_tokens = []
            selected_tokens.extend(tachyonic_tokens[:2])
            selected_tokens.extend([t for t in levy_tokens[:2] if t not in selected_tokens])
            
            # Add some regular tokens if needed
            remaining = 5 - len(selected_tokens)
            if remaining > 0 and self.gw.N_act > len(selected_tokens):
                regular_tokens = [i for i in range(self.gw.N_act) 
                                 if i not in selected_tokens]
                selected_tokens.extend(np.random.choice(
                    regular_tokens, 
                    size=min(remaining, len(regular_tokens)),
                    replace=False
                ))
            
            # Plot loss profiles
            for i, token_idx in enumerate(selected_tokens):
                # Get loss values along trajectory
                loss_values = []
                
                for t in range(len(self.gw.traj)):
                    pos = self.gw.traj[t][token_idx]
                    
                    # Skip invalid positions
                    if not np.all(np.isfinite(pos)):
                        loss_values.append(np.nan)
                        continue
                    
                    # Convert to Cartesian
                    r_t = np.exp(pos[0])
                    theta_t = pos[1]
                    
                    x_t = r_t * np.cos(theta_t)
                    y_t = r_t * np.sin(theta_t)
                    
                    # Find loss value
                    x_idx = np.argmin(np.abs(x_grid - x_t))
                    y_idx = np.argmin(np.abs(y_grid - y_t))
                    
                    if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                        loss_values.append(loss[y_idx, x_idx])
                    else:
                        loss_values.append(np.nan)
                
                # Plot profile
                time_steps = np.arange(len(loss_values))
                valid_steps = ~np.isnan(loss_values)
                
                if np.any(valid_steps):
                    # Get color based on token type
                    if token_idx in tachyonic_tokens:
                        color = 'red'
                        label = f'Tachyonic Token {token_idx}'
                    elif token_idx in levy_tokens:
                        color = 'green'
                        label = f'Lévy Flight Token {token_idx}'
                    else:
                        color = 'blue'
                        label = f'Regular Token {token_idx}'
                    
                    # Plot loss profile
                    ax4.plot(time_steps[valid_steps], np.array(loss_values)[valid_steps], 
                           color=color, linewidth=1.5, alpha=0.8, label=label)
                    
                    # Mark lévy flight events
                    for event in self.levy_flight_events:
                        if event['token'] == token_idx:
                            t_idx = event['time_step']
                            if t_idx < len(loss_values) and not np.isnan(loss_values[t_idx]):
                                ax4.scatter(t_idx, loss_values[t_idx], color='green', s=80, 
                                         marker='o', edgecolors='white', linewidth=1)
            
            # Add legend
            if selected_tokens:
                ax4.legend(loc='upper right')
        
        # Axes settings
        ax4.set_xlabel('Time Steps', fontsize=12)
        ax4.set_ylabel('Loss Value', fontsize=12)
        ax4.set_title('Loss Profiles Along Token Trajectories', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Global figure settings
        plt.tight_layout()
        fig.suptitle('Quantum Loss Landscape Navigation', fontsize=16)
        plt.subplots_adjust(top=0.92)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {os.path.join(self.output_dir, filename)}")
        
        return fig
    
    def visualize_dual_vortices(self, filename="dual_vortices.png"):
        """
        Visualize dual vortices in phase space with counterrotating fields.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Convert to Cartesian
        r = np.exp(self.gw.pos[:self.gw.N_act, 0])
        theta = self.gw.pos[:self.gw.N_act, 1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Filter out invalid positions
        valid_mask = np.isfinite(x) & np.isfinite(y)
        
        # Create a grid for the vortex field interpolation
        grid_size = 80
        x_grid = np.linspace(-5, 5, grid_size)
        y_grid = np.linspace(-5, 5, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Calculate polar coordinates for grid
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # Create two counter-rotating vortex centers
        vortex1_center = (PHI - 1, 0)  # First vortex at (φ-1, 0)
        vortex2_center = (-1/PHI, 0)   # Second vortex at (-1/φ, 0)
        
        # Distance from each grid point to vortex centers
        dist1 = np.sqrt((X - vortex1_center[0])**2 + (Y - vortex1_center[1])**2)
        dist2 = np.sqrt((X - vortex2_center[0])**2 + (Y - vortex2_center[1])**2)
        
        # Calculate vortex strength based on distance
        vortex1_strength = np.exp(-dist1 / (PHI/2))
        vortex2_strength = np.exp(-dist2 / (PHI/2))
        
        # Combine vortex fields (first clockwise, second counterclockwise)
        U = np.zeros_like(X)  # x-component of vector field
        V = np.zeros_like(Y)  # y-component of vector field
        
        # Generate the dual vortex field
        for i in range(grid_size):
            for j in range(grid_size):
                # Vector from point to vortex1 center
                dx1 = X[i,j] - vortex1_center[0]
                dy1 = Y[i,j] - vortex1_center[1]
                d1 = np.sqrt(dx1**2 + dy1**2) + 1e-10
                
                # Vector from point to vortex2 center
                dx2 = X[i,j] - vortex2_center[0]
                dy2 = Y[i,j] - vortex2_center[1]
                d2 = np.sqrt(dx2**2 + dy2**2) + 1e-10
                
                # Clockwise rotation for vortex1
                U[i,j] += vortex1_strength[i,j] * dy1 / d1
                V[i,j] += -vortex1_strength[i,j] * dx1 / d1
                
                # Counter-clockwise rotation for vortex2
                U[i,j] += -vortex2_strength[i,j] * dy2 / d2
                V[i,j] += vortex2_strength[i,j] * dx2 / d2
        
        # Normalize the vector field for better visualization
        magnitude = np.sqrt(U**2 + V**2)
        max_mag = np.max(magnitude)
        U_norm = U / max_mag
        V_norm = V / max_mag
        
        # Calculate vorticity (curl of the vector field)
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        dVdx = np.gradient(V, dx, axis=1)
        dUdy = np.gradient(U, dy, axis=0)
        vorticity = dVdx - dUdy
        
        # Create a custom colormap for vorticity
        vortex_colors = [(0.0, 0.0, 0.5), (0.0, 0.4, 0.7), (1.0, 1.0, 1.0), 
                         (0.7, 0.0, 0.0), (0.5, 0.0, 0.0)]
        vortex_cmap = LinearSegmentedColormap.from_list('vortex', vortex_colors)
        
        # Normalize vorticity for visualization
        vmax = np.max(np.abs(vorticity))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        # Plot vorticity as colored contour
        contour = ax.pcolormesh(X, Y, vorticity, cmap=vortex_cmap, 
                             norm=norm, alpha=0.7, shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Vorticity (Curl)', fontsize=12)
        
        # Add streamlines to show the flow
        ax.streamplot(x_grid, y_grid, U_norm, V_norm, color='white', 
                    density=1.5, linewidth=0.8, arrowsize=1.0)
        
        # Plot tokens
        token_colors = ['gold' if f else 'cyan' for f in self.gw.froz[:self.gw.N_act]]
        ax.scatter(x[valid_mask], y[valid_mask], 
                 c=[token_colors[i] for i in range(len(token_colors)) if valid_mask[i]], 
                 s=50, edgecolors='white', linewidth=0.5, zorder=10)
        
        # Mark vortex centers
        ax.scatter([vortex1_center[0]], [vortex1_center[1]], s=150, marker='*', 
                 color='gold', edgecolor='black', linewidth=1.5, zorder=11, 
                 label='Clockwise Vortex (φ-1)')
        ax.scatter([vortex2_center[0]], [vortex2_center[1]], s=150, marker='*', 
                 color='blue', edgecolor='black', linewidth=1.5, zorder=11, 
                 label='Counter-Clockwise Vortex (1/φ)')
        
        # Add preferred radii
        theta_circle = np.linspace(0, 2*np.pi, 100)
        
        # Critical radius for tachyonic behavior
        r_critical = self.gw.cfg.c * PHI**2 / np.pi
        x_crit = r_critical * np.cos(theta_circle)
        y_crit = r_critical * np.sin(theta_circle)
        ax.plot(x_crit, y_crit, 'red', alpha=0.7, linewidth=1.5, linestyle='--',
               label=f'Tachyonic Threshold (r={r_critical:.3f})')
        
        # Add annotations explaining the dual vortices
        ax.annotate('Clockwise\nVortex', xy=(vortex1_center[0], vortex1_center[1]),
                  xytext=(vortex1_center[0]+1, vortex1_center[1]+1),
                  arrowprops=dict(facecolor='black', shrink=0.05),
                  fontsize=12, ha='center')
        ax.annotate('Counter-\nClockwise\nVortex', xy=(vortex2_center[0], vortex2_center[1]),
                  xytext=(vortex2_center[0]-1, vortex2_center[1]-1),
                  arrowprops=dict(facecolor='black', shrink=0.05),
                  fontsize=12, ha='center')
        
        # Add the golden ratio relationship
        ax.annotate(f'Golden Ratio Relationship: \n1/φ ≈ {1/PHI:.3f}',
                  xy=(0, -4), fontsize=12, ha='center',
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))
        
        # Labels and title
        ax.set_xlabel('X = r·cos(θ)', fontsize=14)
        ax.set_ylabel('Y = r·sin(θ)', fontsize=14)
        ax.set_title('Dual Vortices in Phase Space', fontsize=16)
        ax.set_aspect('equal')
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(False)
        
        # Set axis limits for better focus on the vortices
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {os.path.join(self.output_dir, filename)}")
        
        return fig
        
    def visualize_phase_locked_evolution(self, steps=10, filename_prefix="phase_locked_evolution"):
        """
        Visualize loss field evolution with phase locking until sequence completion.
        Creates a series of frames showing how the loss field evolves.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.animation as animation
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Initialize grid for loss field
        grid_size = 80
        x_grid = np.linspace(-4, 4, grid_size)
        y_grid = np.linspace(-4, 4, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Convert to log-cylindrical
        R_grid = np.sqrt(X**2 + Y**2)
        Theta_grid = np.arctan2(Y, X)
        Ell_grid = np.log(R_grid + 1e-10)
        
        # Phase lock specific positions based on golden ratio
        phase_lock_positions = [
            (1/PHI, 0),               # Inner vortex center
            (PHI - 1, 0),             # Outer vortex center
            (PHI, PHI/2),             # Phi position with angular offset
            (PHI**2, -PHI/3),         # Phi^2 position with angular offset
            (1/PHI**2, 2*PHI/3)       # 1/Phi^2 position with angular offset
        ]
        
        # Convert to log-cylindrical
        phase_lock_ell_theta = []
        for pos in phase_lock_positions:
            r, theta = pos
            ell = np.log(r)
            phase_lock_ell_theta.append((ell, theta))
        
        # Array to store frames
        frames = []
        
        # Generate evolving loss field frames
        for t in range(steps):
            # Progress indicator
            print(f"Generating phase-locked frame {t+1}/{steps}")
            
            # Clear previous plot
            ax.clear()
            
            # Calculate loss field at this time step
            loss = np.zeros((grid_size, grid_size))
            
            # Phase locking factor (increases over time)
            lock_factor = t / (steps - 1)  # 0 to 1
            
            # Generate base loss field from token positions
            for i in range(self.gw.N_act):
                if not (np.isfinite(self.gw.pos[i, 0]) and np.isfinite(self.gw.pos[i, 1])):
                    continue
                    
                # Token parameters
                ell_i = self.gw.pos[i, 0]
                theta_i = self.gw.pos[i, 1]
                strength = self.gw.s[i]
                
                # Distance in log-cylindrical coordinates
                d_ell = Ell_grid - ell_i
                d_theta = np.minimum(
                    np.abs(Theta_grid - theta_i),
                    2*np.pi - np.abs(Theta_grid - theta_i)
                )
                
                # Combined squared distance
                d2 = d_ell**2 + d_theta**2
                
                # Repulsive potential (inverse of distance)
                potential = strength / (d2 + 0.1)
                
                # Add to loss landscape with diminishing influence
                loss += potential * (1 - 0.7 * lock_factor)
            
            # Add increasingly strong phase-locked points
            for ell, theta in phase_lock_ell_theta:
                # Convert to Cartesian for visualization
                r = np.exp(ell)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                # Distance in log-cylindrical coordinates
                d_ell = Ell_grid - ell
                d_theta = np.minimum(
                    np.abs(Theta_grid - theta),
                    2*np.pi - np.abs(Theta_grid - theta)
                )
                
                # Combined squared distance
                d2 = d_ell**2 + d_theta**2
                
                # Attractive phase-locked potential (gets stronger over time)
                potential = -5.0 * lock_factor * np.exp(-d2 / (0.2 * (1 - 0.5*lock_factor)))
                
                # Add to loss landscape
                loss += potential
            
            # Add phi^n attractors to the loss landscape
            for n in range(1, 6):
                r_phi_n = PHI**n
                ell_phi_n = np.log(r_phi_n)
                
                # Attractive potential at phi^n radius (strengthens over time)
                d_ell = np.abs(Ell_grid - ell_phi_n)
                potential = -0.8 * (0.5 + 0.5 * lock_factor) * np.exp(-d_ell**2 / 0.1)
                
                # Add to loss landscape
                loss += potential
            
            # Add dual vortex influence (strengthens over time)
            vortex1_center = (np.log(PHI - 1), 0)  # log-cylindrical
            vortex2_center = (np.log(1/PHI), 0)    # log-cylindrical
            
            # Distance to vortex centers
            d_vortex1 = np.sqrt((Ell_grid - vortex1_center[0])**2 + 
                              (Theta_grid - vortex1_center[1])**2)
            d_vortex2 = np.sqrt((Ell_grid - vortex2_center[0])**2 + 
                              (Theta_grid - vortex2_center[1])**2)
            
            # Vortex potentials
            vortex1_pot = -2.0 * lock_factor * np.exp(-d_vortex1 / (PHI/4))
            vortex2_pot = -2.0 * lock_factor * np.exp(-d_vortex2 / (PHI/4))
            
            # Add to loss landscape
            loss += vortex1_pot + vortex2_pot
            
            # Apply Gaussian blur for smoother landscape
            loss = gaussian_filter(loss, sigma=1.0)
            
            # Normalize for better visualization
            loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss) + 1e-10)
            
            # Plot loss field
            im = ax.pcolormesh(X, Y, loss, cmap=GOLDEN_CMAP, shading='auto', alpha=0.8)
            
            # Add contour lines
            contour = ax.contour(X, Y, loss, 15, colors='white', linewidths=0.5, alpha=0.7)
            
            # Calculate gradient for streamlines
            dy, dx = np.gradient(loss)
            
            # Normalize gradient for better visualization
            gradient_mag = np.sqrt(dx**2 + dy**2)
            max_grad = np.max(gradient_mag)
            
            if max_grad > 0:
                dx = dx / max_grad
                dy = dy / max_grad
            
            # Plot streamlines (following negative gradient - descent direction)
            ax.streamplot(x_grid, y_grid, -dx, -dy, color='white', 
                       density=1.0, linewidth=0.8, arrowsize=0.8)
            
            # Mark phase-locked positions
            for i, (ell, theta) in enumerate(phase_lock_ell_theta):
                r = np.exp(ell)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                # Size increases with lock_factor
                size = 50 + 100 * lock_factor
                
                # Color indicates lock progression
                alpha = 0.5 + 0.5 * lock_factor
                
                ax.scatter(x, y, s=size, color='gold', edgecolor='white', 
                        linewidth=1.5, alpha=alpha, zorder=10)
                
                # Add labels for the first and last frames
                if t == 0 or t == steps-1:
                    if i == 0:
                        label = "Inner Vortex (1/φ)"
                    elif i == 1:
                        label = "Outer Vortex (φ-1)"
                    else:
                        label = f"Phase Lock Point {i+1}"
                        
                    ax.annotate(label, xy=(x, y), xytext=(x + 0.5, y + 0.5),
                              arrowprops=dict(facecolor='white', shrink=0.05, alpha=0.7),
                              fontsize=10, color='white', ha='center')
            
            # Add preferred radii
            theta_circle = np.linspace(0, 2*np.pi, 100)
            
            # Add phi^n circles with increasing prominence
            for n in range(1, 6):
                r_phi_n = PHI**n
                x_phi = r_phi_n * np.cos(theta_circle)
                y_phi = r_phi_n * np.sin(theta_circle)
                
                # Line becomes more prominent with locking
                alpha = 0.3 + 0.4 * lock_factor
                linewidth = 0.5 + 1.0 * lock_factor
                
                ax.plot(x_phi, y_phi, 'cyan', alpha=alpha, linewidth=linewidth,
                     linestyle='--' if n % 2 == 0 else '-')
            
            # Add title with phase locking progress
            ax.set_title(f'Loss Field Evolution with Phase Locking: {int(lock_factor*100)}% Complete', 
                      fontsize=14)
            
            # Set axes properties
            ax.set_xlabel('X = r·cos(θ)', fontsize=12)
            ax.set_ylabel('Y = r·sin(θ)', fontsize=12)
            ax.set_aspect('equal')
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            
            # Add progress indicator
            ax.text(0.5, -3.5, f"Phase Lock Progress: {int(lock_factor*100)}%",
                 fontsize=12, ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", alpha=0.7),
                 color='white')
            
            # Save this frame
            filename = f"{filename_prefix}_{t+1:02d}.png"
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            
            # Store frame for potential animation
            frames.append(fig)
        
        print(f"Generated {steps} phase-locked evolution frames")
        return frames

    def visualize_all(self):
        """Generate all advanced visualizations."""
        print("Generating advanced visualizations...")
        
        # Wave mechanics visualization
        self.visualize_wave_mechanics()
        
        # Tachyonic helical visualization (if there are tachyonic events)
        if self.gw.tachyonic_events:
            self.visualize_tachyonic_helical()
        
        # Loss landscape visualization
        self.visualize_loss_landscape()
        
        # Dual vortices visualization
        self.visualize_dual_vortices()
        
        # Phase-locked evolution (5 frames to keep it manageable)
        self.visualize_phase_locked_evolution(steps=5)
        
        print("All advanced visualizations generated successfully!")

# Helper function to create the visualizations for a GwaveCore instance
def visualize_gwave(gwave_core: GwaveCore):
    """Generate advanced visualizations for a GwaveCore instance."""
    viz = GwaveAdvancedViz(gwave_core)
    viz.visualize_all()
    
    return viz

# If run as a script, visualize a provided GwaveCore dump file
if __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(
        description="Generate advanced visualizations for Gwave quantum field dynamics")
    parser.add_argument("--dump_file", type=str,
                        help="Path to a .npz file containing a GwaveCore dump")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if args.dump_file:
        # Load GwaveCore from dump file
        dump_data = np.load(args.dump_file, allow_pickle=True)
        
        # Recreate GwaveCore instance
        from gwave_core_fixed import GwaveConfig
        
        # Extract configuration
        cfg_dict = dump_data['cfg'].item()
        cfg = GwaveConfig(**cfg_dict)
        
        # Create GwaveCore instance
        gw = GwaveCore(cfg)
        
        # Load state from dump
        gw.pos[:len(dump_data['pos'])] = dump_data['pos']
        gw.mass[:len(dump_data['mass'])] = dump_data['mass']
        gw.H[:len(dump_data['H']), :len(dump_data['H'])] = dump_data['H']
        gw.froz[:len(dump_data['froz'])] = dump_data['froz']
        gw.N_act = len(dump_data['pos'])
        
        # Load additional data if available
        if 'vortex' in dump_data:
            gw.vortex_field[:len(dump_data['vortex'])] = dump_data['vortex']
        
        if 'tachyonic' in dump_data:
            gw.tachyonic_events = dump_data['tachyonic'].tolist()
        
        if 'phi_n_counts' in dump_data:
            gw.phi_n_counts = dump_data['phi_n_counts']
        
        # Create visualizations
        viz = GwaveAdvancedViz(gw)
        viz.visualize_all()
        
        print(f"Visualizations for {args.dump_file} saved to {OUTPUT_DIR}/")
    else:
        print("No dump file provided. Please specify a GwaveCore dump file with --dump_file.")
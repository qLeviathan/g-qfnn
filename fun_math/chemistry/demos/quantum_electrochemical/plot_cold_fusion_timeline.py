#!/usr/bin/env python3
"""
Visualization module for quantum cold fusion simulation

Provides slide-based timeline visualization of cold fusion processes
showing the evolution of quantum interactions during the reaction
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict, List, Optional

# Make sure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_cold_fusion_timeline(results, num_frames=6, save_path=None):
    """
    Create a slide-based timeline visualization of the cold fusion simulation
    
    Args:
        results: Results from cold fusion simulation
        num_frames: Number of frames to display in timeline
        save_path: Path to save the visualization (if None, display only)
    
    Returns:
        fig: Figure with timeline visualization
    """
    # Extract key data
    all_states = results["all_states"]
    all_fusion_probs = results["all_fusion_probs"]
    grid_size = results["grid_size"]
    grid_dim = len(grid_size)
    num_steps = all_states.shape[0]
    
    # Calculate frame indices for timeline
    frame_indices = np.linspace(0, num_steps-1, num_frames, dtype=int)
    
    # Create figure with timeline layout
    fig = plt.figure(figsize=(20, 16))
    
    # Set up grid of subplots - 5 rows (metrics) x num_frames columns
    gs = gridspec.GridSpec(5, num_frames, figure=fig, height_ratios=[1, 1, 1, 0.8, 0.5],
                          hspace=0.3, wspace=0.3)
    
    # Set up colormap ranges
    deuterium_norm = Normalize(vmin=0, vmax=np.max(all_states[:, :, :, 0]) * 1.2)
    electron_norm = Normalize(vmin=0, vmax=np.max(all_states[:, :, :, 1]) * 1.2)
    fusion_prob_norm = Normalize(vmin=0, vmax=1)
    
    # Title for the whole figure
    fig.suptitle("Quantum Cold Fusion Simulation Timeline", fontsize=20, y=0.98)
    
    # Plot each frame in the timeline
    for i, frame_idx in enumerate(frame_indices):
        # Extract data for this frame
        deuterium = all_states[frame_idx, 0, :, 0]
        electrons = all_states[frame_idx, 0, :, 1]
        palladium = all_states[frame_idx, 0, :, 2]
        fusion_prob = all_fusion_probs[frame_idx, 0, :]
        
        # Display time step information
        time_percentage = frame_idx / (num_steps - 1) * 100
        time_label = f"Time Step: {frame_idx} ({time_percentage:.1f}%)"
        
        # ROW 1: Deuterium concentration
        ax1 = fig.add_subplot(gs[0, i])
        if grid_dim == 1:
            # 1D plot
            x = np.linspace(0, 1, len(deuterium))
            ax1.plot(x, deuterium)
            ax1.set_ylim(0, np.max(all_states[:, :, :, 0]) * 1.2)
            ax1.set_ylabel("Concentration")
        else:
            # 2D heatmap
            deuterium_2d = deuterium.reshape(grid_size[:2])
            im1 = ax1.imshow(deuterium_2d, origin='lower', cmap='viridis', norm=deuterium_norm)
            if i == num_frames - 1:
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im1, cax=cax)
        
        # Add title for first row only
        if i == 0:
            ax1.set_title("Deuterium\nDistribution", fontsize=12)
        else:
            ax1.set_title(time_label, fontsize=10)
            
        # Remove axes for cleaner look
        if grid_dim > 1:
            ax1.set_xticks([])
            ax1.set_yticks([])
        
        # ROW 2: Electron density
        ax2 = fig.add_subplot(gs[1, i])
        if grid_dim == 1:
            # 1D plot
            ax2.plot(x, electrons)
            ax2.set_ylim(0, np.max(all_states[:, :, :, 1]) * 1.2)
            ax2.set_ylabel("Density")
        else:
            # 2D heatmap
            electrons_2d = electrons.reshape(grid_size[:2])
            im2 = ax2.imshow(electrons_2d, origin='lower', cmap='plasma', norm=electron_norm)
            if i == num_frames - 1:
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im2, cax=cax)
        
        # Add title for first column only
        if i == 0:
            ax2.set_title("Electron\nScreening", fontsize=12)
            
        # Remove axes for cleaner look
        if grid_dim > 1:
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        # ROW 3: Fusion probability
        ax3 = fig.add_subplot(gs[2, i])
        if grid_dim == 1:
            # 1D plot
            ax3.plot(x, fusion_prob)
            ax3.set_ylim(0, 1.0)
            ax3.set_ylabel("Probability")
        else:
            # 2D heatmap
            fusion_prob_2d = fusion_prob.reshape(grid_size[:2])
            im3 = ax3.imshow(fusion_prob_2d, origin='lower', cmap='hot', norm=fusion_prob_norm)
            if i == num_frames - 1:
                divider = make_axes_locatable(ax3)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im3, cax=cax)
        
        # Add title for first column only
        if i == 0:
            ax3.set_title("Fusion\nProbability", fontsize=12)
            
        # Remove axes for cleaner look
        if grid_dim > 1:
            ax3.set_xticks([])
            ax3.set_yticks([])
            
        # ROW 4: Key metrics at this time step
        ax4 = fig.add_subplot(gs[3, i])
        ax4.axis('off')  # No axes needed
        
        # Calculate metrics for this frame
        if frame_idx < len(results["tunneling_history"]):
            tunnel_prob = results["tunneling_history"][frame_idx]
            screening = results["screening_history"][frame_idx]
            if frame_idx < len(results["fusion_events"]):
                fusion_count = sum(results["fusion_events"][:frame_idx+1])
            else:
                fusion_count = 0
            
            # Create formatted text for metrics
            metrics_text = (
                f"Tunneling: {tunnel_prob:.3f}\n"
                f"Screening: {screening:.3f}\n"
                f"Fusion Events: {fusion_count}"
            )
            ax4.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # ROW 5: Shared timeline metrics across all frames
    ax_timeline = fig.add_subplot(gs[4, :])
    
    # Get time-series data
    energy_history = np.array(results["fusion_energy"])
    tunneling_history = np.array(results["tunneling_history"])
    screening_history = np.array(results["screening_history"])
    coherence_history = np.array(results["phase_coherence"])
    
    # Create consistent time steps - use the shortest series length as reference
    min_length = min(len(tunneling_history), len(screening_history), len(coherence_history))
    time_steps = np.arange(min_length)
    
    # Ensure all arrays have the same length by truncating to min_length
    tunneling_history = tunneling_history[:min_length] if len(tunneling_history) > min_length else tunneling_history
    screening_history = screening_history[:min_length] if len(screening_history) > min_length else screening_history
    coherence_history = coherence_history[:min_length] if len(coherence_history) > min_length else coherence_history
    
    # Normalize all data to 0-1 range for comparable plotting
    ax_timeline.plot(time_steps, tunneling_history, 'r-', label='Tunneling')
    ax_timeline.plot(time_steps, screening_history, 'g-', label='Screening')
    ax_timeline.plot(time_steps, coherence_history, 'b-', label='Coherence')
    
    # Add markers for the shown frames
    for i, frame_idx in enumerate(frame_indices):
        if frame_idx < len(time_steps):
            ax_timeline.axvline(x=frame_idx, color='k', linestyle='--', alpha=0.5)
            
    ax_timeline.set_xlabel("Simulation Time Steps")
    ax_timeline.set_ylabel("Normalized Metrics")
    ax_timeline.legend(loc='upper right', ncol=4)
    ax_timeline.grid(True, alpha=0.3)
    
    # Add a second y-axis for fusion energy
    if len(energy_history) > 0:
        ax_energy = ax_timeline.twinx()
        ax_energy.plot(np.arange(len(energy_history)), energy_history, 'c-', 
                      linewidth=2, label='Fusion Energy')
        ax_energy.set_ylabel("Fusion Energy (MeV)", color='c')
        ax_energy.tick_params(axis='y', colors='c')
        
        # Add to legend
        lines, labels = ax_timeline.get_legend_handles_labels()
        lines2, labels2 = ax_energy.get_legend_handles_labels()
        ax_timeline.legend(lines + lines2, labels + labels2, loc='upper left', ncol=4)
    
    # Adjust layout and save/display
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timeline visualization saved to {save_path}")
    
    return fig


def plot_cold_fusion_3d(results, frame_idx=-1, save_path=None):
    """
    Create a 3D visualization of cold fusion activity for a specific time frame
    
    Args:
        results: Results from cold fusion simulation
        frame_idx: Time step to visualize (-1 for final state)
        save_path: Path to save the visualization (if None, display only)
    
    Returns:
        fig: 3D figure
    """
    # Extract key data
    all_states = results["all_states"]
    grid_positions = results["grid_positions"][0]  # First batch
    fusion_probs = results["all_fusion_probs"]
    
    # Select time frame
    if frame_idx < 0:
        frame_idx = all_states.shape[0] - 1
    
    # Extract data for this frame
    deuterium = all_states[frame_idx, 0, :, 0]
    electrons = all_states[frame_idx, 0, :, 1]
    fusion_prob = fusion_probs[frame_idx, 0, :]
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    if grid_positions.shape[1] == 1:
        # 1D
        x = grid_positions[:, 0]
        y = np.zeros_like(x)
        z = np.zeros_like(x)
    elif grid_positions.shape[1] == 2:
        # 2D
        x = grid_positions[:, 0]
        y = grid_positions[:, 1]
        z = np.zeros_like(x)
    else:
        # 3D
        x = grid_positions[:, 0]
        y = grid_positions[:, 1]
        z = grid_positions[:, 2]
    
    # Create fusion probability colormap
    colormap = cm.hot
    colors = colormap(fusion_prob)
    
    # Scale point sizes by deuterium concentration
    sizes = 10 + 40 * deuterium / np.max(deuterium)
    
    # Draw 3D scatter plot
    scatter = ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Fusion Probability')
    
    # Add fusion event locations if available
    if len(results["fusion_locations"]) > 0 and len(results["fusion_timesteps"]) > 0:
        # Filter fusion events that happened up to this frame
        relevant_events = [loc for loc, ts in zip(results["fusion_locations"], 
                                                results["fusion_timesteps"]) 
                          if ts <= frame_idx]
        
        if relevant_events:
            fusion_locs = np.array(relevant_events)
            if fusion_locs.shape[1] == 1:
                f_x = fusion_locs[:, 0]
                f_y = np.zeros_like(f_x)
                f_z = np.zeros_like(f_x)
            elif fusion_locs.shape[1] == 2:
                f_x = fusion_locs[:, 0]
                f_y = fusion_locs[:, 1]
                f_z = np.zeros_like(f_x)
            else:
                f_x = fusion_locs[:, 0]
                f_y = fusion_locs[:, 1]
                f_z = fusion_locs[:, 2]
            
            # Draw fusion events as stars
            ax.scatter(f_x, f_y, f_z, marker='*', s=200, c='yellow', 
                      edgecolor='k', linewidth=1, label='Fusion Events')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Visualization of Cold Fusion Simulation (Time Step {frame_idx})')
    
    # Add legend
    ax.legend()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D visualization saved to {save_path}")
    
    return fig

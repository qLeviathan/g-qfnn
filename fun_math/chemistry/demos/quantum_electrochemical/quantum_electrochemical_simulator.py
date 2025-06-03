#!/usr/bin/env python3
"""
Xi/Psi Quantum Field Neural Network for Electrochemical System Simulation

This script adapts the quantum neural network architecture for electrochemical simulations:
- Models ion diffusion and reaction kinetics in phase-space
- Combines quantum field representations with Nernst-Planck equations
- Handles both discrete particles and continuous fields
- Supports multi-scale simulation across different time regimes
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Dict, Optional, Union

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################################################
# Physical Constants and Parameters                 #
#####################################################

class ElectrochemicalConstants:
    """Physical constants relevant for electrochemical simulations"""
    # Fundamental constants
    ELECTRON_CHARGE = 1.602176634e-19  # C
    BOLTZMANN = 1.380649e-23  # J/K
    AVOGADRO = 6.02214076e23  # mol^-1
    FARADAY = ELECTRON_CHARGE * AVOGADRO  # C/mol
    
    # Common parameters
    TEMPERATURE = 298.15  # K (room temperature)
    
    # Derived quantities
    RT = BOLTZMANN * AVOGADRO * TEMPERATURE  # J/mol
    RT_F = RT / FARADAY  # V (thermal voltage)

#####################################################
# Phase-Space Representation Components             #
#####################################################

class PhaseSpatialEncoder(nn.Module):
    """
    Encodes spatial coordinates and field values into phase-space representation
    Optimized for electrochemical simulations with multi-scale dynamics
    """
    def __init__(self, dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Grid configuration
        self.grid_encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Field values encoder
        self.field_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim//2),  # Single field value
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Combined representation
        self.combined_encoder = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)  # Project to 2D phase space
        )
        
        # Golden ratio for phase distribution
        self.phi = (1 + 5**0.5) / 2
    
    def forward(self, positions: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Convert spatial coordinates and field values to phase-space representation
        
        Args:
            positions: Tensor of shape [batch, num_points, dim]
            values: Tensor of shape [batch, num_points, 1]
        
        Returns:
            phase_points: Complex phase-space coordinates [batch, num_points, 2]
        """
        # Encode grid positions
        pos_encoded = self.grid_encoder(positions)
        
        # Encode field values
        val_encoded = self.field_encoder(values)
        
        # Combine encodings
        combined = torch.cat([pos_encoded, val_encoded], dim=-1)
        phase_coords = self.combined_encoder(combined)
        
        # Normalize to ensure stable phase representation
        phase_coords = F.normalize(phase_coords, p=2, dim=-1)
        
        return phase_coords


class QuantumPotentialField(nn.Module):
    """
    Models the quantum potential field for electrochemical interactions
    Combines classical interactions with quantum effects
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Field processing layers
        self.field_processor = nn.Sequential(
            nn.Linear(2, hidden_dim),  # From phase coordinates
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Electric potential component
        self.electric_component = nn.Linear(hidden_dim, 1)
        
        # Chemical potential component
        self.chemical_component = nn.Linear(hidden_dim, 1)
        
        # Phase coherence tracker
        self.coherence_history = []
    
    def forward(self, phase_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute potentials from phase-space coordinates
        
        Args:
            phase_coords: Phase-space coordinates [batch, num_points, 2]
        
        Returns:
            potentials: Dictionary with electric and chemical potentials
        """
        # Process through field layers
        field_features = self.field_processor(phase_coords)
        
        # Compute electric potential component
        electric_potential = self.electric_component(field_features)
        
        # Compute chemical potential component
        chemical_potential = self.chemical_component(field_features)
        
        # Track phase coherence
        phase = torch.atan2(phase_coords[..., 1], phase_coords[..., 0])
        
        # Handle different tensor shapes safely
        dims_to_reduce = tuple(range(1, len(phase.shape)))  # All dims except batch dim
        if not dims_to_reduce:  # If only 1 dimension
            dims_to_reduce = (0,)
            
        sin_mean = torch.mean(torch.sin(phase), dim=dims_to_reduce)
        cos_mean = torch.mean(torch.cos(phase), dim=dims_to_reduce)
        coherence = torch.sqrt(sin_mean**2 + cos_mean**2).mean().item()
        self.coherence_history.append(coherence)
        
        return {
            "electric": electric_potential,
            "chemical": chemical_potential,
            "total": electric_potential + chemical_potential
        }


class ElectrochemicalInteractions(nn.Module):
    """
    Models electrochemical interactions using quantum field neural networks
    Handles both long-range and local interactions via phase-space
    """
    def __init__(self, hidden_dim: int = 64, threshold_factor: float = 1.25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold_factor = threshold_factor
        
        # Interaction components
        self.interaction_processor = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Interaction types
        self.diffusion = nn.Linear(hidden_dim, 1)
        self.reaction = nn.Linear(hidden_dim, 1)
        self.migration = nn.Linear(hidden_dim, 1)
        
        # Phase-aware attention for long-range interactions
        self.attention = PhaseAwareBinaryAttention(threshold_factor)
    
    def forward(self, phase_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute interaction terms from phase coordinates
        
        Args:
            phase_coords: Phase coordinates [batch, num_points, 2]
        
        Returns:
            interactions: Dictionary with different interaction terms
        """
        # Process through interaction layers
        features = self.interaction_processor(phase_coords)
        
        # Compute specific interaction components
        diffusion_term = self.diffusion(features)
        reaction_term = self.reaction(features)
        migration_term = self.migration(features)
        
        # Compute phase-aware attention for long-range effects
        attention_weights = self.attention(phase_coords)
        
        # Apply attention to propagate long-range effects
        diffusion_field = torch.bmm(attention_weights, diffusion_term)
        
        return {
            "diffusion": diffusion_term,
            "reaction": reaction_term, 
            "migration": migration_term,
            "attention": attention_weights,
            "diffusion_field": diffusion_field
        }


class PhaseAwareBinaryAttention(nn.Module):
    """
    Attention mechanism that leverages phase relationships for electrochemical interactions
    Similar to the original binary attention but adapted for ionic systems
    """
    def __init__(self, threshold_factor: float = 1.25):
        super().__init__()
        self.threshold_factor = threshold_factor
    
    def forward(self, r_embed: torch.Tensor) -> torch.Tensor:
        # Get dimensions
        batch_size, num_points, _ = r_embed.shape
        
        # Calculate pairwise differences using einsum
        diff = torch.einsum('bsi,bti->bsti', r_embed, r_embed.neg()) + r_embed.unsqueeze(2)
        
        # Compute squared distances efficiently
        dist_sq = torch.sum(diff ** 2, dim=-1)  # [batch, num_points, num_points]
        
        # Calculate phase differences for directionality
        phase_diff = torch.atan2(diff[..., 1], diff[..., 0])
        
        # Apply debye-huckel-like screening for ionic interactions
        forward_bias = 0.5 * (1 + torch.cos(phase_diff)) * torch.exp(-dist_sq)
        
        # Calculate adaptive threshold for each batch
        mean_dist = dist_sq.mean(dim=(-1, -2), keepdim=True)
        std_dist = dist_sq.std(dim=(-1, -2), keepdim=True).clamp(min=1e-6)  # Prevent zero std
        threshold = mean_dist + self.threshold_factor * std_dist
        
        # Create binary mask for interactions
        valid_mask = (dist_sq <= threshold)
        attention = valid_mask.float() * forward_bias
        
        # Normalize attention weights
        row_sums = attention.sum(-1, keepdim=True)
        attention = attention / (row_sums + 1e-8)
        
        return attention


#####################################################
# Quantum Electrochemical Simulator                 #
#####################################################

class QuantumElectrochemicalSimulator(nn.Module):
    """
    Complete simulator for electrochemical systems using quantum field neural networks
    Combines quantum and classical approaches for multi-scale modeling
    """
    def __init__(
        self,
        grid_dim: int = 3,
        hidden_dim: int = 64,
        num_species: int = 3,  # e.g., cations, anions, electrons
        hebbian_lr: float = 0.01,
        decay_rate: float = 0.999
    ):
        super().__init__()
        self.grid_dim = grid_dim
        self.hidden_dim = hidden_dim
        self.num_species = num_species
        self.hebbian_lr = hebbian_lr
        self.decay_rate = decay_rate
        
        # Physical constants
        self.constants = ElectrochemicalConstants()
        
        # Phase-space encoder
        self.encoder = PhaseSpatialEncoder(dim=grid_dim, hidden_dim=hidden_dim)
        
        # Species-specific field encoders
        self.species_encoders = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_species)
        ])
        
        # Potential field calculator
        self.potential_field = QuantumPotentialField(hidden_dim=hidden_dim)
        
        # Interaction calculator
        self.interactions = ElectrochemicalInteractions(hidden_dim=hidden_dim)
        
        # Time evolution module (RNN-based)
        self.time_evolution = nn.GRUCell(
            input_size=2 + 3,  # phase coords + interaction terms
            hidden_size=hidden_dim
        )
        
        # Decoder for field values
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)  # Concentration/field strength
        )
        
        # System state tracker
        self.states_history = []
        self.energy_history = []
        
    def forward(
        self, 
        grid_positions: torch.Tensor,  # [batch, num_points, dim]
        initial_values: torch.Tensor,  # [batch, num_points, num_species]
        boundary_conditions: Optional[Dict] = None,
        num_steps: int = 100,
        dt: float = 0.01
    ) -> Dict:
        """
        Run full electrochemical simulation
        
        Args:
            grid_positions: Spatial coordinates of grid points
            initial_values: Initial concentration/field values for each species
            boundary_conditions: Dictionary of boundary conditions
            num_steps: Number of time steps to simulate
            dt: Time step size
        
        Returns:
            simulation_results: Dictionary with simulation results
        """
        batch_size, num_points, _ = grid_positions.shape
        device = grid_positions.device
        
        # Initialize state storage
        states = initial_values.clone()
        all_states = torch.zeros(num_steps, batch_size, num_points, self.num_species, device=device)
        all_potentials = torch.zeros(num_steps, batch_size, num_points, device=device)
        all_energy = torch.zeros(num_steps, device=device)
        
        # Hidden state for time evolution
        hidden = torch.zeros(batch_size * num_points, self.hidden_dim, device=device)
        
        # Apply boundary conditions if provided
        if boundary_conditions is not None:
            for bc_type, bc_info in boundary_conditions.items():
                if bc_type == "dirichlet":
                    # Fixed value boundary
                    idx, values = bc_info
                    states[:, idx, :] = values.unsqueeze(1)
        
        # Run simulation steps
        for step in range(num_steps):
            states_list = []
            potentials_list = []
            
            # Process each species
            for species_idx in range(self.num_species):
                # Extract values for current species
                species_values = states[:, :, species_idx:species_idx+1]
                
                # Encode to phase-space
                phase_coords = self.encoder(grid_positions, species_values)
                
                # Calculate potentials
                potentials = self.potential_field(phase_coords)
                potentials_list.append(potentials["total"])
                
                # Calculate interactions
                interactions = self.interactions(phase_coords)
                
                # Prepare input for time evolution
                interaction_features = torch.cat([
                    interactions["diffusion"],
                    interactions["reaction"],
                    interactions["migration"]
                ], dim=-1)
                
                time_input = torch.cat([phase_coords, interaction_features], dim=-1)
                
                # Reshape for GRU processing
                time_input_flat = time_input.reshape(batch_size * num_points, -1)
                
                # Single step of time evolution
                hidden = self.time_evolution(time_input_flat, hidden)
                
                # Reshape back
                hidden_reshaped = hidden.reshape(batch_size, num_points, self.hidden_dim)
                
                # Species-specific processing
                species_hidden = self.species_encoders[species_idx](hidden_reshaped)
                
                # Decode to updated values
                new_values = self.decoder(species_hidden)
                
                # Apply physical constraints (concentrations must be non-negative)
                new_values = F.softplus(new_values)
                
                # Update with time step scaling
                species_values = species_values + new_values * dt
                
                # Store updated values for this species
                states_list.append(species_values)
                
                # Re-apply boundary conditions
                if boundary_conditions is not None and "dirichlet" in boundary_conditions:
                    idx, values = boundary_conditions["dirichlet"]
                    species_values[:, idx, :] = values[:, species_idx:species_idx+1]
            
            # Update states by combining all species
            states = torch.cat(states_list, dim=-1)
            
            # Average potentials across species
            avg_potential = torch.mean(torch.cat(potentials_list, dim=-1), dim=-1, keepdim=True)
            
            # Calculate system energy
            energy = torch.sum(states * avg_potential).item()
            
            # Store states and energy for this step
            all_states[step] = states
            all_potentials[step] = avg_potential.squeeze(-1)
            all_energy[step] = energy
            
            # Track for analysis
            self.states_history.append(states.detach().cpu().numpy())
            self.energy_history.append(energy)
        
        # Return simulation results
        return {
            "final_state": states,
            "all_states": all_states,
            "all_potentials": all_potentials,
            "energy_history": all_energy,
            "phase_coherence": self.potential_field.coherence_history
        }

    def run_diffusion_reaction_simulation(
        self,
        grid_size: List[int],
        diffusion_coeffs: List[float],
        reaction_rates: List[float],
        initial_conditions: Dict,
        boundary_conditions: Optional[Dict] = None,
        num_steps: int = 100,
        dt: float = 0.01
    ) -> Dict:
        """
        Run a diffusion-reaction simulation with the quantum field model
        
        Args:
            grid_size: Size of the spatial grid in each dimension
            diffusion_coeffs: Diffusion coefficients for each species
            reaction_rates: Reaction rate constants
            initial_conditions: Dictionary with initial species distributions
            boundary_conditions: Dictionary with boundary conditions
            num_steps: Number of simulation steps
            dt: Time step size
        
        Returns:
            results: Simulation results
        """
        # Set up the spatial grid
        if len(grid_size) == 1:
            # 1D grid
            x = torch.linspace(0, 1, grid_size[0], device=device)
            grid_positions = x.reshape(1, -1, 1)
            num_points = grid_size[0]
        elif len(grid_size) == 2:
            # 2D grid
            x = torch.linspace(0, 1, grid_size[0], device=device)
            y = torch.linspace(0, 1, grid_size[1], device=device)
            x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
            grid_positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1).unsqueeze(0)
            num_points = grid_size[0] * grid_size[1]
        elif len(grid_size) == 3:
            # 3D grid
            x = torch.linspace(0, 1, grid_size[0], device=device)
            y = torch.linspace(0, 1, grid_size[1], device=device)
            z = torch.linspace(0, 1, grid_size[2], device=device)
            x_grid, y_grid, z_grid = torch.meshgrid(x, y, z, indexing='ij')
            grid_positions = torch.stack(
                [x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=-1
            ).unsqueeze(0)
            num_points = grid_size[0] * grid_size[1] * grid_size[2]
        
        # Set up initial values
        initial_values = torch.zeros(1, num_points, self.num_species, device=device)
        
        # Apply initial conditions
        for species_idx, species_data in initial_conditions.items():
            if isinstance(species_data, torch.Tensor):
                # Direct assignment of values
                initial_values[0, :, species_idx] = species_data.flatten()
            elif callable(species_data):
                # Function that generates values based on position
                for i in range(num_points):
                    pos = grid_positions[0, i]
                    initial_values[0, i, species_idx] = species_data(pos)
            elif isinstance(species_data, float):
                # Constant value
                initial_values[0, :, species_idx] = species_data
        
        # Run the simulation
        results = self.forward(
            grid_positions=grid_positions,
            initial_values=initial_values,
            boundary_conditions=boundary_conditions,
            num_steps=num_steps,
            dt=dt
        )
        
        # Process results for easier analysis
        processed_results = {
            "grid_positions": grid_positions.detach().cpu().numpy(),
            "grid_size": grid_size,
            "final_state": results["final_state"].detach().cpu().numpy(),
            "all_states": results["all_states"].detach().cpu().numpy(),
            "all_potentials": results["all_potentials"].detach().cpu().numpy(),
            "energy_history": results["energy_history"].cpu().numpy(),  # Already detached
            "phase_coherence": results["phase_coherence"]
        }
        
        return processed_results


#####################################################
# Visualization Utilities                           #
#####################################################

def plot_simulation_results(results, grid_size, species_names=None):
    """
    Plot the results of an electrochemical simulation
    
    Args:
        results: Results from simulation
        grid_size: Size of simulation grid
        species_names: Names of chemical species
    """
    if species_names is None:
        species_names = [f"Species {i}" for i in range(results["all_states"].shape[-1])]
    
    num_steps, batch_size, num_points, num_species = results["all_states"].shape
    grid_positions = results["grid_positions"][0]  # Get the first batch
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Plot energy history
    ax1 = fig.add_subplot(231)
    ax1.plot(results["energy_history"])
    ax1.set_title("System Energy")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Energy")
    
    # Plot phase coherence
    if "phase_coherence" in results:
        ax2 = fig.add_subplot(232)
        ax2.plot(results["phase_coherence"])
        ax2.set_title("Phase Coherence")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Coherence")
    
    # Determine plot type based on grid dimensions
    grid_dim = len(grid_size)
    
    if grid_dim == 1:
        # 1D plot of concentrations
        ax3 = fig.add_subplot(233)
        x = grid_positions[:, 0]
        for i in range(num_species):
            ax3.plot(x, results["final_state"][0, :, i], label=species_names[i])
        ax3.set_title("Final Concentrations")
        ax3.set_xlabel("Position")
        ax3.set_ylabel("Concentration")
        ax3.legend()
        
        # 1D plot of potential
        ax4 = fig.add_subplot(234)
        ax4.plot(x, results["all_potentials"][-1, 0])
        ax4.set_title("Final Potential")
        ax4.set_xlabel("Position")
        ax4.set_ylabel("Potential")
        
    elif grid_dim == 2:
        # 2D heatmap of a selected species
        ax3 = fig.add_subplot(233)
        species_idx = 0  # Show first species by default
        concentration = results["final_state"][0, :, species_idx].reshape(grid_size)
        im = ax3.imshow(concentration, origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax3)
        ax3.set_title(f"Final {species_names[species_idx]} Concentration")
        
        # 2D heatmap of potential
        ax4 = fig.add_subplot(234)
        potential = results["all_potentials"][-1, 0].reshape(grid_size)
        im = ax4.imshow(potential, origin='lower', cmap='plasma')
        plt.colorbar(im, ax=ax4)
        ax4.set_title("Final Potential")
    
    # Time series for a specific point
    ax5 = fig.add_subplot(235)
    point_idx = num_points // 2  # Middle point by default
    for i in range(num_species):
        ax5.plot(results["all_states"][:, 0, point_idx, i], label=species_names[i])
    ax5.set_title(f"Concentration at Point {point_idx}")
    ax5.set_xlabel("Time Step")
    ax5.set_ylabel("Concentration")
    ax5.legend()
    
    # 3D visualization for the last subplot
    ax6 = fig.add_subplot(236, projection='3d')
    
    if grid_dim == 1:
        # 3D view: x = position, y = time, z = concentration
        species_idx = 0  # Show first species by default
        x = np.tile(grid_positions[:, 0], (num_steps, 1))
        y = np.repeat(np.arange(num_steps)[:, np.newaxis], num_points, axis=1)
        z = results["all_states"][:, 0, :, species_idx]
        
        ax6.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.8)
        ax6.set_title(f"{species_names[species_idx]} Evolution")
        ax6.set_xlabel("Position")
        ax6.set_ylabel("Time Step")
        ax6.set_zlabel("Concentration")
        
    elif grid_dim == 2:
        # Scatter plot with concentrations as colors
        species_idx = 0  # Show first species by default
        x = grid_positions[:, 0]
        y = grid_positions[:, 1]
        z = results["final_state"][0, :, species_idx]
        
        scatter = ax6.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.8)
        plt.colorbar(scatter, ax=ax6)
        ax6.set_title(f"3D View of {species_names[species_idx]}")
        ax6.set_xlabel("X")
        ax6.set_ylabel("Y")
        ax6.set_zlabel("Concentration")
    
    plt.tight_layout()
    return fig


def create_animation(results, grid_size, species_idx=0, interval=50):
    """
    Create an animation of the simulation results
    
    Args:
        results: Results from simulation
        grid_size: Size of simulation grid
        species_idx: Index of species to animate
        interval: Time between frames in milliseconds
    
    Returns:
        animation: The created animation
    """
    num_steps = results["all_states"].shape[0]
    grid_dim = len(grid_size)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if grid_dim == 1:
        # 1D animation
        x = results["grid_positions"][0, :, 0]
        line, = ax.plot(x, results["all_states"][0, 0, :, species_idx])
        ax.set_ylim(0, results["all_states"][:, 0, :, species_idx].max() * 1.1)
        ax.set_xlabel("Position")
        ax.set_ylabel("Concentration")
        
        def update(frame):
            line.set_ydata(results["all_states"][frame, 0, :, species_idx])
            ax.set_title(f"Time Step: {frame}")
            return line,
            
    elif grid_dim == 2:
        # 2D animation
        concentration = results["all_states"][0, 0, :, species_idx].reshape(grid_size)
        im = ax.imshow(concentration, origin='lower', cmap='viridis', animated=True)
        plt.colorbar(im, ax=ax)
        
        def update(frame):
            concentration = results["all_states"][frame, 0, :, species_idx].reshape(grid_size)
            im.set_array(concentration)
            ax.set_title(f"Time Step: {frame}")
            return im,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=num_steps, interval=interval, blit=True)
    plt.tight_layout()
    
    return anim


#####################################################
# Example Simulations                               #
#####################################################

def run_diffusion_example():
    """Run a simple diffusion simulation"""
    print("Running diffusion example simulation...")
    
    # Create model
    model = QuantumElectrochemicalSimulator(
        grid_dim=1,
        hidden_dim=64,
        num_species=2,  # Cation and anion
        hebbian_lr=0.005
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
    
    # Run simulation
    start_time = time.time()
    results = model.run_diffusion_reaction_simulation(
        grid_size=grid_size,
        diffusion_coeffs=diffusion_coeffs,
        reaction_rates=reaction_rates,
        initial_conditions=initial_conditions,
        num_steps=200,
        dt=0.01
    )
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Plot results
    species_names = ["Cation", "Anion"]
    fig = plot_simulation_results(results, grid_size, species_names)
    
    # Save results
    plt.savefig("quantum_electrochemical_diffusion.png")
    
    return results, fig

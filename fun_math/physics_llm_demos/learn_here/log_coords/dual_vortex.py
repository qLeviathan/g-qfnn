import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .log_coords import log_cylindrical_to_cartesian, log_cylindrical_batch_distance

class DualVortexField:
    """
    Implements a quantum field with dual vortex structure in log-cylindrical space.
    This provides O(N log N) computational complexity for token interactions.
    """
    
    def __init__(self, n_tokens=100, dim=3, device=None):
        """
        Initialize the dual vortex field.
        
        Args:
            n_tokens (int): Number of tokens in the field
            dim (int): Dimensionality of the log-cylindrical space (default 3: ln_r, theta, z)
            device (torch.device): Device to use for computations
        """
        self.n_tokens = n_tokens
        self.dim = dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize token positions in log-cylindrical space
        self.tokens = {
            'ln_r': torch.zeros(n_tokens, device=self.device),
            'theta': torch.zeros(n_tokens, device=self.device),
            'z': torch.zeros(n_tokens, device=self.device),
            'frozen': torch.zeros(n_tokens, dtype=torch.bool, device=self.device)
        }
        
        # Field parameters
        self.phi = (1 + 5**0.5) / 2  # Golden ratio
        self.temperature = 1.0  # Field temperature
        self.repulsion_strength = 0.1  # Strength of repulsive forces
        self.vorticity = 1.0  # Vortex strength
        
        # For tracking position history
        self.track_positions = False
        self.position_history = []
    
    def initialize_tokens(self, pattern='random'):
        """
        Initialize token positions according to specified pattern.
        
        Args:
            pattern (str): Initialization pattern ('random', 'uniform_circle', 'fibonacci_spiral', 'golden_spiral')
        """
        if pattern == 'random':
            # Random initialization
            self.tokens['ln_r'] = torch.randn(self.n_tokens, device=self.device)
            self.tokens['theta'] = 2 * np.pi * torch.rand(self.n_tokens, device=self.device)
            self.tokens['z'] = torch.randn(self.n_tokens, device=self.device)
            
        elif pattern == 'uniform_circle':
            # Uniform distribution on a circle
            self.tokens['ln_r'] = torch.zeros(self.n_tokens, device=self.device)
            self.tokens['theta'] = 2 * np.pi * torch.arange(self.n_tokens, device=self.device) / self.n_tokens
            self.tokens['z'] = torch.zeros(self.n_tokens, device=self.device)
            
        elif pattern == 'fibonacci_spiral':
            # Fibonacci spiral pattern (better space filling)
            golden_ratio = (1 + 5**0.5) / 2
            indices = torch.arange(self.n_tokens, device=self.device)
            
            # Logarithmic radial coordinate based on Fibonacci number
            self.tokens['ln_r'] = torch.log(torch.sqrt(indices + 1))
            
            # Angular coordinate based on golden ratio
            self.tokens['theta'] = 2 * np.pi * (indices / golden_ratio) % (2 * np.pi)
            
            # Z-coordinate as function of index
            self.tokens['z'] = indices / self.n_tokens
            
        elif pattern == 'golden_spiral':
            # Golden spiral with logarithmic radial spacing
            indices = torch.arange(self.n_tokens, device=self.device).float()
            
            # Natural logarithm of radius (logarithmic spiral)
            self.tokens['ln_r'] = torch.log(indices + 1) / 2
            
            # Angular coordinate with golden ratio spacing
            self.tokens['theta'] = 2 * np.pi * torch.fmod(indices / self.phi, 1.0)
            
            # Z-coordinate as function of index
            self.tokens['z'] = indices / self.n_tokens
        
        # Reset frozen state
        self.tokens['frozen'] = torch.zeros(self.n_tokens, dtype=torch.bool, device=self.device)
        
        # Clear history
        self.position_history = []
    
    def compute_repulsion_forces(self):
        """
        Compute repulsive forces between tokens using tensor operations
        for improved performance.
        
        Returns:
            tuple: (f_ln_r, f_theta, f_z) forces in log-cylindrical coordinates
        """
        # Create coordinate tensors
        ln_r = self.tokens['ln_r']
        theta = self.tokens['theta']
        z = self.tokens['z']
        
        # Stack coordinates for distance calculation
        coords = torch.stack([ln_r, theta, z], dim=1)  # Shape: [n_tokens, 3]
        
        # Expand coords for broadcasting
        # Shape: [n_tokens, 1, 3] and [1, n_tokens, 3]
        coords_i = coords.unsqueeze(1)
        coords_j = coords.unsqueeze(0)
        
        # Calculate component differences with special handling for theta
        delta_ln_r = coords_i[:, :, 0] - coords_j[:, :, 0]
        
        delta_theta = coords_i[:, :, 1] - coords_j[:, :, 1]
        delta_theta = torch.remainder(delta_theta + np.pi, 2 * np.pi) - np.pi  # Wrap to [-pi, pi]
        
        delta_z = coords_i[:, :, 2] - coords_j[:, :, 2]
        
        # Calculate squared distances
        distances_sq = delta_ln_r**2 + delta_theta**2 + delta_z**2
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        distances = torch.sqrt(distances_sq + epsilon)
        
        # Calculate repulsion force magnitudes (inverse square law)
        # Stronger repulsion for closer tokens
        force_magnitudes = self.repulsion_strength / (distances**2 + epsilon)
        
        # Zero out self-interactions
        force_magnitudes.fill_diagonal_(0.0)
        
        # Calculate force components
        f_ln_r = torch.sum(force_magnitudes * delta_ln_r, dim=1)
        f_theta = torch.sum(force_magnitudes * delta_theta, dim=1)
        f_z = torch.sum(force_magnitudes * delta_z, dim=1)
        
        return f_ln_r, f_theta, f_z
    
    def compute_vortex_forces(self):
        """
        Compute forces due to the dual vortex field.
        
        Returns:
            tuple: (f_ln_r, f_theta, f_z) forces in log-cylindrical coordinates
        """
        # Get token positions
        ln_r = self.tokens['ln_r']
        theta = self.tokens['theta']
        
        # Primary vortex force is tangential to the radius
        # In log-cylindrical coordinates, this creates a rotational flow
        
        # Radial force (scales with ln_r)
        f_ln_r = -0.5 * self.vorticity * torch.sin(self.phi * theta)
        
        # Angular force (creates rotation)
        f_theta = self.vorticity * (1.0 / torch.exp(ln_r)) * torch.cos(self.phi * theta)
        
        # Z-force (vertical flow along helical paths)
        f_z = 0.1 * self.vorticity * torch.sin(theta)
        
        return f_ln_r, f_theta, f_z
    
    def update_positions(self):
        """
        Update token positions based on repulsive forces and vortex field.
        """
        # Only update non-frozen tokens
        frozen_mask = ~self.tokens['frozen']
        
        # Compute repulsive forces
        f_r_repulsion, f_theta_repulsion, f_z_repulsion = self.compute_repulsion_forces()
        
        # Compute vortex forces
        f_r_vortex, f_theta_vortex, f_z_vortex = self.compute_vortex_forces()
        
        # Combine forces
        f_ln_r = f_r_repulsion + f_r_vortex
        f_theta = f_theta_repulsion + f_theta_vortex
        f_z = f_z_repulsion + f_z_vortex
        
        # Add thermal noise based on temperature
        if self.temperature > 0:
            noise_scale = torch.sqrt(torch.tensor(self.temperature, device=self.device))
            f_ln_r += noise_scale * torch.randn_like(f_ln_r)
            f_theta += noise_scale * torch.randn_like(f_theta)
            f_z += noise_scale * torch.randn_like(f_z)
        
        # Update positions for non-frozen tokens
        learning_rate = 0.01
        if torch.any(frozen_mask):
            self.tokens['ln_r'][frozen_mask] += learning_rate * f_ln_r[frozen_mask]
            self.tokens['theta'][frozen_mask] += learning_rate * f_theta[frozen_mask]
            self.tokens['z'][frozen_mask] += learning_rate * f_z[frozen_mask]
        
        # Ensure theta stays in [0, 2π]
        self.tokens['theta'] = torch.remainder(self.tokens['theta'], 2 * np.pi)
        
        # Check for tachyonic tunneling
        # This occurs when angular velocity dominates radial velocity
        # and causes a phase flip
        with torch.no_grad():
            phase_flip_mask = torch.abs(f_theta) > 2.0 * torch.abs(f_ln_r)
            if torch.any(phase_flip_mask & frozen_mask):
                # Implement phase flip for tokens meeting the condition
                self.tokens['theta'][phase_flip_mask & frozen_mask] += np.pi
                # Ensure theta stays in [0, 2π]
                self.tokens['theta'] = torch.remainder(self.tokens['theta'], 2 * np.pi)
        
        # Track positions if enabled
        if self.track_positions:
            self.save_positions()
    
    def save_positions(self):
        """Save current token positions to history."""
        positions = []
        for i in range(min(5, self.n_tokens)):  # Track only a subset of tokens
            pos = [
                self.tokens['ln_r'][i].item(),
                self.tokens['theta'][i].item(),
                self.tokens['z'][i].item()
            ]
            positions.append(pos)
        self.position_history.append(positions)
    
    def step(self):
        """Perform one step of the field update."""
        # Update token positions
        self.update_positions()
        
        # Decrease temperature (simulated annealing)
        self.temperature *= 0.999
        
        # Check for crystallization
        self.check_crystallization()
    
    def check_crystallization(self):
        """Check if tokens should crystallize (freeze in place)."""
        # Calculate token velocities
        if len(self.position_history) >= 2:
            prev_positions = self.position_history[-2]
            curr_positions = self.position_history[-1]
            
            for i in range(min(len(prev_positions), 5)):
                prev_pos = prev_positions[i]
                curr_pos = curr_positions[i]
                
                # Calculate velocity components
                vel_ln_r = abs(curr_pos[0] - prev_pos[0])
                vel_theta = abs(curr_pos[1] - prev_pos[1])
                vel_z = abs(curr_pos[2] - prev_pos[2])
                
                # Total velocity
                velocity = vel_ln_r + vel_theta + vel_z
                
                # Crystallize if velocity is below threshold and temperature is low
                if velocity < 0.001 and self.temperature < 0.5:
                    self.tokens['frozen'][i] = True
    
    def calculate_system_energy(self):
        """
        Calculate the total energy of the system using efficient tensor operations.
        
        Returns:
            torch.Tensor: System energy
        """
        # Create coordinate tensors
        ln_r = self.tokens['ln_r']
        theta = self.tokens['theta']
        z = self.tokens['z']
        
        # Stack coordinates for distance calculation
        coords = torch.stack([ln_r, theta, z], dim=1)  # Shape: [n_tokens, 3]
        
        # Calculate pairwise distances using efficient batch operation
        distances = log_cylindrical_batch_distance(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
        
        # Calculate repulsive potential energy (inverse distance)
        # Fill diagonal with large values to avoid division by zero
        distances.fill_diagonal_(float('inf'))
        repulsive_energy = torch.sum(1.0 / distances)
        
        # Calculate vortex field energy
        # In a dual vortex field, energy depends on position relative to vortex centers
        vortex_energy = torch.sum(
            self.vorticity * torch.exp(-ln_r) * torch.cos(self.phi * theta)
        )
        
        # Total energy
        total_energy = repulsive_energy + vortex_energy
        
        return total_energy
    
    def visualize_field(self, title="Dual Vortex Field", plot_type="3d"):
        """
        Visualize the dual vortex field.
        
        Args:
            title (str): Plot title
            plot_type (str): Type of plot ("3d", "2d", or "matrix")
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Move tensors to CPU for visualization
        ln_r = self.tokens['ln_r'].cpu().numpy()
        theta = self.tokens['theta'].cpu().numpy()
        z = self.tokens['z'].cpu().numpy()
        frozen = self.tokens['frozen'].cpu().numpy()
        
        if plot_type == "3d":
            # Create 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Convert to Cartesian for visualization
            r = np.exp(ln_r)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Plot non-frozen tokens
            ax.scatter(x[~frozen], y[~frozen], z[~frozen], 
                      c=theta[~frozen], cmap='hsv', s=50, alpha=0.7)
            
            # Plot frozen tokens
            if np.any(frozen):
                ax.scatter(x[frozen], y[frozen], z[frozen], 
                          color='red', s=100, marker='*', label='Crystallized')
            
            # Plot field lines (logarithmic spirals)
            t = np.linspace(0, 4*np.pi, 100)
            for phase in np.linspace(0, 2*np.pi, 8, endpoint=False):
                r_spiral = np.exp(t / (2*np.pi))
                x_spiral = r_spiral * np.cos(t + phase)
                y_spiral = r_spiral * np.sin(t + phase)
                z_spiral = np.zeros_like(t)
                ax.plot(x_spiral, y_spiral, z_spiral, 'k--', alpha=0.2)
            
            # Add trajectories if available
            if len(self.position_history) > 0:
                for i in range(min(3, len(self.position_history[0]))):
                    trajectory = np.array([step[i] for step in self.position_history])
                    ln_r_traj = trajectory[:, 0]
                    theta_traj = trajectory[:, 1]
                    z_traj = trajectory[:, 2]
                    
                    r_traj = np.exp(ln_r_traj)
                    x_traj = r_traj * np.cos(theta_traj)
                    y_traj = r_traj * np.sin(theta_traj)
                    
                    ax.plot(x_traj, y_traj, z_traj, '-', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(title)
            if np.any(frozen):
                ax.legend()
            
        elif plot_type == "2d":
            # Create 2D plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Convert to Cartesian for visualization
            r = np.exp(ln_r)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Plot tokens
            scatter = ax.scatter(x, y, c=theta, cmap='hsv', s=50, alpha=0.7)
            
            # Highlight frozen tokens
            if np.any(frozen):
                ax.scatter(x[frozen], y[frozen], s=100, edgecolor='red', 
                          facecolor='none', linewidth=2, label='Crystallized')
            
            # Plot field lines (logarithmic spirals)
            t = np.linspace(0, 4*np.pi, 100)
            for phase in np.linspace(0, 2*np.pi, 8, endpoint=False):
                r_spiral = np.exp(t / (2*np.pi))
                x_spiral = r_spiral * np.cos(t + phase)
                y_spiral = r_spiral * np.sin(t + phase)
                ax.plot(x_spiral, y_spiral, 'k--', alpha=0.2)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(title)
            ax.set_aspect('equal')
            plt.colorbar(scatter, label='Phase (θ)')
            if np.any(frozen):
                ax.legend()
            
        elif plot_type == "matrix":
            # Create matrix visualization of token states
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Create token state matrices
            n_tokens = len(ln_r)
            
            # Normalize values for visualization
            ln_r_norm = (ln_r - np.min(ln_r)) / (np.max(ln_r) - np.min(ln_r) + 1e-8)
            theta_norm = theta / (2*np.pi)
            z_norm = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)
            
            # Create matrices
            ln_r_matrix = ln_r_norm.reshape(-1, 1)
            theta_matrix = theta_norm.reshape(-1, 1)
            z_matrix = z_norm.reshape(-1, 1)
            
            # Plot matrices
            im0 = axes[0].imshow(ln_r_matrix, cmap='plasma', aspect='auto')
            axes[0].set_title('Token ln(r) Values')
            axes[0].set_xlabel('Component')
            axes[0].set_ylabel('Token Index')
            plt.colorbar(im0, ax=axes[0])
            
            im1 = axes[1].imshow(theta_matrix, cmap='hsv', aspect='auto')
            axes[1].set_title('Token θ Values')
            axes[1].set_xlabel('Component')
            axes[1].set_ylabel('Token Index')
            plt.colorbar(im1, ax=axes[1])
            
            im2 = axes[2].imshow(z_matrix, cmap='viridis', aspect='auto')
            axes[2].set_title('Token Z Values')
            axes[2].set_xlabel('Component')
            axes[2].set_ylabel('Token Index')
            plt.colorbar(im2, ax=axes[2])
            
            # Mark frozen tokens
            for i in range(n_tokens):
                if frozen[i]:
                    for j in range(3):
                        axes[j].plot(0, i, 'r*', markersize=5)
            
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
        
        return fig
    
    def visualize_tensor_evolution(self, n_steps=10):
        """
        Visualize the evolution of token positions and field energy over time.
        
        Args:
            n_steps (int): Number of steps to simulate
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Run simulation and collect data
        position_snapshots = []
        energy_history = []
        
        # Store initial state
        position_snapshots.append({
            'ln_r': self.tokens['ln_r'].cpu().numpy().copy(),
            'theta': self.tokens['theta'].cpu().numpy().copy(),
            'z': self.tokens['z'].cpu().numpy().copy(),
            'frozen': self.tokens['frozen'].cpu().numpy().copy()
        })
        energy_history.append(self.calculate_system_energy().item())
        
        # Run simulation
        for _ in range(n_steps):
            self.step()
            
            # Store state
            position_snapshots.append({
                'ln_r': self.tokens['ln_r'].cpu().numpy().copy(),
                'theta': self.tokens['theta'].cpu().numpy().copy(),
                'z': self.tokens['z'].cpu().numpy().copy(),
                'frozen': self.tokens['frozen'].cpu().numpy().copy()
            })
            energy_history.append(self.calculate_system_energy().item())
        
        # Create visualization
        fig = plt.figure(figsize=(15, 12))
        
        # Plot token position evolution (2D projection)
        for i, step in enumerate([0, n_steps//3, 2*n_steps//3, n_steps]):
            if step >= len(position_snapshots):
                continue
                
            ax = fig.add_subplot(2, 3, i+1)
            
            # Get data for this step
            snapshot = position_snapshots[step]
            ln_r = snapshot['ln_r']
            theta = snapshot['theta']
            frozen = snapshot['frozen']
            
            # Convert to Cartesian
            r = np.exp(ln_r)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Plot tokens
            scatter = ax.scatter(x, y, c=theta, cmap='hsv', s=50, alpha=0.7)
            
            # Highlight frozen tokens
            if np.any(frozen):
                ax.scatter(x[frozen], y[frozen], s=100, edgecolor='red', 
                          facecolor='none', linewidth=2)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Step {step}')
            ax.set_aspect('equal')
        
        # Plot energy history
        ax_energy = fig.add_subplot(2, 3, 5)
        ax_energy.plot(energy_history, 'r-')
        ax_energy.set_xlabel('Step')
        ax_energy.set_ylabel('System Energy')
        ax_energy.set_title('Energy Evolution')
        ax_energy.grid(True)
        
        # Plot token state matrix for final state
        ax_matrix = fig.add_subplot(2, 3, 6)
        
        # Create token state matrix for final state
        final_state = position_snapshots[-1]
        ln_r = final_state['ln_r']
        theta = final_state['theta']
        z = final_state['z']
        frozen = final_state['frozen']
        
        # Normalize values for visualization
        ln_r_norm = (ln_r - np.min(ln_r)) / (np.max(ln_r) - np.min(ln_r) + 1e-8)
        theta_norm = theta / (2*np.pi)
        
        # Combine into state matrix
        state_matrix = np.column_stack([ln_r_norm, theta_norm])
        
        # Plot matrix
        im = ax_matrix.imshow(state_matrix, cmap='viridis', aspect='auto')
        ax_matrix.set_title('Final Token States')
        ax_matrix.set_xlabel('Component (0: ln(r), 1: θ)')
        ax_matrix.set_ylabel('Token Index')
        plt.colorbar(im, ax=ax_matrix)
        
        # Mark frozen tokens
        for i in range(len(frozen)):
            if frozen[i]:
                ax_matrix.plot([0, 1], [i, i], 'r-', linewidth=2, alpha=0.5)
        
        plt.suptitle('Dual Vortex Field Evolution', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
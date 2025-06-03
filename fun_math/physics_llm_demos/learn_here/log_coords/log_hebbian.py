import torch
import numpy as np
import matplotlib.pyplot as plt
from .log_coords import log_cylindrical_batch_distance, log_cylindrical_to_cartesian

class LogHebbianNetwork:
    """
    Implements a Hebbian learning network in log-cylindrical space
    for efficient O(N log N) sequence processing.
    """
    
    def __init__(self, n_tokens=100, dim=3, device=None):
        """
        Initialize the Hebbian network.
        
        Args:
            n_tokens (int): Number of tokens in the network
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
        
        # Initialize Hebbian connection matrix
        self.connections = torch.zeros((n_tokens, n_tokens), device=self.device)
        
        # Parameters
        self.learning_rate = 0.01
        self.temperature = 1.0
        self.decay_rate = 0.999
        
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
            phi = (1 + 5**0.5) / 2  # Golden ratio
            indices = torch.arange(self.n_tokens, device=self.device).float()
            
            # Natural logarithm of radius (logarithmic spiral)
            self.tokens['ln_r'] = torch.log(indices + 1) / 2
            
            # Angular coordinate with golden ratio spacing
            self.tokens['theta'] = 2 * np.pi * torch.fmod(indices / phi, 1.0)
            
            # Z-coordinate as function of index
            self.tokens['z'] = indices / self.n_tokens
        
        # Reset frozen state
        self.tokens['frozen'] = torch.zeros(self.n_tokens, dtype=torch.bool, device=self.device)
        
        # Reset connections
        self.connections = torch.zeros((self.n_tokens, self.n_tokens), device=self.device)
        
        # Clear history
        self.position_history = []
    
    def update_connections(self):
        """
        Update Hebbian connections based on token proximity in log-cylindrical space.
        Uses efficient tensor operations instead of explicit loops.
        """
        # Stack coordinates
        coords = torch.stack([self.tokens['ln_r'], self.tokens['theta'], self.tokens['z']], dim=1)
        
        # Calculate pairwise distances
        distances = log_cylindrical_batch_distance(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
        
        # Calculate proximity (closer tokens have higher values)
        proximity = torch.exp(-distances / self.temperature)
        
        # Apply Hebbian learning
        self.connections = self.connections * self.decay_rate + self.learning_rate * proximity
        
        # Zero out self-connections
        self.connections.fill_diagonal_(0)
    
    def update_positions(self):
        """
        Update token positions based on Hebbian connections and repulsive forces.
        Uses efficient tensor operations for better performance.
        """
        # Create coordinate tensors
        ln_r = self.tokens['ln_r']
        theta = self.tokens['theta']
        z = self.tokens['z']
        
        # Only update non-frozen tokens
        frozen_mask = ~self.tokens['frozen']
        
        # Stack coordinates for distance calculation
        coords = torch.stack([ln_r, theta, z], dim=1)  # Shape: [n_tokens, 3]
        
        # Calculate repulsive forces
        # We'll use a simplified inverse square law in log-cylindrical space
        
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
        
        # Calculate force magnitudes (inverse square law)
        # Stronger repulsion for closer tokens
        force_magnitudes = 1.0 / (distances**2 + epsilon)
        
        # Zero out self-interactions
        force_magnitudes.fill_diagonal_(0.0)
        
        # Calculate force components
        f_ln_r = torch.sum(force_magnitudes * delta_ln_r, dim=1)
        f_theta = torch.sum(force_magnitudes * delta_theta, dim=1)
        f_z = torch.sum(force_magnitudes * delta_z, dim=1)
        
        # Add attractive forces from Hebbian connections
        # Connections act as springs pulling tokens together
        attractive_forces_ln_r = torch.sum(self.connections * delta_ln_r, dim=1)
        attractive_forces_theta = torch.sum(self.connections * delta_theta, dim=1)
        attractive_forces_z = torch.sum(self.connections * delta_z, dim=1)
        
        # Combine repulsive and attractive forces
        net_f_ln_r = f_ln_r - attractive_forces_ln_r
        net_f_theta = f_theta - attractive_forces_theta
        net_f_z = f_z - attractive_forces_z
        
        # Update positions for non-frozen tokens
        learning_rate = 0.01
        if torch.any(frozen_mask):
            self.tokens['ln_r'][frozen_mask] += learning_rate * net_f_ln_r[frozen_mask]
            self.tokens['theta'][frozen_mask] += learning_rate * net_f_theta[frozen_mask]
            self.tokens['z'][frozen_mask] += learning_rate * net_f_z[frozen_mask]
        
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
        """Perform one step of the Hebbian network update."""
        # Update Hebbian connections
        self.update_connections()
        
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
        """Calculate the total energy of the system."""
        # Stack coordinates
        coords = torch.stack([self.tokens['ln_r'], self.tokens['theta'], self.tokens['z']], dim=1)
        
        # Calculate pairwise distances
        distances = log_cylindrical_batch_distance(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
        
        # Calculate repulsive potential energy (inverse distance)
        repulsive_energy = torch.sum(1.0 / (distances + 1e-8))
        
        # Calculate attractive potential energy from Hebbian connections
        attractive_energy = -torch.sum(self.connections * torch.exp(-distances))
        
        # Total energy
        total_energy = repulsive_energy + attractive_energy
        
        return total_energy
    
    def visualize_network(self, title="Hebbian Network in Log-Cylindrical Space"):
        """
        Visualize the Hebbian network in log-cylindrical space.
        
        Args:
            title (str): Plot title
        """
        # Move tensors to CPU for visualization
        ln_r = self.tokens['ln_r'].cpu().numpy()
        theta = self.tokens['theta'].cpu().numpy()
        z = self.tokens['z'].cpu().numpy()
        frozen = self.tokens['frozen'].cpu().numpy()
        connections = self.connections.cpu().numpy()
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        # 3D plot of token positions
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Convert to Cartesian for visualization
        r = np.exp(ln_r)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Plot non-frozen tokens
        ax1.scatter(x[~frozen], y[~frozen], z[~frozen], c=theta[~frozen], 
                   cmap='hsv', s=50, alpha=0.7)
        
        # Plot frozen tokens
        if np.any(frozen):
            ax1.scatter(x[frozen], y[frozen], z[frozen], c='red', s=100, marker='*')
        
        # Plot strongest connections
        threshold = np.percentile(connections[connections > 0], 95)
        for i in range(self.n_tokens):
            for j in range(i+1, self.n_tokens):
                if connections[i, j] > threshold:
                    ax1.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 
                            'k-', alpha=0.3, linewidth=connections[i, j] * 5)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Token Positions')
        
        # Plot Hebbian connection matrix
        ax2 = fig.add_subplot(122)
        im = ax2.imshow(connections, cmap='viridis')
        plt.colorbar(im, ax=ax2)
        ax2.set_title('Hebbian Connection Matrix')
        ax2.set_xlabel('Token Index')
        ax2.set_ylabel('Token Index')
        
        # Set overall title
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def visualize_tensor_evolution(self, n_steps=10):
        """
        Visualize the evolution of the connection tensor over time.
        
        Args:
            n_steps (int): Number of steps to simulate
        """
        # Store connection matrices over time
        connection_history = []
        energy_history = []
        
        # Run simulation
        for _ in range(n_steps):
            # Store current state
            connection_history.append(self.connections.cpu().numpy().copy())
            energy_history.append(self.calculate_system_energy().item())
            
            # Update network
            self.step()
        
        # Visualize evolution
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Select steps to visualize
        steps_to_show = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]
        
        # Plot connection matrices
        for i, step in enumerate(steps_to_show[:5]):
            ax = axes.flatten()[i]
            im = ax.imshow(connection_history[step], cmap='viridis')
            ax.set_title(f'Step {step}')
            plt.colorbar(im, ax=ax)
        
        # Plot energy history
        ax = axes.flatten()[5]
        ax.plot(energy_history, 'r-')
        ax.set_xlabel('Step')
        ax.set_ylabel('System Energy')
        ax.set_title('Energy Evolution')
        ax.grid(True)
        
        # Set overall title
        plt.suptitle('Hebbian Connection Matrix Evolution', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
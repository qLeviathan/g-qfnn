"""
Sparse Log-Hebbian Neural Field Module
Implements Hebbian learning in logarithmic space for numerical stability
across many orders of magnitude, optimized for GPU acceleration
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import time
import sys

# Import the log-coordinate system
from log_coords import LogCylindricalCoords, device, PHI, PI, TAU, EPS

class SparseLogHebbian:
    """
    Sparse implementation of Hebbian learning in logarithmic space
    Uses O(N·k) operations where k is the average connections per token
    """
    def __init__(self, size: int, default_ln_value: float = float('-inf'), device=None):
        """
        Initialize the sparse Hebbian matrix in logarithmic space
        
        Args:
            size: Size of the token population
            default_ln_value: Default log value for non-existent connections
            device: Computation device
        """
        self.size = size
        self.default_ln_value = default_ln_value
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Constants
        self.phi = torch.tensor((1 + np.sqrt(5)) / 2, device=self.device)
        self.eps = torch.tensor(1e-10, device=self.device)
        
        # Initialize sparse structure
        # Indices: List of (i, j) tuples for non-zero entries
        # Values: Corresponding ln-values (natural log of connection strength)
        self.indices = []
        self.ln_values = []
        
        # Create mapping for fast lookups
        self.index_map = {}  # Maps (i, j) -> position in values list
        
        # Cache for sparse matrix-vector operations
        self.cache = {}
    
    def get_ln_value(self, i: int, j: int) -> float:
        """
        Get log-value of a connection
        
        Args:
            i, j: Token indices
            
        Returns:
            ln_value: Natural log of connection strength
        """
        key = (i, j)
        if key in self.index_map:
            return self.ln_values[self.index_map[key]]
        return self.default_ln_value
    
    def set_ln_value(self, i: int, j: int, ln_value: float):
        """
        Set log-value of a connection
        
        Args:
            i, j: Token indices
            ln_value: Natural log of connection strength
        """
        key = (i, j)
        if key in self.index_map:
            # Update existing entry
            self.ln_values[self.index_map[key]] = ln_value
        else:
            # Add new entry
            self.index_map[key] = len(self.ln_values)
            self.indices.append(key)
            self.ln_values.append(ln_value)
        
        # Clear cache
        self.cache = {}
    
    def sparse_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sparse matrix-vector product in log space
        Computes H·x efficiently using sparse structure
        
        Args:
            x: Input tensor of shape (N, D) or (N,)
            
        Returns:
            result: H·x with same shape as x
        """
        # Check if result is in cache
        cache_key = hash(x.data_ptr())
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Handle empty matrix case
        if not self.indices:
            if x.dim() == 2:
                result = torch.zeros(self.size, x.shape[1], device=x.device)
            else:
                result = torch.zeros(self.size, device=x.device)
            return result
        
        # Create PyTorch sparse tensor for efficient operations
        i_indices = [i for i, j in self.indices]
        j_indices = [j for i, j in self.indices]
        indices = torch.tensor([i_indices, j_indices], dtype=torch.long, device=self.device)
        
        # Create values tensor (exponentiate to convert from log space)
        values = torch.tensor(self.ln_values, device=self.device).exp()
        
        # Create sparse tensor
        sparse_matrix = torch.sparse.FloatTensor(
            indices, values, (self.size, self.size)
        )
        
        # Compute matrix-vector product
        if x.dim() == 2:
            # Batched version for matrix inputs
            result = torch.zeros(self.size, x.shape[1], device=self.device)
            for k in range(x.shape[1]):
                result[:, k] = torch.sparse.mm(sparse_matrix, x[:, k].unsqueeze(1)).squeeze(1)
        else:
            # Single vector version
            result = torch.sparse.mm(sparse_matrix, x.unsqueeze(1)).squeeze(1)
        
        # Cache result
        self.cache[cache_key] = result
        return result
    
    def log_update(self, ln_r: torch.Tensor, theta: torch.Tensor, 
                  coords: LogCylindricalCoords, dt: float, cutoff_distance: float = None):
        """
        Update Hebbian connections in log space
        
        Args:
            ln_r: Log-radius of tokens
            theta: Angle of tokens
            coords: Log-cylindrical coordinate system
            dt: Time step
            cutoff_distance: Maximum connection distance (ln-value)
        """
        N = ln_r.shape[0]
        
        # Set default cutoff if not provided (λ = φ² from the paper)
        if cutoff_distance is None:
            cutoff_distance = torch.log(self.phi ** 2)
        
        # Decay existing connections
        gamma = 0.1 * dt  # Decay rate
        for idx in range(len(self.ln_values)):
            # Decay log value
            self.ln_values[idx] -= gamma
        
        # Remove very weak connections
        threshold = cutoff_distance - 5.0  # ln-threshold for pruning
        
        # Filter connections
        valid_indices = []
        valid_ln_values = []
        new_index_map = {}
        
        for idx, (i, j) in enumerate(self.indices):
            if self.ln_values[idx] > threshold:
                new_index_map[(i, j)] = len(valid_ln_values)
                valid_indices.append((i, j))
                valid_ln_values.append(self.ln_values[idx])
        
        self.indices = valid_indices
        self.ln_values = valid_ln_values
        self.index_map = new_index_map
        
        # Add new connections between nearby tokens
        # This is a simplified approach - in a full implementation,
        # we would use a spatial hash or Barnes-Hut tree for O(N·log(N)) performance
        
        # Process in batches for efficiency
        batch_size = min(100, N)  # Smaller batch for testing
        
        for i in range(N):
            # Sample a few potential connections
            j_candidates = torch.randint(0, N, (batch_size,), device=self.device)
            
            for j in j_candidates:
                j = j.item()
                if i != j:
                    # Compute log-Cartesian distance
                    ln_dist = coords.log_cartesian_distance(ln_r[i], theta[i], ln_r[j], theta[j])
                    
                    # Add/strengthen connection if within cutoff
                    if ln_dist < cutoff_distance:
                        # Get current strength
                        current_ln_value = self.get_ln_value(i, j)
                        
                        if current_ln_value == self.default_ln_value:
                            # New connection: initialize with distance
                            new_ln_value = ln_dist
                        else:
                            # Existing connection: strengthen with log-sum-exp
                            # ln(exp(a) + exp(b)) = a + ln(1 + exp(b-a))
                            max_ln = max(current_ln_value, ln_dist + dt)
                            min_ln = min(current_ln_value, ln_dist + dt)
                            new_ln_value = max_ln + np.log(1.0 + np.exp(min_ln - max_ln))
                        
                        self.set_ln_value(i, j, new_ln_value)
        
        # Clear cache after update
        self.cache = {}
    
    def compute_hebbian_pitch(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute pitch (preferred angle) for each token based on Hebbian connections
        
        Args:
            theta: Current angles of tokens
            
        Returns:
            pitch: Preferred angle for each token
        """
        # Compute p_x = H·cos(θ), p_y = H·sin(θ)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        p_x = self.sparse_matvec(cos_theta)
        p_y = self.sparse_matvec(sin_theta)
        
        # Compute preferred angle: pitch = atan2(p_y, p_x)
        pitch = torch.atan2(p_y, p_x)
        
        return pitch
    
    def visualize_hebbian_network(self, ln_r: torch.Tensor, theta: torch.Tensor, 
                                coords: LogCylindricalCoords, 
                                save_path: Optional[str] = None):
        """
        Visualize the Hebbian connection network
        
        Args:
            ln_r: Log-radius of tokens
            theta: Angle of tokens
            coords: Log-cylindrical coordinate system
            save_path: Optional path to save the figure
        """
        # Make sure everything is on the same device
        ln_r = ln_r.to(self.device)
        theta = theta.to(self.device)
        
        # Convert to Cartesian for visualization
        x, y = coords.ln_r_theta_to_cartesian(ln_r, theta)
        
        # Compute pitch
        pitch = self.compute_hebbian_pitch(theta)
        
        # Calculate pitch alignment error
        d_theta = torch.remainder(pitch - theta + torch.tensor(np.pi, device=self.device), 
                                 torch.tensor(2*np.pi, device=self.device)) - torch.tensor(np.pi, device=self.device)
        
        # Convert to CPU for plotting
        x_cpu = x.cpu().numpy()
        y_cpu = y.cpu().numpy()
        theta_cpu = theta.cpu().numpy()
        pitch_cpu = pitch.cpu().numpy()
        d_theta_cpu = d_theta.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot tokens in Cartesian space
        scatter = axes[0, 0].scatter(x_cpu, y_cpu, c=theta_cpu, cmap='hsv', s=50, alpha=0.7)
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_title('Token Positions (colored by θ)')
        axes[0, 0].set_aspect('equal')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0], label='θ')
        
        # Plot Hebbian connections
        axes[0, 1].scatter(x_cpu, y_cpu, c='black', s=30, alpha=0.5)
        
        # Draw connections
        max_connections = 200  # Limit for visualization
        connection_count = min(max_connections, len(self.indices))
        
        # Sort connections by strength
        ln_values_np = np.array(self.ln_values)
        sorted_indices = np.argsort(ln_values_np)[-connection_count:]
        
        for idx in sorted_indices:
            i, j = self.indices[idx]
            strength = np.exp(self.ln_values[idx])
            
            # Draw a line between connected tokens
            axes[0, 1].plot([x_cpu[i], x_cpu[j]], [y_cpu[i], y_cpu[j]], 
                          alpha=min(0.8, strength), 
                          linewidth=max(0.5, 2 * strength), 
                          color='blue')
        
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        axes[0, 1].set_title(f'Hebbian Connections (top {connection_count})')
        axes[0, 1].set_aspect('equal')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot pitch vs. theta
        axes[1, 0].scatter(theta_cpu, pitch_cpu, c=np.arange(len(theta_cpu)), cmap='viridis', s=30, alpha=0.7)
        axes[1, 0].plot([0, TAU.item()], [0, TAU.item()], 'r--', label='Perfect Alignment')
        axes[1, 0].set_xlabel('θ')
        axes[1, 0].set_ylabel('Pitch')
        axes[1, 0].set_title('Hebbian Pitch vs. θ')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot pitch alignment error
        scatter = axes[1, 1].scatter(x_cpu, y_cpu, c=d_theta_cpu, cmap='coolwarm', s=50, alpha=0.7, vmin=-PI.item(), vmax=PI.item())
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        axes[1, 1].set_title('Pitch Alignment Error')
        axes[1, 1].set_aspect('equal')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='Pitch - θ')
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Hebbian network visualization saved to {save_path}")
        else:
            # Try to show, but don't error in non-interactive environments
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plot: {e}")
        
        plt.close()

# Example usage
if __name__ == "__main__":
    print("Testing Sparse Log-Hebbian Learning")
    
    # Use CPU for testing to avoid numpy conversion issues
    cpu_device = torch.device('cpu')
    
    # Create coordinate system
    coords = LogCylindricalCoords(device=cpu_device)
    
    # Create a population of tokens
    N = 100
    ln_r, theta = coords.generate_golden_spiral(N)
    
    # Create Hebbian network
    hebbian = SparseLogHebbian(N, device=cpu_device)
    
    # Perform Hebbian updates
    print("Performing Hebbian updates...")
    start_time = time.time()
    
    # Run multiple updates
    num_updates = 10
    dt = 0.1
    
    for i in range(num_updates):
        hebbian.log_update(ln_r, theta, coords, dt)
        
        if i % 2 == 0:
            print(f"Update {i+1}/{num_updates}, connections: {len(hebbian.indices)}")
    
    end_time = time.time()
    print(f"Hebbian updates completed in {end_time - start_time:.2f} seconds")
    
    # Visualize the network
    hebbian.visualize_hebbian_network(ln_r, theta, coords, save_path="hebbian_network.png")
    
    # Test individual components
    # Compute pitch for each token
    pitch = hebbian.compute_hebbian_pitch(theta)
    
    # Compute alignment error
    d_theta = torch.remainder(pitch - theta + PI, TAU) - PI
    
    print("\nHebbian Learning Test:")
    print(f"Number of connections: {len(hebbian.indices)}")
    print(f"Average pitch error: {torch.abs(d_theta).mean().item():.4f} radians")
    print(f"Max pitch error: {torch.abs(d_theta).max().item():.4f} radians")
    
    # Memory usage
    memory_bytes = sum(sys.getsizeof(v) for v in hebbian.ln_values)
    memory_bytes += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in hebbian.index_map.items())
    
    print(f"Sparse memory usage: {memory_bytes / 1024:.2f} KB for {N} tokens")
    print(f"Full matrix would use: {N * N * 4 / 1024:.2f} KB")
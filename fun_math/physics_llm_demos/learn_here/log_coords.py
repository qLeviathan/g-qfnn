"""
Log-Cylindrical Coordinate System for Neural Field Dynamics
Using natural logarithm representation for numerical stability

Key features:
- CUDA-accelerated tensor operations
- Natural log coordinates: ln(r) = ln(1 + n·ln(φ))
- Dual coordinate representation: log-polar and Cartesian
- Visualization tools for comparison and ablation studies
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

# Set device - always use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Constants (on device)
PHI = torch.tensor((1 + np.sqrt(5)) / 2, device=device)  # Golden ratio
PI = torch.tensor(np.pi, device=device)
TAU = torch.tensor(2 * np.pi, device=device)
EPS = torch.tensor(1e-10, device=device)  # Numerical stability epsilon

class LogCylindricalCoords:
    """
    Natural logarithm cylindrical coordinate system
    Handles conversion between log-polar and Cartesian coordinates
    with numerical stability across many orders of magnitude
    """
    def __init__(self, device=None):
        """Initialize coordinate system"""
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Constants (all on device)
        self.phi = torch.tensor((1 + np.sqrt(5)) / 2, device=self.device)
        self.pi = torch.tensor(np.pi, device=self.device)
        self.tau = torch.tensor(2 * np.pi, device=self.device)
        self.eps = torch.tensor(1e-10, device=self.device)
    
    def generate_golden_spiral(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate points along a golden spiral in ln(r), θ space
        
        Args:
            n: Number of points
            
        Returns:
            ln_r: Log-radius coordinates (natural log)
            theta: Angular coordinates
        """
        # Generate points with golden ratio spacing
        indices = torch.arange(n, device=self.device, dtype=torch.float32)
        
        # Theta: golden angle in radians (2π/φ per step)
        theta = torch.remainder(indices * self.tau / self.phi, self.tau)
        
        # ln(r): natural log spacing following φ
        # ln(r) = ln(1 + n·ln(φ))
        ln_r = torch.log(1 + indices * torch.log(self.phi))
        
        return ln_r, theta
    
    def ln_r_theta_to_cartesian(self, ln_r: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert log-polar coordinates to Cartesian
        
        Args:
            ln_r: Log-radius (natural log)
            theta: Angle
            
        Returns:
            x, y: Cartesian coordinates
        """
        # r = exp(ln_r)
        r = torch.exp(ln_r)
        
        # Convert to Cartesian
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        return x, y
    
    def cartesian_to_ln_r_theta(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert Cartesian coordinates to log-polar
        
        Args:
            x, y: Cartesian coordinates
            
        Returns:
            ln_r: Log-radius (natural log)
            theta: Angle
        """
        # r = sqrt(x² + y²)
        r = torch.sqrt(x*x + y*y + self.eps)
        
        # ln(r) = ln(sqrt(x² + y²))
        ln_r = torch.log(r)
        
        # θ = atan2(y, x)
        theta = torch.atan2(y, x)
        
        # Ensure theta is in [0, 2π)
        theta = torch.remainder(theta, self.tau)
        
        return ln_r, theta
    
    def log_cartesian_distance(self, ln_r1: torch.Tensor, theta1: torch.Tensor, 
                               ln_r2: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        """
        Compute distance in log-Cartesian space
        Uses log-sum-exp for numerical stability
        
        Args:
            ln_r1, theta1: First point in log-polar coordinates
            ln_r2, theta2: Second point in log-polar coordinates
            
        Returns:
            ln_dist: Log of Euclidean distance (natural log)
        """
        # Get Cartesian coordinates
        x1, y1 = self.ln_r_theta_to_cartesian(ln_r1, theta1)
        x2, y2 = self.ln_r_theta_to_cartesian(ln_r2, theta2)
        
        # Compute log of squared distances
        ln_dx2 = torch.log(torch.abs(x1 - x2)**2 + self.eps)
        ln_dy2 = torch.log(torch.abs(y1 - y2)**2 + self.eps)
        
        # Compute log of distance: ln(sqrt(dx² + dy²)) = 0.5 * ln(dx² + dy²)
        ln_dist = 0.5 * torch.logsumexp(torch.stack([ln_dx2, ln_dy2]), dim=0)
        
        return ln_dist
    
    def log_cartesian_components(self, ln_r1: torch.Tensor, theta1: torch.Tensor,
                                ln_r2: torch.Tensor, theta2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-cylindrical displacement components
        
        Args:
            ln_r1, theta1: First point
            ln_r2, theta2: Second point
            
        Returns:
            d_ln_r: Difference in log-radius
            d_theta: Difference in angle (normalized to [-π, π])
        """
        # Direct difference in log-radius
        d_ln_r = ln_r2 - ln_r1
        
        # Angular difference (normalized to [-π, π])
        d_theta = torch.remainder(theta2 - theta1 + self.pi, self.tau) - self.pi
        
        return d_ln_r, d_theta
    
    def visualize_comparison(self, n: int = 100, save_path: Optional[str] = None):
        """
        Visualize comparison between log-cylindrical and standard coordinates
        
        Args:
            n: Number of points to visualize
            save_path: Path to save figure, if provided
        """
        # Generate golden spiral
        ln_r, theta = self.generate_golden_spiral(n)
        
        # Convert to Cartesian
        x, y = self.ln_r_theta_to_cartesian(ln_r, theta)
        
        # Convert to CPU for visualization
        ln_r_cpu = ln_r.cpu().numpy()
        theta_cpu = theta.cpu().numpy()
        x_cpu = x.cpu().numpy()
        y_cpu = y.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot in log-polar space
        axes[0, 0].scatter(ln_r_cpu, theta_cpu, c=np.arange(n), cmap='viridis', s=30, alpha=0.7)
        axes[0, 0].set_xlabel('ln(r)')
        axes[0, 0].set_ylabel('θ')
        axes[0, 0].set_title('Log-Cylindrical Coordinates')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot in Cartesian space
        axes[0, 1].scatter(x_cpu, y_cpu, c=np.arange(n), cmap='viridis', s=30, alpha=0.7)
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        axes[0, 1].set_title('Cartesian Coordinates')
        axes[0, 1].set_aspect('equal')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot distance calculations
        # Select a reference point
        ref_idx = n // 4
        ref_ln_r = ln_r[ref_idx]
        ref_theta = theta[ref_idx]
        
        # Compute distances
        ln_distances = torch.zeros_like(ln_r)
        cart_distances = torch.zeros_like(ln_r)
        
        for i in range(n):
            # Log-Cartesian distance
            ln_distances[i] = self.log_cartesian_distance(ref_ln_r, ref_theta, ln_r[i], theta[i])
            
            # Regular Cartesian distance
            ref_x, ref_y = self.ln_r_theta_to_cartesian(ref_ln_r, ref_theta)
            xi, yi = self.ln_r_theta_to_cartesian(ln_r[i], theta[i])
            cart_distances[i] = torch.sqrt((ref_x - xi)**2 + (ref_y - yi)**2)
        
        # Convert to CPU for plotting
        ln_distances_cpu = ln_distances.cpu().numpy()
        cart_distances_cpu = cart_distances.cpu().numpy()
        
        # Plot log-distance
        axes[1, 0].scatter(np.arange(n), ln_distances_cpu, c=np.arange(n), cmap='viridis', s=30, alpha=0.7)
        axes[1, 0].axvline(x=ref_idx, color='r', linestyle='--', label='Reference Point')
        axes[1, 0].set_xlabel('Point Index')
        axes[1, 0].set_ylabel('ln(distance)')
        axes[1, 0].set_title('Log-Cylindrical Distances')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot Cartesian distance
        axes[1, 1].scatter(np.arange(n), cart_distances_cpu, c=np.arange(n), cmap='viridis', s=30, alpha=0.7)
        axes[1, 1].axvline(x=ref_idx, color='r', linestyle='--', label='Reference Point')
        axes[1, 1].set_xlabel('Point Index')
        axes[1, 1].set_ylabel('Distance')
        axes[1, 1].set_title('Cartesian Distances')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def ablation_study(self, n: int = 100, noise_levels: list = [0, 0.1, 0.5, 1.0], 
                      save_path: Optional[str] = None):
        """
        Perform an ablation study showing robustness of log vs. standard coordinates
        
        Args:
            n: Number of points to use
            noise_levels: List of noise standard deviations to test
            save_path: Path to save figure, if provided
        """
        # Generate points
        ln_r, theta = self.generate_golden_spiral(n)
        
        # Create figure
        fig, axes = plt.subplots(len(noise_levels), 2, figsize=(12, 4*len(noise_levels)))
        
        for i, noise_level in enumerate(noise_levels):
            # Add noise to log-cylindrical coordinates
            noise_ln_r = ln_r + noise_level * torch.randn_like(ln_r)
            noise_theta = theta + noise_level * torch.randn_like(theta)
            
            # Convert to Cartesian
            noise_x, noise_y = self.ln_r_theta_to_cartesian(noise_ln_r, noise_theta)
            
            # Convert back to log-cylindrical
            recovered_ln_r, recovered_theta = self.cartesian_to_ln_r_theta(noise_x, noise_y)
            
            # Convert to CPU for visualization
            ln_r_cpu = ln_r.cpu().numpy()
            noise_ln_r_cpu = noise_ln_r.cpu().numpy()
            recovered_ln_r_cpu = recovered_ln_r.cpu().numpy()
            
            theta_cpu = theta.cpu().numpy()
            noise_theta_cpu = noise_theta.cpu().numpy()
            recovered_theta_cpu = recovered_theta.cpu().numpy()
            
            # Plot log-radius error
            axes[i, 0].scatter(np.arange(n), ln_r_cpu, label='Original', alpha=0.5)
            axes[i, 0].scatter(np.arange(n), noise_ln_r_cpu, label='Noisy', alpha=0.3)
            axes[i, 0].scatter(np.arange(n), recovered_ln_r_cpu, label='Recovered', alpha=0.3)
            axes[i, 0].set_xlabel('Point Index')
            axes[i, 0].set_ylabel('ln(r)')
            axes[i, 0].set_title(f'Log-Radius with Noise σ={noise_level}')
            axes[i, 0].grid(True, alpha=0.3)
            if i == 0:
                axes[i, 0].legend()
            
            # Plot theta error
            axes[i, 1].scatter(np.arange(n), theta_cpu, label='Original', alpha=0.5)
            axes[i, 1].scatter(np.arange(n), noise_theta_cpu, label='Noisy', alpha=0.3)
            axes[i, 1].scatter(np.arange(n), recovered_theta_cpu, label='Recovered', alpha=0.3)
            axes[i, 1].set_xlabel('Point Index')
            axes[i, 1].set_ylabel('θ')
            axes[i, 1].set_title(f'Angle with Noise σ={noise_level}')
            axes[i, 1].grid(True, alpha=0.3)
            if i == 0:
                axes[i, 1].legend()
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Ablation study saved to {save_path}")
        
        plt.show()

# Example usage and validation
if __name__ == "__main__":
    print("Testing Log-Cylindrical Coordinate System")
    
    # Create coordinate system
    log_coords = LogCylindricalCoords()
    
    # Generate and visualize golden spiral
    log_coords.visualize_comparison(n=200, save_path="log_cylindrical_comparison.png")
    
    # Perform ablation study
    log_coords.ablation_study(n=100, save_path="log_cylindrical_ablation.png")
    
    # Test individual conversions
    # Generate test points
    test_ln_r = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device)
    test_theta = torch.tensor([0.0, 0.5, 1.0, 1.5], device=device) * PI
    
    # Convert to Cartesian
    test_x, test_y = log_coords.ln_r_theta_to_cartesian(test_ln_r, test_theta)
    
    # Convert back to log-cylindrical
    test_ln_r2, test_theta2 = log_coords.cartesian_to_ln_r_theta(test_x, test_y)
    
    # Print results
    print("\nCoordinate Conversion Test:")
    print(f"Original ln(r): {test_ln_r.cpu().numpy()}")
    print(f"Original θ: {test_theta.cpu().numpy()}")
    print(f"Cartesian x: {test_x.cpu().numpy()}")
    print(f"Cartesian y: {test_y.cpu().numpy()}")
    print(f"Recovered ln(r): {test_ln_r2.cpu().numpy()}")
    print(f"Recovered θ: {test_theta2.cpu().numpy()}")
    
    # Test distance calculations
    test_dist = log_coords.log_cartesian_distance(
        test_ln_r[0], test_theta[0], 
        test_ln_r[1], test_theta[1]
    )
    
    # Verify with standard formula
    r1 = torch.exp(test_ln_r[0])
    r2 = torch.exp(test_ln_r[1])
    x1 = r1 * torch.cos(test_theta[0])
    y1 = r1 * torch.sin(test_theta[0])
    x2 = r2 * torch.cos(test_theta[1])
    y2 = r2 * torch.sin(test_theta[1])
    
    std_dist = torch.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    print("\nDistance Calculation Test:")
    print(f"Log-Cartesian distance ln(dist): {test_dist.item()}")
    print(f"Exp(ln(dist)): {torch.exp(test_dist).item()}")
    print(f"Standard Cartesian distance: {std_dist.item()}")
"""
Dual Vortex Field Dynamics with Tachyonic Tunneling
Implements the logarithmic-cylindrical field equations with 
dual vortex repulsion and tachyonic tunneling behavior
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Import our modules
from log_coords import LogCylindricalCoords, device, PHI, PI, TAU, EPS
from log_hebbian import SparseLogHebbian

class DualVortexField:
    """
    Dual vortex field dynamics in logarithmic-cylindrical space
    Includes tachyonic tunneling and repulsive forces
    """
    def __init__(self, N: int, device=None):
        """
        Initialize the dual vortex field
        
        Args:
            N: Number of tokens
            device: Computation device
        """
        self.N = N
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Constants (from the whitepaper)
        self.phi = torch.tensor((1 + np.sqrt(5)) / 2, device=self.device)
        self.dt = torch.tensor(self.phi ** (-2), device=self.device)  # default timestep
        self.lambda_cutoff = torch.tensor(self.phi ** 2, device=self.device)  # log-metric cut-off
        self.sigma_gate = torch.tensor(PI / self.phi, device=self.device)  # rotor half-width
        self.eps_freeze = torch.tensor(self.phi ** (-3), device=self.device)  # force & velocity tolerance
        self.z_step = torch.tensor(TAU / (self.phi ** 3), device=self.device)  # rotor increment per dt
        self.alpha_levy = self.phi.item()  # Lévy index for tachyonic jumps
        
        # Token state
        self.tokens = {
            'ln_r': torch.zeros(N, device=self.device),  # natural log of radius
            'theta': torch.zeros(N, device=self.device),  # angle
            'z': torch.zeros(N, device=self.device),  # rotor coordinate (phase)
            'mass': torch.ones(N, device=self.device),  # mass (proportional to ln_r)
            'charge': torch.ones(N, device=self.device),  # charge (for repulsion)
            'frozen': torch.zeros(N, dtype=torch.bool, device=self.device)  # crystallization state
        }
        
        # Create coordinate system
        self.coords = LogCylindricalCoords(device=self.device)
        
        # Create Hebbian network
        self.hebbian = SparseLogHebbian(N, device=self.device)
        
        # Rotor position
        self.z_rotor = torch.tensor(0.0, device=self.device)
        
        # History for visualization
        self.position_history = []
        self.energy_history = []
        self.tachyonic_events = []
        
        # Parameters
        self.m0 = 1.0  # Base mass
        self.k_bound = 1.0  # Boundary force constant
        self.record_interval = 10  # Steps between recordings
    
    def initialize_tokens(self, pattern: str = 'golden_spiral'):
        """
        Initialize token positions
        
        Args:
            pattern: Initialization pattern ('golden_spiral', 'random', 'grid')
        """
        N = self.N
        
        if pattern == 'golden_spiral':
            # Generate golden spiral
            ln_r, theta = self.coords.generate_golden_spiral(N)
            
        elif pattern == 'random':
            # Random positions
            ln_r = torch.randn(N, device=self.device).abs()
            theta = torch.rand(N, device=self.device) * TAU
            
        elif pattern == 'grid':
            # Grid pattern in log-polar space
            grid_size = int(np.sqrt(N))
            actual_N = grid_size ** 2
            
            # Create grid
            ln_r_values = torch.linspace(0, torch.log(torch.tensor(5.0)), grid_size, device=self.device)
            theta_values = torch.linspace(0, TAU, grid_size, device=self.device)
            
            # Create all combinations
            ln_r = torch.zeros(actual_N, device=self.device)
            theta = torch.zeros(actual_N, device=self.device)
            
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    ln_r[idx] = ln_r_values[i]
                    theta[idx] = theta_values[j]
                    idx += 1
            
            # Adjust N if needed
            if actual_N < N:
                # Pad with random positions
                extra = N - actual_N
                ln_r = torch.cat([ln_r, torch.randn(extra, device=self.device).abs()])
                theta = torch.cat([theta, torch.rand(extra, device=self.device) * TAU])
            elif actual_N > N:
                # Truncate
                ln_r = ln_r[:N]
                theta = theta[:N]
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        # Set token positions
        self.tokens['ln_r'] = ln_r
        self.tokens['theta'] = theta
        
        # Set mass based on ln_r (as in whitepaper)
        self.tokens['mass'] = self.m0 * ln_r.clamp(min=EPS)
        
        # All tokens active initially
        self.tokens['frozen'].fill_(False)
        
        # Record initial state
        self.record_state()
    
    def stable_levy(self, size: int, alpha: float = None, scale: float = 1.0) -> torch.Tensor:
        """
        Generate stable Lévy random variables
        Uses Chambers-Mallows-Stuck method
        
        Args:
            size: Number of samples
            alpha: Stability parameter (default: phi)
            scale: Scale parameter
            
        Returns:
            levy: Lévy random variables
        """
        if alpha is None:
            alpha = self.alpha_levy
        
        # Generate uniform random variables
        u = torch.rand(size, device=self.device) * PI
        w = -torch.log(torch.rand(size, device=self.device))  # Exponential with mean 1
        
        # CMS method
        levy = torch.sin(alpha * u) / torch.pow(torch.sin(u), 1/alpha) * \
               torch.pow(torch.sin((1-alpha) * u) / w, (1-alpha)/alpha)
        
        return levy * scale
    
    def compute_repulsion_forces(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute repulsion forces between tokens
        Uses efficient tensor operations with einsum for better performance
        
        Mathematical formulas:
        ln_dist = ln(√((x₂-x₁)² + (y₂-y₁)²))
        F = q₁q₂/r
        F_ln_r = F·d_ln_r/r
        F_θ = F·d_θ/(r·e^ln_r)
        
        Returns:
            F_ln_r: Radial force component (in log space)
            F_theta: Angular force component
        """
        N = self.N
        
        # Initialize force arrays
        F_ln_r = torch.zeros(N, device=self.device)
        F_theta = torch.zeros(N, device=self.device)
        
        # Extract token positions
        ln_r = self.tokens['ln_r']
        theta = self.tokens['theta']
        charge = self.tokens['charge']
        frozen = self.tokens['frozen']
        
        # Compute all pairwise distances and forces using einsum
        # First, expand dimensions for broadcasting
        ln_r_i = ln_r.unsqueeze(1)  # Shape: [N, 1]
        ln_r_j = ln_r.unsqueeze(0)  # Shape: [1, N]
        theta_i = theta.unsqueeze(1)  # Shape: [N, 1]
        theta_j = theta.unsqueeze(0)  # Shape: [1, N]
        
        # Create mask for valid pairs (not self, not frozen)
        self_mask = ~torch.eye(N, dtype=torch.bool, device=self.device)  # Exclude self-interactions
        frozen_mask_i = (~frozen).unsqueeze(1).expand(N, N)  # Only non-frozen i's
        valid_pairs = self_mask & frozen_mask_i
        
        # Compute log distances between all pairs using einsum
        # We're computing this for all pairs simultaneously
        r_i = torch.exp(ln_r_i)  # Shape: [N, 1]
        r_j = torch.exp(ln_r_j)  # Shape: [1, N]
        
        cos_theta_i = torch.cos(theta_i)  # Shape: [N, 1]
        sin_theta_i = torch.sin(theta_i)  # Shape: [N, 1]
        cos_theta_j = torch.cos(theta_j)  # Shape: [1, N]
        sin_theta_j = torch.sin(theta_j)  # Shape: [1, N]
        
        # Compute x and y for all tokens
        x_i = torch.einsum('ij,ij->ij', r_i, cos_theta_i)  # Shape: [N, 1]
        y_i = torch.einsum('ij,ij->ij', r_i, sin_theta_i)  # Shape: [N, 1]
        x_j = torch.einsum('ij,ij->ij', r_j, cos_theta_j)  # Shape: [1, N]
        y_j = torch.einsum('ij,ij->ij', r_j, sin_theta_j)  # Shape: [1, N]
        
        # Compute squared distances
        dx = x_i - x_j  # Shape: [N, N]
        dy = y_i - y_j  # Shape: [N, N]
        dist_squared = dx**2 + dy**2  # Shape: [N, N]
        
        # Compute log distance
        ln_dist = 0.5 * torch.log(dist_squared + EPS)  # Shape: [N, N]
        
        # Create cutoff mask
        cutoff_mask = ln_dist <= torch.log(self.lambda_cutoff)
        
        # Combine masks
        active_mask = valid_pairs & cutoff_mask
        
        # Compute force strengths for valid pairs
        dist = torch.exp(ln_dist)  # Shape: [N, N]
        charge_i = charge.unsqueeze(1)  # Shape: [N, 1]
        charge_j = charge.unsqueeze(0)  # Shape: [1, N]
        force_strength = torch.einsum('ij,ij->ij', charge_i, charge_j) / (dist + EPS)  # Shape: [N, N]
        force_strength = force_strength * active_mask  # Apply mask
        
        # Compute displacement vectors
        d_ln_r = ln_r_j - ln_r_i  # Shape: [N, N]
        d_theta = torch.remainder(theta_j - theta_i + PI, TAU) - PI  # Shape: [N, N]
        
        # Compute force components
        F_ln_r_matrix = force_strength * d_ln_r / (dist + EPS)  # Shape: [N, N]
        F_theta_matrix = force_strength * d_theta / (dist + EPS) / (r_i + EPS)  # Shape: [N, N]
        
        # Sum forces for each token
        F_ln_r = torch.sum(F_ln_r_matrix, dim=1)  # Shape: [N]
        F_theta = torch.sum(F_theta_matrix, dim=1)  # Shape: [N]
        
        return F_ln_r, F_theta
    
    def compute_boundary_forces(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute boundary forces to keep tokens within bounds
        
        Returns:
            F_ln_r: Radial force component
            F_theta: Angular force component
        """
        # Extract token positions
        ln_r = self.tokens['ln_r']
        
        # Initialize force arrays
        F_ln_r = torch.zeros_like(ln_r)
        F_theta = torch.zeros_like(ln_r)
        
        # Boundary in log-radius (from whitepaper: ln(phi^5))
        ln_r_max = torch.log(self.phi ** 5)
        
        # Apply boundary force in radial direction
        too_far = ln_r > ln_r_max
        F_ln_r[too_far] -= self.k_bound * (ln_r[too_far] - ln_r_max)
        
        # Minimum radius boundary
        too_close = ln_r < 0
        F_ln_r[too_close] -= self.k_bound * ln_r[too_close]  # Push outward if negative
        
        return F_ln_r, F_theta
    
    def compute_hebbian_forces(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute forces from Hebbian connections
        
        Returns:
            F_ln_r: Radial force component
            F_theta: Angular force component
        """
        # Extract token positions
        theta = self.tokens['theta']
        
        # Initialize force arrays
        F_ln_r = torch.zeros_like(theta)
        F_theta = torch.zeros_like(theta)
        
        # Compute pitch (preferred angle) via Hebbian connections
        pitch = self.hebbian.compute_hebbian_pitch(theta)
        
        # Compute angle difference (normalized to [-π, π])
        d_theta = torch.remainder(pitch - theta + PI, TAU) - PI
        
        # Apply force toward preferred angle
        # Only angular component is affected
        kappa = 1.0  # Hebbian force constant
        F_theta = -kappa * d_theta
        
        return F_ln_r, F_theta
    
    def integrate_step(self):
        """
        Perform one integration step using Heun's method (predictor-corrector)
        """
        # Extract token state
        ln_r = self.tokens['ln_r']
        theta = self.tokens['theta']
        mass = self.tokens['mass']
        frozen = self.tokens['frozen']
        
        # 1. Update rotor
        self.z_rotor = (self.z_rotor + self.z_step) % TAU
        
        # 2. Determine active tokens (gate mask)
        d_theta_rotor = torch.remainder(theta - self.z_rotor + PI, TAU) - PI
        active = (~frozen) & (torch.abs(d_theta_rotor) < self.sigma_gate)
        
        # 3. Compute forces
        # Repulsion forces
        F_ln_r_rep, F_theta_rep = self.compute_repulsion_forces()
        
        # Boundary forces
        F_ln_r_bound, F_theta_bound = self.compute_boundary_forces()
        
        # Hebbian forces
        F_ln_r_hebb, F_theta_hebb = self.compute_hebbian_forces()
        
        # Combine forces
        F_ln_r = F_ln_r_rep + F_ln_r_bound + F_ln_r_hebb
        F_theta = F_theta_rep + F_theta_bound + F_theta_hebb
        
        # 4. Convert forces to accelerations
        a_ln_r = F_ln_r / (mass + EPS)
        a_theta = F_theta / (mass + EPS)
        
        # 5. Predictor step (Euler)
        ln_r_pred = ln_r + a_ln_r * self.dt
        theta_pred = torch.remainder(theta + a_theta * self.dt, TAU)
        mass_pred = self.m0 * ln_r_pred.clamp(min=EPS)
        
        # 6. Recompute forces at predicted state
        # Temporarily update token state
        orig_tokens = self.tokens.copy()
        
        self.tokens['ln_r'] = ln_r_pred
        self.tokens['theta'] = theta_pred
        self.tokens['mass'] = mass_pred
        
        # Compute forces at predicted state
        F_ln_r_pred, F_theta_pred = self.compute_repulsion_forces()
        F_ln_r_bound_pred, F_theta_bound_pred = self.compute_boundary_forces()
        F_ln_r_hebb_pred, F_theta_hebb_pred = self.compute_hebbian_forces()
        
        # Combine forces at predicted state
        F_ln_r_pred_total = F_ln_r_pred + F_ln_r_bound_pred + F_ln_r_hebb_pred
        F_theta_pred_total = F_theta_pred + F_theta_bound_pred + F_theta_hebb_pred
        
        # Compute accelerations at predicted state
        a_ln_r_pred = F_ln_r_pred_total / (mass_pred + EPS)
        a_theta_pred = F_theta_pred_total / (mass_pred + EPS)
        
        # Restore original token state
        self.tokens = orig_tokens
        
        # 7. Corrector step (average accelerations)
        a_ln_r_avg = 0.5 * (a_ln_r + a_ln_r_pred)
        a_theta_avg = 0.5 * (a_theta + a_theta_pred)
        
        # 8. Final update
        self.tokens['ln_r'] = (ln_r + a_ln_r_avg * self.dt).clamp(min=0)
        self.tokens['theta'] = torch.remainder(theta + a_theta_avg * self.dt, TAU)
        self.tokens['mass'] = self.m0 * self.tokens['ln_r'].clamp(min=EPS)
        self.tokens['z'] = self.z_rotor.expand_as(theta)
        
        # 9. Crystallization & Tunneling
        # Crystallize tokens with small forces
        small_force = (torch.abs(F_ln_r) < self.eps_freeze) & \
                     (torch.abs(F_theta) < self.eps_freeze) & \
                     active
        
        self.tokens['frozen'][small_force] = True
        
        # Tunneling (tachyonic transitions)
        # Occurs when angular velocity dominates radial velocity
        v_ln_r = a_ln_r * self.dt
        v_theta = a_theta * self.dt
        
        # Check for tunneling condition
        ratio = torch.abs(v_theta) / (torch.abs(v_ln_r) + EPS)
        tunnel_mask = (ratio > self.phi) & active & (~frozen)
        
        if tunnel_mask.any():
            # Record tunneling event
            event = {
                'step': len(self.position_history),
                'indices': torch.where(tunnel_mask)[0].cpu().numpy().tolist(),
                'ln_r': self.tokens['ln_r'][tunnel_mask].cpu().numpy().tolist(),
                'theta': self.tokens['theta'][tunnel_mask].cpu().numpy().tolist()
            }
            self.tachyonic_events.append(event)
            
            # Phase flip: θ → θ + π
            self.tokens['theta'][tunnel_mask] = torch.remainder(
                self.tokens['theta'][tunnel_mask] + PI, TAU
            )
            
            # Lévy jump in log-radius
            levy_jumps = self.stable_levy(
                tunnel_mask.sum(), alpha=self.alpha_levy, scale=0.5
            )
            
            self.tokens['ln_r'][tunnel_mask] += levy_jumps.clamp(min=0)
            self.tokens['mass'] = self.m0 * self.tokens['ln_r'].clamp(min=EPS)
        
        # 10. Update Hebbian connections
        self.hebbian.log_update(
            self.tokens['ln_r'], 
            self.tokens['theta'], 
            self.coords, 
            self.dt.item()
        )
    
    def compute_system_energy(self) -> float:
        """
        Compute total system energy
        
        Returns:
            energy: Total system energy
        """
        # Extract token state
        ln_r = self.tokens['ln_r']
        theta = self.tokens['theta']
        charge = self.tokens['charge']
        
        # 1. Repulsion energy
        E_rep = torch.tensor(0.0, device=self.device)
        
        # Sample a subset for efficiency
        sample_size = min(self.N, 100)
        indices = torch.randperm(self.N, device=self.device)[:sample_size]
        
        # Compute repulsion energy (pair-wise)
        for i in range(sample_size):
            idx_i = indices[i]
            for j in range(i+1, sample_size):
                idx_j = indices[j]
                
                # Compute log-distance
                ln_dist = self.coords.log_cartesian_distance(
                    ln_r[idx_i], theta[idx_i], 
                    ln_r[idx_j], theta[idx_j]
                )
                
                # Add to energy if within cutoff
                if ln_dist < torch.log(self.lambda_cutoff):
                    dist = torch.exp(ln_dist)
                    E_rep += charge[idx_i] * charge[idx_j] / dist
        
        # Scale up to full system
        scale = (self.N * (self.N - 1)) / (sample_size * (sample_size - 1))
        E_rep *= scale
        
        # 2. Hebbian energy
        pitch = self.hebbian.compute_hebbian_pitch(theta)
        d_theta = torch.remainder(pitch - theta + PI, TAU) - PI
        E_hebb = 0.5 * torch.sum(d_theta ** 2)
        
        # 3. Boundary energy
        ln_r_max = torch.log(self.phi ** 5)
        too_far = ln_r > ln_r_max
        E_bound = 0.5 * self.k_bound * torch.sum((ln_r[too_far] - ln_r_max) ** 2)
        
        # Total energy
        return (E_rep + E_hebb + E_bound).item()
    
    def record_state(self):
        """Record current state for visualization"""
        # Extract positions
        ln_r = self.tokens['ln_r'].cpu().numpy()
        theta = self.tokens['theta'].cpu().numpy()
        z = self.tokens['z'].cpu().numpy()
        
        # Stack into position array
        positions = np.stack([ln_r, theta, z], axis=1)
        
        # Record
        self.position_history.append(positions)
        
        # Compute and record energy
        energy = self.compute_system_energy()
        self.energy_history.append(energy)
    
    def run_simulation(self, steps: int = 100, record_every: int = 10):
        """
        Run simulation for a number of steps
        
        Args:
            steps: Number of simulation steps
            record_every: How often to record state
        """
        print(f"Running simulation for {steps} steps...")
        start_time = time.time()
        
        # Set record interval
        self.record_interval = record_every
        
        # Run simulation
        for step in range(steps):
            # Perform integration step
            self.integrate_step()
            
            # Record state
            if step % record_every == 0:
                self.record_state()
                
            # Print progress
            if step % (steps // 10) == 0:
                frozen_count = self.tokens['frozen'].sum().item()
                print(f"Step {step}/{steps}: {frozen_count}/{self.N} tokens frozen")
            
            # Check for all frozen
            if self.tokens['frozen'].all():
                print(f"All tokens frozen at step {step}")
                self.record_state()  # Record final state
                break
        
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
        print(f"Recorded {len(self.position_history)} states")
        print(f"Tachyonic events: {len(self.tachyonic_events)}")
    
    def visualize_trajectories(self, save_path: Optional[str] = None):
        """
        Visualize token trajectories
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.position_history:
            print("No trajectory data to visualize")
            return
        
        # Create 3D figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample tokens to visualize (for clarity)
        sample_size = min(20, self.N)
        indices = np.random.choice(self.N, sample_size, replace=False)
        
        # Colors
        colors = plt.cm.viridis(np.linspace(0, 1, sample_size))
        
        # Plot trajectories
        for i, idx in enumerate(indices):
            # Extract trajectory
            trajectory = np.array([step[idx] for step in self.position_history])
            
            # Extract coordinates
            ln_r = trajectory[:, 0]
            theta = trajectory[:, 1]
            z = trajectory[:, 2]
            
            # Convert to Cartesian for visualization
            r = np.exp(ln_r)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Plot trajectory
            ax.plot(x, y, z, c=colors[i], linewidth=1.5, alpha=0.7)
            
            # Mark start and end
            ax.scatter(x[0], y[0], z[0], c=[colors[i]], marker='o', s=30)
            ax.scatter(x[-1], y[-1], z[-1], c=[colors[i]], marker='*', s=80)
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Rotor)')
        ax.set_title('Token Trajectories in Log-Cylindrical Space')
        
        # Add golden spiral reference
        t = np.linspace(0, 4*np.pi, 1000)
        r_spiral = np.exp(t / (2*np.pi))
        x_spiral = r_spiral * np.cos(t)
        y_spiral = r_spiral * np.sin(t)
        z_spiral = np.zeros_like(t)
        
        ax.plot(x_spiral, y_spiral, z_spiral, 'k--', alpha=0.3, linewidth=1)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Trajectories saved to {save_path}")
        else:
            # Try to show, but don't error in non-interactive environments
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plot: {e}")
        
        plt.close()
    
    def visualize_energy(self, save_path: Optional[str] = None):
        """
        Visualize energy history with enhanced visualizations
        similar to relativistic_vortex_energy.png
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.energy_history:
            print("No energy data to visualize")
            return
        
        # Calculate time steps and smoothed energy
        steps = np.arange(len(self.energy_history)) * self.record_interval
        energy = np.array(self.energy_history)
        
        # Calculate energy components if we have history data
        repulsion_energy = []
        hebbian_energy = []
        boundary_energy = []
        
        if len(self.position_history) > 0:
            # Compute approximate energy components
            for i, step in enumerate(self.position_history):
                # Sample a subset of tokens for efficiency
                N = step.shape[0]
                sample_size = min(N, 20)
                indices = np.random.choice(N, sample_size, replace=False)
                
                # Extract token positions
                ln_r_sample = torch.tensor(step[indices, 0], device=self.device)
                theta_sample = torch.tensor(step[indices, 1], device=self.device)
                
                # Compute repulsion energy
                E_rep = 0.0
                for i in range(sample_size):
                    for j in range(i+1, sample_size):
                        # Log-distance
                        ln_dist = self.coords.log_cartesian_distance(
                            ln_r_sample[i], theta_sample[i], 
                            ln_r_sample[j], theta_sample[j]
                        )
                        
                        if ln_dist < torch.log(self.lambda_cutoff):
                            dist = torch.exp(ln_dist)
                            E_rep += 1.0 / dist.item()
                
                # Scale to full system
                E_rep *= (N * (N-1)) / (sample_size * (sample_size-1))
                repulsion_energy.append(E_rep)
                
                # Estimate Hebbian energy
                if hasattr(self, 'hebbian') and self.hebbian is not None:
                    pitch = self.hebbian.compute_hebbian_pitch(theta_sample)
                    d_theta = torch.remainder(pitch - theta_sample + PI, TAU) - PI
                    E_heb = 0.5 * torch.sum(d_theta ** 2).item()
                    hebbian_energy.append(E_heb)
                else:
                    hebbian_energy.append(0.0)
                
                # Estimate boundary energy
                ln_r_max = torch.log(self.phi ** 5)
                too_far = ln_r_sample > ln_r_max
                E_bound = 0.5 * self.k_bound * torch.sum((ln_r_sample[too_far] - ln_r_max) ** 2).item()
                boundary_energy.append(E_bound)
        
        # Use dark background for visualizations
        plt.style.use('dark_background')
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot main energy evolution with gradient fill
        ax1.plot(steps, energy, 'white', linewidth=2.5, label='Total Energy')
        
        # Add gradient fill below the curve
        if len(steps) > 1:
            # Create gradient fill
            min_energy = min(energy)
            energy_norm = (energy - min_energy) / (max(energy) - min_energy + 1e-10)
            
            for i in range(len(steps)-1):
                ax1.fill_between(
                    [steps[i], steps[i+1]], 
                    [min_energy, min_energy], 
                    [energy[i], energy[i+1]], 
                    color=plt.cm.viridis(energy_norm[i]),
                    alpha=0.7
                )
        
        # Plot energy components if available
        if repulsion_energy:
            ax1.plot(steps[:len(repulsion_energy)], repulsion_energy, 'r--', linewidth=1.5, 
                    alpha=0.8, label='Repulsion Energy')
        
        if hebbian_energy:
            ax1.plot(steps[:len(hebbian_energy)], hebbian_energy, 'c--', linewidth=1.5, 
                    alpha=0.8, label='Hebbian Energy')
            
        if boundary_energy:
            ax1.plot(steps[:len(boundary_energy)], boundary_energy, 'y--', linewidth=1.5, 
                    alpha=0.8, label='Boundary Energy')
        
        # Mark tachyonic events with bright flashes
        if self.tachyonic_events:
            event_steps = [event['step'] for event in self.tachyonic_events]
            event_counts = [len(event['indices']) for event in self.tachyonic_events]
            
            # Get corresponding energy values
            event_energies = []
            for step in event_steps:
                energy_idx = step // self.record_interval
                if energy_idx < len(self.energy_history):
                    event_energies.append(self.energy_history[energy_idx])
                else:
                    event_energies.append(None)
            
            # Filter out None values
            valid_indices = [i for i, e in enumerate(event_energies) if e is not None]
            if valid_indices:
                event_steps = [event_steps[i] for i in valid_indices]
                event_counts = [event_counts[i] for i in valid_indices]
                event_energies = [event_energies[i] for i in valid_indices]
                
                # Plot events
                for step, e_val, count in zip(event_steps, event_energies, event_counts):
                    # Add starburst for tachyonic events
                    ax1.scatter(step, e_val, s=count*50, c='gold', marker='*', 
                               edgecolors='white', linewidths=1, zorder=10)
                    
                    # Add vertical line
                    ax1.axvline(x=step, ymin=0, ymax=1, color='gold', alpha=0.3, linewidth=1.5)
                
                # Add legend entry
                ax1.scatter([], [], s=100, c='gold', marker='*', edgecolors='white', 
                           linewidths=1, label='Tachyonic Events')
        
        # Add annotations for key points
        if len(energy) > 5:
            # Find significant energy drops
            energy_changes = np.diff(energy)
            significant_drops = np.where(energy_changes / energy[:-1] < -0.1)[0]
            
            for idx in significant_drops[:3]:  # Limit to 3 annotations
                ax1.annotate(
                    f"Energy drop: {100 * energy_changes[idx] / energy[idx]:.1f}%", 
                    xy=(steps[idx+1], energy[idx+1]),
                    xytext=(steps[idx+1] + 5, energy[idx+1] * 1.5),
                    arrowprops=dict(arrowstyle="->", color="white", alpha=0.7),
                    color="white", fontsize=9
                )
        
        # Configure main plot
        ax1.set_xlabel('Simulation Step', fontsize=12)
        ax1.set_ylabel('Energy', fontsize=12)
        ax1.set_title('Dual Vortex Field Energy Evolution', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.2)
        ax1.set_yscale('log')
        ax1.legend(loc='upper right', fontsize=10)
        
        # Add annotations for convergence criteria
        if len(energy) > 10:
            # Check if converging
            last_section = energy[-10:]
            first_val, last_val = last_section[0], last_section[-1]
            percent_change = 100 * (last_val - first_val) / first_val
            
            if abs(percent_change) < 5:
                stability_text = f"Stable (±{abs(percent_change):.1f}%)"
                color = "green"
            else:
                stability_text = f"Changing ({percent_change:.1f}%)"
                color = "red"
            
            ax1.text(0.02, 0.05, f"Convergence: {stability_text}", 
                    transform=ax1.transAxes, color=color, fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='gray', boxstyle='round'))
        
        # Plot token crystallization in second subplot
        if len(self.position_history) > 0:
            frozen_counts = []
            for i, step in enumerate(self.position_history):
                if i < len(self.energy_history):
                    if i == 0:
                        frozen_counts.append(0)  # Assume none frozen initially
                    else:
                        # Extract frozen state if available
                        if hasattr(self, 'tokens') and 'frozen' in self.tokens:
                            frozen_count = torch.sum(self.tokens['frozen']).item()
                            frozen_counts.append(frozen_count)
                        else:
                            # Estimate from energy reduction
                            frozen_count = int(N * (1 - energy[i] / energy[0]))
                            frozen_counts.append(max(0, frozen_count))
            
            # Ensure we have at least as many points as energy data
            while len(frozen_counts) < len(self.energy_history):
                frozen_counts.append(frozen_counts[-1] if frozen_counts else 0)
                
            # Plot crystallization progress
            ax2.plot(steps, frozen_counts, 'c-', linewidth=2, label='Crystallized Tokens')
            
            # Add percentage on right y-axis
            ax2_percentage = ax2.twinx()
            if N > 0:
                percentage = [100 * count / N for count in frozen_counts]
                ax2_percentage.plot(steps, percentage, 'gold', linewidth=1.5, alpha=0.7)
                ax2_percentage.set_ylabel('Crystallization %', color='gold', fontsize=10)
                ax2_percentage.tick_params(axis='y', colors='gold')
                ax2_percentage.set_ylim(0, 100)
            
            # Mark crystallization threshold
            ax2.axhline(y=N, color='white', linestyle='--', alpha=0.5, 
                       label=f'Total Tokens ({N})')
            
            # Configure subplot
            ax2.set_xlabel('Simulation Step', fontsize=12)
            ax2.set_ylabel('Crystallized Tokens', fontsize=10)
            ax2.set_title('Crystallization Progress', fontsize=12)
            ax2.grid(True, alpha=0.2)
            ax2.legend(loc='upper left', fontsize=10)
            ax2.set_ylim(0, N * 1.1)
            
            # Add tachyonic events here too
            if self.tachyonic_events:
                for step, count in zip(event_steps, event_counts):
                    if step < steps[-1]:
                        # Find the closest matching time step
                        step_idx = np.argmin(np.abs(np.array(steps) - step))
                        if step_idx < len(frozen_counts):
                            # Add tachyonic marker
                            ax2.scatter(step, frozen_counts[step_idx], s=count*30, c='gold', 
                                      marker='*', edgecolors='white', linewidths=1, zorder=10)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Energy plot saved to {save_path}")
        else:
            # Try to show, but don't error in non-interactive environments
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plot: {e}")
        
        plt.close()
    
    def visualize_field(self, save_path: Optional[str] = None, show_tensor_evolution: bool = True):
        """
        Visualize the current field state with advanced visualizations
        similar to relativistic_vortex_physics.png, including tensor matrix evolution
        
        Mathematical representation:
        - Field equations: F = q₁q₂/r² (repulsive force law)
        - Z-gated tokens: |θ - z| < σ_gate = π/φ
        - Tachyonic condition: |vθ/vr| > φ
        
        Args:
            save_path: Optional path to save figure
            show_tensor_evolution: Whether to show tensor matrix evolution visualization
        """
        # Extract token state
        ln_r = self.tokens['ln_r'].cpu().numpy()
        theta = self.tokens['theta'].cpu().numpy()
        frozen = self.tokens['frozen'].cpu().numpy()
        mass = self.tokens['mass'].cpu().numpy()
        
        # Convert to Cartesian
        r = np.exp(ln_r)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Create figure with dark background
        plt.style.use('dark_background')
        
        # Check if we have history for tensor evolution
        has_history = len(self.position_history) >= 3
        
        if has_history and show_tensor_evolution:
            # Create a 3x2 grid for enhanced visualization including tensor evolution
            fig = plt.figure(figsize=(20, 16))
            gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8])
            axes = np.array([
                [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
                [fig.add_subplot(gs[2, :])],
            ], dtype=object)
        else:
            # Standard 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. Plot token positions with velocity vectors
        # Extract velocity components if we have history
        if len(self.position_history) >= 2:
            prev_state = self.position_history[-2]
            curr_state = self.position_history[-1]
            
            # Compute velocities
            dt = self.dt.item() * self.record_interval
            v_ln_r = (curr_state[:, 0] - prev_state[:, 0]) / dt
            v_theta = np.remainder(curr_state[:, 1] - prev_state[:, 1] + np.pi, 2*np.pi) - np.pi
            v_theta = v_theta / dt
            
            # Convert to Cartesian velocities
            v_x = r * (-np.sin(theta) * v_theta + np.cos(theta) * v_ln_r)
            v_y = r * (np.cos(theta) * v_theta + np.sin(theta) * v_ln_r)
            
            # Normalize for better visualization
            velocity_magnitude = np.sqrt(v_x**2 + v_y**2)
            max_velocity = np.max(velocity_magnitude) if velocity_magnitude.size > 0 else 1.0
            v_x = v_x / (max_velocity + 1e-10) * 0.5
            v_y = v_y / (max_velocity + 1e-10) * 0.5
            
            # Plot velocity vectors
            axes[0, 0].quiver(x, y, v_x, v_y, color='cyan', alpha=0.7, width=0.003)
        
        # Plot tokens with enhanced visualization
        scatter = axes[0, 0].scatter(x, y, c=theta, cmap='hsv', s=80*np.sqrt(mass/np.mean(mass)), 
                                   alpha=0.8, edgecolors='white', linewidths=0.5)
        
        # Mark frozen tokens
        if frozen.any():
            axes[0, 0].scatter(x[frozen], y[frozen], s=120*np.sqrt(mass[frozen]/np.mean(mass)), 
                             facecolors='none', edgecolors='gold', linewidths=2, label='Crystallized')
            axes[0, 0].legend(loc='upper right', fontsize=10)
        
        # Add field lines (simplified dual vortex field)
        phi_grid = np.linspace(0, 2*np.pi, 12, endpoint=False)
        r_grid = np.logspace(-1, 1, 5)
        
        for r_val in r_grid:
            x_circle = r_val * np.cos(phi_grid)
            y_circle = r_val * np.sin(phi_grid)
            axes[0, 0].plot(x_circle, y_circle, 'white', alpha=0.15, linewidth=0.5)
            
        for phi_val in phi_grid:
            x_line = np.array([0, 1.5*np.max(r)]) * np.cos(phi_val)
            y_line = np.array([0, 1.5*np.max(r)]) * np.sin(phi_val)
            axes[0, 0].plot(x_line, y_line, 'white', alpha=0.15, linewidth=0.5)
            
        axes[0, 0].set_xlabel('x', fontsize=12)
        axes[0, 0].set_ylabel('y', fontsize=12)
        axes[0, 0].set_title('Dual Vortex Field Dynamics', fontsize=14, fontweight='bold')
        axes[0, 0].set_aspect('equal')
        axes[0, 0].grid(False)
        
        # Add mathematical formula to plot
        formula = r"$F_{ij} = \frac{q_i q_j}{r_{ij}^2} \hat{r}_{ij}$"
        axes[0, 0].text(0.05, 0.05, formula, transform=axes[0, 0].transAxes, 
                      fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.7))
        
        # Add colorbar for theta
        cbar = plt.colorbar(scatter, ax=axes[0, 0], ticks=[0, np.pi, 2*np.pi])
        cbar.ax.set_yticklabels(['0', 'π', '2π'])
        cbar.set_label('θ (Phase)', fontsize=10)
        
        # 2. Plot log-cylindrical space with gate activation
        # Compute gate activation
        z_rotor = self.z_rotor.cpu().item()
        dtheta_rotor = np.remainder(theta - z_rotor + np.pi, 2*np.pi) - np.pi
        gate_active = np.abs(dtheta_rotor) < self.sigma_gate.cpu().item()
        
        # Create colormap for gate activation
        gate_colors = np.zeros(len(theta), dtype=int)
        gate_colors[gate_active & ~frozen] = 1  # Active and not frozen
        gate_colors[gate_active & frozen] = 2   # Active but frozen
        gate_cmap = plt.cm.colors.ListedColormap(['blue', 'green', 'gold'])
        
        scatter = axes[0, 1].scatter(ln_r, theta, c=gate_colors, cmap=gate_cmap, s=60, 
                                   alpha=0.8, edgecolors='white', linewidths=0.5)
        
        # Add rotor line
        axes[0, 1].axhline(y=z_rotor, color='cyan', linestyle='-', linewidth=1.5, 
                         label=f'Z-Rotor: {z_rotor:.2f}')
        
        # Add gate width
        gate_min = z_rotor - self.sigma_gate.cpu().item()
        gate_max = z_rotor + self.sigma_gate.cpu().item()
        axes[0, 1].axhspan(gate_min, gate_max, alpha=0.2, color='cyan', 
                         label=f'Gate Width: σ={self.sigma_gate.cpu().item():.2f}')
        
        axes[0, 1].set_xlabel('ln(r)', fontsize=12)
        axes[0, 1].set_ylabel('θ (radians)', fontsize=12)
        axes[0, 1].set_title('Log-Cylindrical Space with Gate Activation', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=10)
        
        # Add mathematical formula for gate condition
        gate_formula = r"$\text{Gate condition: } |\theta - z| < \sigma_\text{gate} = \pi/\phi$"
        axes[0, 1].text(0.05, 0.05, gate_formula, transform=axes[0, 1].transAxes, 
                      fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.7))
        
        # Plot Hebbian connections
        if self.hebbian.indices:
            # Get pitch
            pitch = self.hebbian.compute_hebbian_pitch(self.tokens['theta']).cpu().numpy()
            
            # Plot tokens colored by pitch
            scatter = axes[1, 0].scatter(x, y, c=pitch, cmap='hsv', s=50, alpha=0.7)
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('y')
            axes[1, 0].set_title('Hebbian Pitch Alignment', fontsize=14, fontweight='bold')
            axes[1, 0].set_aspect('equal')
            axes[1, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 0], label='Pitch')
            
            # Add mathematical formula for Hebbian learning
            hebbian_formula = r"$\text{Hebbian update: } H_{ij} = H_{ij} + \eta \cdot e^{-\ln(d_{ij})}$"
            axes[1, 0].text(0.05, 0.05, hebbian_formula, transform=axes[1, 0].transAxes, 
                          fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.7))
            
            # Compute pitch alignment error
            d_theta = np.remainder(pitch - theta + np.pi, 2*np.pi) - np.pi
            
            # Plot error
            scatter = axes[1, 1].scatter(x, y, c=d_theta, cmap='coolwarm', 
                                       vmin=-np.pi, vmax=np.pi, s=50, alpha=0.7)
            axes[1, 1].set_xlabel('x')
            axes[1, 1].set_ylabel('y')
            axes[1, 1].set_title('Pitch Alignment Error', fontsize=14, fontweight='bold')
            axes[1, 1].set_aspect('equal')
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1], label='Pitch - θ')
            
            # Add mathematical formula for tachyonic condition
            tachyonic_formula = r"$\text{Tachyonic tunneling when: } \left|\frac{v_\theta}{v_r}\right| > \phi$"
            axes[1, 1].text(0.05, 0.05, tachyonic_formula, transform=axes[1, 1].transAxes, 
                          fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.7))
        
        # 3. Add tensor matrix evolution visualization
        if has_history and show_tensor_evolution and len(axes) > 2:
            # Extract last 3 states for tensor visualization
            history_states = self.position_history[-3:]
            
            # Create a grid for the matrix evolution
            ax_matrix = axes[2, 0]  # Bottom panel
            
            # Number of tokens to visualize (limit for clarity)
            vis_tokens = min(20, self.N)
            
            # Prepare matrices for visualization
            matrices = []
            for state in history_states:
                # Extract ln_r and theta
                ln_r_state = state[:vis_tokens, 0]
                theta_state = state[:vis_tokens, 1]
                
                # Compute pairwise distances
                dist_matrix = np.zeros((vis_tokens, vis_tokens))
                for i in range(vis_tokens):
                    for j in range(vis_tokens):
                        if i != j:
                            # Log distance
                            r_i, r_j = np.exp(ln_r_state[i]), np.exp(ln_r_state[j])
                            theta_i, theta_j = theta_state[i], theta_state[j]
                            
                            # Cartesian coordinates
                            x_i, y_i = r_i * np.cos(theta_i), r_i * np.sin(theta_i)
                            x_j, y_j = r_j * np.cos(theta_j), r_j * np.sin(theta_j)
                            
                            # Euclidean distance
                            dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                            dist_matrix[i, j] = dist
                
                matrices.append(dist_matrix)
            
            # Plot matrices side by side
            n_matrices = len(matrices)
            width_ratios = [1] * n_matrices
            
            # Create subplots within the matrix axis
            gs_matrices = plt.GridSpec(1, n_matrices, width_ratios=width_ratios)
            gs_matrices.update(top=0.9, bottom=0.1, left=0.1, right=0.9)
            
            # Plot each matrix
            for i, matrix in enumerate(matrices):
                # Create subplot
                ax_sub = fig.add_subplot(gs_matrices[0, i])
                
                # Normalize matrix for visualization
                matrix_norm = matrix / (matrix.max() + 1e-10)
                
                # Plot matrix
                im = ax_sub.imshow(matrix_norm, cmap='viridis', 
                                 interpolation='nearest', aspect='auto')
                
                # Set title
                step = len(self.position_history) - n_matrices + i
                ax_sub.set_title(f"Step {step*self.record_interval}")
                
                # Set labels
                if i == 0:
                    ax_sub.set_ylabel("Token Index")
                ax_sub.set_xlabel("Token Index")
                
                # Add colorbar
                plt.colorbar(im, ax=ax_sub, orientation='horizontal', pad=0.05, 
                           label="Normalized Distance")
            
            # Set overall title for the matrix section
            fig.text(0.5, 0.02, "Tensor Matrix Evolution: Token Pairwise Distances", 
                    ha='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Field visualization saved to {save_path}")
        else:
            # Try to show, but don't error in non-interactive environments
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plot: {e}")
        
        plt.close()
    
    def create_animation(self, save_path: Optional[str] = None):
        """
        Create animation of token evolution
        
        Args:
            save_path: Optional path to save animation
        """
        if not self.position_history:
            print("No trajectory data to animate")
            return
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get full trajectory data
        trajectories = []
        for i in range(self.N):
            trajectory = np.array([step[i] for step in self.position_history])
            trajectories.append(trajectory)
        
        # Convert to Cartesian
        x_data = []
        y_data = []
        z_data = []
        
        for trajectory in trajectories:
            ln_r = trajectory[:, 0]
            theta = trajectory[:, 1]
            z = trajectory[:, 2]
            
            r = np.exp(ln_r)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            x_data.append(x)
            y_data.append(y)
            z_data.append(z)
        
        # Initial scatter plot
        scatter = ax.scatter(
            [x[0] for x in x_data],
            [y[0] for y in y_data],
            [z[0] for z in z_data],
            c=np.arange(self.N), cmap='viridis', s=30, alpha=0.7
        )
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Rotor)')
        ax.set_title('Token Evolution')
        
        # Set fixed limits
        x_all = np.concatenate(x_data)
        y_all = np.concatenate(y_data)
        z_all = np.concatenate(z_data)
        
        max_range = max(np.ptp(x_all), np.ptp(y_all), np.ptp(z_all))
        mid_x = (np.max(x_all) + np.min(x_all)) / 2
        mid_y = (np.max(y_all) + np.min(y_all)) / 2
        mid_z = (np.max(z_all) + np.min(z_all)) / 2
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # Animation update function
        def update(frame):
            scatter._offsets3d = (
                [x[frame] for x in x_data],
                [y[frame] for y in y_data],
                [z[frame] for z in z_data]
            )
            ax.view_init(elev=30, azim=frame/2)  # Rotate view
            ax.set_title(f'Token Evolution (Step {frame*self.record_interval})')
            return scatter,
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(self.position_history),
            interval=100, blit=False
        )
        
        # Save or show
        if save_path:
            anim.save(save_path, fps=10, dpi=100)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

# Example usage
if __name__ == "__main__":
    print("Testing Dual Vortex Field with Tachyonic Tunneling")
    
    # Use CPU for testing to avoid numpy conversion issues
    cpu_device = torch.device('cpu')
    
    # Create field
    N = 50  # Use smaller N for testing
    field = DualVortexField(N, device=cpu_device)
    
    # Initialize tokens
    field.initialize_tokens(pattern='golden_spiral')
    
    # Run simulation
    field.run_simulation(steps=100, record_every=5)
    
    # Visualize results
    field.visualize_trajectories(save_path="dual_vortex_trajectories.png")
    field.visualize_energy(save_path="dual_vortex_energy.png")
    field.visualize_field(save_path="dual_vortex_field.png")
    
    # Create animation (optional - can be slow)
    # field.create_animation(save_path="dual_vortex_animation.mp4")
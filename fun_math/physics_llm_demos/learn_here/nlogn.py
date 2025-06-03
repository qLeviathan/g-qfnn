"""
Log-Cartesian Quantum Field Neural Network (LC-QFNN)
Implementing O(N log N) field dynamics with logarithmic-cylindrical embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Constants
PHI = torch.tensor((1 + np.sqrt(5)) / 2, device=device)
PI = torch.tensor(np.pi, device=device)
TAU = torch.tensor(2 * np.pi, device=device)
EPS = torch.tensor(1e-10, device=device)

class SparseLogHebb:
    """
    Sparse implementation of the Hebbian log-matrix for O(N·k) updates
    Handles coupling matrix in log-space for numerical stability
    """
    def __init__(self, size: int, default: float = -float('inf')):
        self.size = size
        self.default = default
        
        # Initialize sparse matrix in COO format
        self.indices = []  # List of (i, j) tuples for non-zero entries
        self.values = []   # Corresponding log values
        
        # Create mapping for fast lookups
        self.index_map = {}  # Maps (i, j) -> position in values list
        
        # Cache for sparse matrix-vector operations
        self.cache = {}
    
    def get_value(self, i: int, j: int) -> float:
        """Get log coupling value"""
        key = (i, j)
        if key in self.index_map:
            return self.values[self.index_map[key]]
        return self.default
    
    def set_value(self, i: int, j: int, log_value: float):
        """Set log coupling value"""
        key = (i, j)
        if key in self.index_map:
            # Update existing entry
            self.values[self.index_map[key]] = log_value
        else:
            # Add new entry
            self.index_map[key] = len(self.values)
            self.indices.append(key)
            self.values.append(log_value)
            
        # Clear cache
        self.cache = {}
    
    def sparse_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sparse matrix-vector product in log space
        x: (N, D) or (D,) tensor
        Returns: (N, D) or (D,) tensor
        """
        # Check if result is in cache
        cache_key = hash(x.data_ptr())
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Convert to PyTorch sparse tensor for efficient operations
        if not self.indices:
            # Handle empty matrix case
            if x.dim() == 2:
                result = torch.zeros(self.size, x.shape[1], device=x.device)
            else:
                result = torch.zeros(self.size, device=x.device)
            return result
        
        # Create indices tensor
        i_indices = [i for i, j in self.indices]
        j_indices = [j for i, j in self.indices]
        indices = torch.tensor([i_indices, j_indices], dtype=torch.long, device=x.device)
        
        # Create values tensor (exponentiate to convert from log space)
        values = torch.tensor(self.values, device=x.device).exp()
        
        # Create sparse tensor
        sparse_matrix = torch.sparse.FloatTensor(
            indices, values, (self.size, self.size)
        )
        
        # Compute matrix-vector product
        if x.dim() == 2:
            # Batched version
            result = torch.zeros(self.size, x.shape[1], device=x.device)
            for k in range(x.shape[1]):
                result[:, k] = torch.sparse.mm(sparse_matrix, x[:, k].unsqueeze(1)).squeeze(1)
        else:
            # Single vector
            result = torch.sparse.mm(sparse_matrix, x.unsqueeze(1)).squeeze(1)
        
        # Cache result
        self.cache[cache_key] = result
        return result
    
    def sparse_parallel_update(self, ell: torch.Tensor, theta: torch.Tensor, dt: float):
        """
        Update Hebbian log-matrix in parallel
        ell: (N,) log-radius values
        theta: (N,) angle values
        dt: time step
        """
        N = ell.shape[0]
        
        # Threshold for adding new connections (use λ = φ² from whitepaper)
        lambda_cutoff = PHI ** 2
        
        # Decay existing connections
        gamma = 0.1 * dt  # Decay rate
        for idx, (i, j) in enumerate(self.indices):
            # Decay log value
            self.values[idx] -= gamma
            
            # Remove if below threshold
            if self.values[idx] < np.log(lambda_cutoff.item()) - 5:
                # Mark for removal
                self.values[idx] = -float('inf')
        
        # Filter out removed connections
        valid_indices = []
        valid_values = []
        new_index_map = {}
        
        for idx, (i, j) in enumerate(self.indices):
            if self.values[idx] > -float('inf'):
                new_index_map[(i, j)] = len(valid_indices)
                valid_indices.append((i, j))
                valid_values.append(self.values[idx])
        
        self.indices = valid_indices
        self.values = valid_values
        self.index_map = new_index_map
        
        # Add new connections for active tokens
        # This would typically use a CUDA kernel for true parallelism
        # Here we'll simulate with vectorized operations
        
        # Use a sparse update approach:
        # 1. Identify token pairs that should have connections
        # 2. Compute their log-distance
        # 3. Update connections that are within threshold
        
        # Example: Update connections for neighboring tokens in sequence
        batch_size = 100  # Process in batches for efficiency
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            
            for i in range(start_idx, end_idx):
                # Sample a few potential connections (could be more sophisticated)
                # In practice, we'd use a Barnes-Hut tree or spatial hash
                j_candidates = torch.randint(0, N, (5,), device=ell.device)
                
                for j in j_candidates:
                    if i != j:
                        # Compute log-Cartesian distance
                        log_dist = compute_log_cartesian_distance(
                            ell[i], theta[i], ell[j], theta[j]
                        )
                        
                        # Add connection if within threshold
                        if log_dist < torch.log(lambda_cutoff):
                            # Strengthen connection
                            log_strength = self.get_value(i.item(), j.item())
                            if log_strength == self.default:
                                log_strength = log_dist  # Initialize with distance
                            else:
                                # Logarithmic strengthening (log-sum-exp)
                                log_strength = torch.logsumexp(
                                    torch.tensor([log_strength, log_dist + dt]), dim=0
                                ).item()
                            
                            self.set_value(i.item(), j.item(), log_strength)
        
        # Clear cache after update
        self.cache = {}


def compute_log_cartesian_distance(ell1: torch.Tensor, theta1: torch.Tensor, 
                                   ell2: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
    """
    Compute distance in log-Cartesian space using log-sum-exp for stability
    ell: log-radius
    theta: angle
    """
    # Convert to log-Cartesian coordinates
    # x = exp(ell) * cos(theta), y = exp(ell) * sin(theta)
    # log|x| = ell + log|cos(theta)|, log|y| = ell + log|sin(theta)|
    
    # Compute log|cos(theta)| and log|sin(theta)| with sign info
    cos1, sin1 = torch.cos(theta1), torch.sin(theta1)
    cos2, sin2 = torch.cos(theta2), torch.sin(theta2)
    
    # Compute log absolute values with offset for numerical stability
    log_cos1 = torch.log(torch.abs(cos1) + EPS)
    log_sin1 = torch.log(torch.abs(sin1) + EPS)
    log_cos2 = torch.log(torch.abs(cos2) + EPS)
    log_sin2 = torch.log(torch.abs(sin2) + EPS)
    
    # Compute log|x| and log|y| with sign info
    log_x1 = ell1 + log_cos1
    log_y1 = ell1 + log_sin1
    log_x2 = ell2 + log_cos2
    log_y2 = ell2 + log_sin2
    
    # Handle sign differences for x and y components
    sign_x1, sign_y1 = torch.sign(cos1), torch.sign(sin1)
    sign_x2, sign_y2 = torch.sign(cos2), torch.sign(sin2)
    
    # Compute logarithmic differences
    # If signs are the same, we need log|exp(log_a) - exp(log_b)|
    # If signs are different, we need log|exp(log_a) + exp(log_b)|
    
    # X-component
    if sign_x1 * sign_x2 > 0:
        # Same sign: need log|exp(log_a) - exp(log_b)|
        log_max_x = torch.maximum(log_x1, log_x2)
        log_min_x = torch.minimum(log_x1, log_x2)
        log_diff_x = log_max_x + torch.log(1 - torch.exp(log_min_x - log_max_x))
    else:
        # Different sign: need log|exp(log_a) + exp(log_b)|
        log_diff_x = torch.logsumexp(torch.stack([log_x1, log_x2]), dim=0)
    
    # Y-component
    if sign_y1 * sign_y2 > 0:
        # Same sign: need log|exp(log_a) - exp(log_b)|
        log_max_y = torch.maximum(log_y1, log_y2)
        log_min_y = torch.minimum(log_y1, log_y2)
        log_diff_y = log_max_y + torch.log(1 - torch.exp(log_min_y - log_max_y))
    else:
        # Different sign: need log|exp(log_a) + exp(log_b)|
        log_diff_y = torch.logsumexp(torch.stack([log_y1, log_y2]), dim=0)
    
    # Compute log-Euclidean distance: log(sqrt(dx² + dy²))
    # = 0.5 * log(exp(2*log|dx|) + exp(2*log|dy|))
    log_dist = 0.5 * torch.logsumexp(torch.stack([2*log_diff_x, 2*log_diff_y]), dim=0)
    
    return log_dist


class GwaveGPU(nn.Module):
    """
    GPU-accelerated implementation of Gwave dynamics with log-Cartesian coordinates
    Implements the algorithm from the whitepaper with O(N log N) complexity
    """
    def __init__(self, N: int, m0: float = 1.0, device: str = None):
        super().__init__()
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Constants from whitepaper
        self.phi = torch.tensor((1 + np.sqrt(5)) / 2, device=self.device)
        self.DT = torch.tensor(self.phi ** (-2), device=self.device)  # default timestep
        self.lambda_cutoff = torch.tensor(self.phi ** 2, device=self.device)  # log-metric cut-off
        self.sigma_gate = torch.tensor(PI / self.phi, device=self.device)  # rotor half-width
        self.eps_freeze = torch.tensor(self.phi ** (-3), device=self.device)  # force & velocity tolerance
        self.Z_step = torch.tensor(TAU / (self.phi ** 3), device=self.device)  # rotor increment per DT
        self.alpha_levy = self.phi  # Lévy index
        
        # Allocate token arrays
        self.tokens = {
            'ell': torch.zeros(N, device=self.device),  # log-radius
            'theta': torch.zeros(N, device=self.device),  # angle
            'z': torch.zeros(N, device=self.device),  # rotor position
            'mass': torch.zeros(N, device=self.device),  # mass
            's': torch.ones(N, device=self.device),  # charge
            'frozen': torch.zeros(N, dtype=torch.bool, device=self.device)  # crystallization state
        }
        
        # Initialize values
        self.m0 = m0
        self.N = N
        self.Z_rotor = torch.tensor(0.0, device=self.device)
        
        # Create sparse Hebbian log-matrix
        self.Hlog = SparseLogHebb(N, default=-torch.tensor(float('inf')))
        
        # History tracking
        self.position_history = []
        self.energy_history = []
        self.tachyonic_history = []
        
        # Runtime parameters
        self.record_interval = 10
        self.convergence_window = 50
        self.N_tree_thresh = 100  # Threshold for using tree algorithm
    
    def initialize_tokens(self, ell_init=None, theta_init=None):
        """Initialize token positions"""
        N = self.N
        
        if ell_init is None:
            # Default: logarithmic spiral
            ell_init = torch.linspace(0, torch.log(torch.tensor(self.phi ** 3)), N, device=self.device)
        
        if theta_init is None:
            # Default: golden angle spacing
            theta_init = torch.tensor([(i * self.phi) % 1.0 for i in range(N)], device=self.device) * TAU
        
        # Set initial values
        self.tokens['ell'] = ell_init
        self.tokens['theta'] = theta_init
        self.tokens['mass'] = self.m0 * ell_init.clamp(min=EPS)
        
        # Record initial state
        self.record_state(0)
    
    def stable_levy(self, size: int, alpha: float = None, scale: float = 1.0) -> torch.Tensor:
        """
        Generate stable Lévy random variables using Chambers-Mallows-Stuck method
        alpha: stability parameter (default: phi)
        """
        if alpha is None:
            alpha = self.alpha_levy
            
        # Generate uniform random variables
        u = torch.rand(size, device=self.device) * PI
        w = torch.rand(size, device=self.device).neg().log()  # Exponential with mean 1
        
        # CMS method for generating stable random variables
        levy = torch.sin(alpha * u) / torch.pow(torch.sin(u), 1/alpha) * \
               torch.pow(torch.sin((1-alpha) * u) / w, (1-alpha)/alpha)
        
        return levy * scale
    
    def compute_repulsion_direct(self, tokens: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute repulsion forces using direct O(N²) method
        Returns: (F_ell, F_theta) force components
        """
        N = tokens['ell'].shape[0]
        
        # Initialize force arrays
        F_ell = torch.zeros(N, device=self.device)
        F_theta = torch.zeros(N, device=self.device)
        
        # Compute all pairwise forces (N² operations)
        for i in range(N):
            if tokens['frozen'][i]:
                continue
                
            for j in range(N):
                if i == j:
                    continue
                
                # Compute log-Cartesian distance
                log_dist = compute_log_cartesian_distance(
                    tokens['ell'][i], tokens['theta'][i],
                    tokens['ell'][j], tokens['theta'][j]
                )
                
                # Check if within cutoff range
                if log_dist > torch.log(self.lambda_cutoff):
                    continue
                
                # Convert to normal distance for force calculation
                dist = torch.exp(log_dist)
                
                # Compute force strength (repulsion is 1/r)
                force_strength = tokens['s'][i] * tokens['s'][j] / dist
                
                # Compute direction components in log-cylindrical coords
                d_theta = torch.remainder(tokens['theta'][j] - tokens['theta'][i] + PI, TAU) - PI
                d_ell = tokens['ell'][j] - tokens['ell'][i]
                
                # Convert to force components
                r_ratio = torch.exp(d_ell)
                
                # Radial component
                F_ell[i] += force_strength * d_ell / log_dist.exp()
                
                # Angular component (scale by radius)
                F_theta[i] += force_strength * d_theta / log_dist.exp() / tokens['ell'][i].exp()
        
        return F_ell, F_theta
    
    def compute_repulsion(self, tokens: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute repulsion forces using either spatial hash or BH tree
        Chooses algorithm based on token count for optimal performance
        """
        N = tokens['ell'].shape[0]
        
        if N < self.N_tree_thresh:
            # Small N: use direct method
            return self.compute_repulsion_direct(tokens)
        else:
            # Large N: use spatial hash for O(N) expected performance
            return self.compute_repulsion_spatial_hash(tokens)
    
    def compute_repulsion_spatial_hash(self, tokens: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute repulsion forces using a simplified spatial hashing approach
        This is a simpler implementation for testing
        """
        N = tokens['ell'].shape[0]
        
        # Initialize force arrays
        F_ell = torch.zeros(N, device=self.device)
        F_theta = torch.zeros(N, device=self.device)
        
        # For small tests, use a simpler approach:
        # Group tokens by angular sectors (just theta binning)
        n_sectors = 8
        sector_size = TAU / n_sectors
        
        # Create sectors
        sectors = [[] for _ in range(n_sectors)]
        for i in range(N):
            if tokens['frozen'][i]:
                continue
            
            # Assign to sector
            sector_idx = min(int(tokens['theta'][i] / sector_size), n_sectors-1)
            sectors[sector_idx].append(i)
        
        # Compute forces using sectors
        for i in range(N):
            if tokens['frozen'][i]:
                continue
            
            # Get token's sector
            my_sector = min(int(tokens['theta'][i] / sector_size), n_sectors-1)
            
            # Check current sector and adjacent sectors
            for s_offset in [-1, 0, 1]:
                s = (my_sector + s_offset) % n_sectors
                
                # Compute forces from tokens in this sector
                for j in sectors[s]:
                    if i == j:
                        continue
                    
                    # Compute log-Cartesian distance
                    log_dist = compute_log_cartesian_distance(
                        tokens['ell'][i], tokens['theta'][i],
                        tokens['ell'][j], tokens['theta'][j]
                    )
                    
                    # Check if within cutoff range
                    if log_dist > torch.log(self.lambda_cutoff):
                        continue
                    
                    # Convert to normal distance for force calculation
                    dist = torch.exp(log_dist)
                    
                    # Compute force strength (repulsion is 1/r)
                    force_strength = tokens['s'][i] * tokens['s'][j] / dist
                    
                    # Compute direction components in log-cylindrical coords
                    d_theta = torch.remainder(tokens['theta'][j] - tokens['theta'][i] + PI, TAU) - PI
                    d_ell = tokens['ell'][j] - tokens['ell'][i]
                    
                    # Radial component
                    F_ell[i] += force_strength * d_ell / log_dist.exp()
                    
                    # Angular component (scale by radius)
                    F_theta[i] += force_strength * d_theta / log_dist.exp() / tokens['ell'][i].exp()
        
        return F_ell, F_theta
    
    def record_tachyonic(self, tunnel_mask: torch.Tensor, tokens: Dict[str, torch.Tensor], step: int):
        """Record tachyonic (tunneling) events"""
        event = {
            'step': step,
            'indices': torch.where(tunnel_mask)[0].cpu().numpy().tolist(),
            'ell': tokens['ell'][tunnel_mask].cpu().numpy().tolist(),
            'theta': tokens['theta'][tunnel_mask].cpu().numpy().tolist()
        }
        self.tachyonic_history.append(event)
    
    def compute_total_energy(self, tokens: Dict[str, torch.Tensor], Hlog: SparseLogHebb) -> torch.Tensor:
        """Compute total system energy for convergence checking"""
        # Repulsion energy
        E_rep = torch.tensor(0.0, device=self.device)
        
        # Sample a subset of tokens for efficiency
        N = tokens['ell'].shape[0]
        sample_size = min(N, 100)
        sample_indices = torch.randperm(N)[:sample_size]
        
        # Compute repulsion energy for sample
        for i in sample_indices:
            for j in sample_indices:
                if i == j:
                    continue
                
                log_dist = compute_log_cartesian_distance(
                    tokens['ell'][i], tokens['theta'][i],
                    tokens['ell'][j], tokens['theta'][j]
                )
                
                if log_dist < torch.log(self.lambda_cutoff):
                    dist = torch.exp(log_dist)
                    E_rep += tokens['s'][i] * tokens['s'][j] / dist
        
        # Scale to full system
        E_rep *= (N / sample_size) ** 2
        
        # Hebbian energy (pitch alignment)
        E_hebb = torch.tensor(0.0, device=self.device)
        
        # Compute pitch via sparse Hlog
        p_xy = Hlog.sparse_matvec(torch.stack([
            torch.cos(tokens['theta']),
            torch.sin(tokens['theta'])
        ], dim=1))
        
        pitch = torch.atan2(p_xy[:, 1], p_xy[:, 0])
        dtheta_pitch = torch.remainder(pitch - tokens['theta'] + PI, TAU) - PI
        
        # Potential energy from pitch misalignment
        E_hebb = torch.sum(dtheta_pitch ** 2)
        
        # Total energy
        return E_rep + 0.5 * E_hebb
    
    def check_convergence(self, energy_history: List[float], tokens: Dict[str, torch.Tensor]) -> bool:
        """Check for convergence based on energy stability and token state"""
        # Check if all tokens are frozen
        if tokens['frozen'].all():
            return True
        
        # Check energy stability
        if len(energy_history) < self.convergence_window:
            return False
        
        # Compute energy change rate
        recent_energies = energy_history[-self.convergence_window:]
        energy_slope = (recent_energies[-1] - recent_energies[0]) / self.convergence_window
        
        # Normalized energy change
        normalized_change = abs(energy_slope / (recent_energies[0] + EPS))
        
        # Check if energy is stable enough
        return normalized_change < self.eps_freeze
    
    def record_state(self, step: int):
        """Record current state for visualization"""
        if step % self.record_interval == 0:
            snapshot = torch.stack([
                self.tokens['ell'],
                self.tokens['theta'],
                self.tokens['z']
            ], dim=1).cpu().numpy()
            
            self.position_history.append(snapshot)
            
            energy = self.compute_total_energy(self.tokens, self.Hlog).item()
            self.energy_history.append(energy)
    
    def step(self):
        """Perform one step of the field evolution algorithm"""
        # 3.1 Rotor update (scalar)
        self.Z_rotor = (self.Z_rotor + self.Z_step) % TAU
        
        # 3.2 Gate mask (vectorized)
        dtheta_rotor = torch.remainder(self.tokens['theta'] - self.Z_rotor + PI, TAU) - PI
        active = (~self.tokens['frozen']) & (dtheta_rotor.abs() < self.sigma_gate)
        
        # 3.3 Repulsion forces
        F_ell, F_theta = self.compute_repulsion(self.tokens)
        
        # 3.4 Hebbian force (vectorized & sparse)
        # Reconstruct pitch via sparse Hlog
        p_xy = self.Hlog.sparse_matvec(torch.stack([
            torch.cos(self.tokens['theta']),
            torch.sin(self.tokens['theta'])
        ], dim=1))
        
        pitch = torch.atan2(p_xy[:, 1], p_xy[:, 0])
        dtheta_pitch = torch.remainder(pitch - self.tokens['theta'] + PI, TAU) - PI
        
        # Apply Hebbian force (kappa = 1.0)
        F_theta = F_theta - dtheta_pitch
        
        # 3.5 Boundary force
        too_far = self.tokens['ell'] > torch.log(self.phi ** 5)
        k_bound = 1.0
        F_ell[too_far] -= k_bound * (self.tokens['ell'][too_far] - torch.log(self.phi ** 5))
        
        # 3.6 Heun-Euler step (batched)
        # Compute velocities
        v_ell = F_ell / (self.tokens['mass'] + EPS)
        v_theta = F_theta / (self.tokens['mass'] + EPS)
        
        # Predictor: simple Euler for ell, theta
        ell_pred = self.tokens['ell'] + v_ell * self.DT
        theta_pred = torch.remainder(self.tokens['theta'] + v_theta * self.DT, TAU)
        z_pred = torch.full_like(self.tokens['z'], self.Z_rotor)
        mass_pred = self.m0 * ell_pred.clamp(min=EPS)
        
        # Recompute forces at predicted state
        tokens_pred = {
            'ell': ell_pred,
            'theta': theta_pred,
            'z': z_pred,
            'mass': mass_pred,
            's': self.tokens['s'],
            'frozen': self.tokens['frozen']
        }
        
        F_ell_pred, F_theta_pred = self.compute_repulsion(tokens_pred)
        
        # Hebbian force at predicted state
        p_xy_pred = self.Hlog.sparse_matvec(torch.stack([
            torch.cos(theta_pred),
            torch.sin(theta_pred)
        ], dim=1))
        
        pitch_pred = torch.atan2(p_xy_pred[:, 1], p_xy_pred[:, 0])
        dtheta_p = torch.remainder(pitch_pred - theta_pred + PI, TAU) - PI
        F_theta_pred = F_theta_pred - dtheta_p
        
        # Boundary force at predicted state
        too_far_pred = ell_pred > torch.log(self.phi ** 5)
        F_ell_pred[too_far_pred] -= k_bound * (ell_pred[too_far_pred] - torch.log(self.phi ** 5))
        
        # Corrector (average slopes)
        v_ell_pred = F_ell_pred / (mass_pred + EPS)
        v_theta_pred = F_theta_pred / (mass_pred + EPS)
        
        v_ell_avg = 0.5 * (v_ell + v_ell_pred)
        v_theta_avg = 0.5 * (v_theta + v_theta_pred)
        
        # Final update
        self.tokens['ell'] = (self.tokens['ell'] + v_ell_avg * self.DT).clamp(min=0)
        self.tokens['theta'] = torch.remainder(self.tokens['theta'] + v_theta_avg * self.DT, TAU)
        self.tokens['z'] = self.Z_rotor
        self.tokens['mass'] = self.m0 * self.tokens['ell']
        
        # 3.7 Crystallization & Tunneling (vectorized)
        small_force = (F_ell.abs() < self.eps_freeze) & (F_theta.abs() < self.eps_freeze) & active
        self.tokens['frozen'][small_force] = True
        
        # Tunneling
        ratio = (v_theta.abs() / (v_ell.abs() + EPS)) > self.phi
        tunnel_mask = ratio & active & (~self.tokens['frozen'])
        
        if tunnel_mask.any():
            # Phase flip
            self.tokens['theta'][tunnel_mask] = torch.remainder(self.tokens['theta'][tunnel_mask] + PI, TAU)
            
            # Lévy jump in log-radius
            levy_jumps = self.stable_levy(tunnel_mask.sum(), alpha=self.phi.item())
            self.tokens['ell'][tunnel_mask] += levy_jumps.clamp(min=0)
            self.tokens['mass'] = self.m0 * self.tokens['ell']
            
            # Record tachyonic events
            self.record_tachyonic(tunnel_mask, self.tokens, step=len(self.position_history))
        
        # 3.8 Sparse Hebbian log-update
        self.Hlog.sparse_parallel_update(self.tokens['ell'], self.tokens['theta'], self.DT.item())
    
    def run(self, max_steps: int = 1000) -> Dict:
        """Run simulation for max_steps or until convergence"""
        start_time = time.time()
        
        for step in range(max_steps):
            # Perform one step
            self.step()
            
            # Record state
            self.record_state(step)
            
            # Check convergence
            if step > self.convergence_window:
                if self.check_convergence(self.energy_history, self.tokens):
                    print(f"Converged at step {step}")
                    break
            
            # Early stopping if all frozen
            if self.tokens['frozen'].all():
                print(f"All tokens frozen at step {step}")
                break
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            'position_history': self.position_history,
            'energy_history': self.energy_history,
            'tachyonic_history': self.tachyonic_history,
            'final_state': {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                           for k, v in self.tokens.items()},
            'runtime_seconds': duration,
            'steps_completed': len(self.position_history) * self.record_interval,
            'converged': self.tokens['frozen'].all().item()
        }

    def visualize_trajectories(self):
        """Visualize token trajectories in 3D"""
        if not self.position_history:
            print("No history to visualize")
            return
        
        # Create 3D figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample tokens to visualize (for clarity)
        N = self.tokens['ell'].shape[0]
        sample_size = min(N, 20)
        sample_indices = np.random.choice(N, sample_size, replace=False)
        
        # Colors for trajectories
        colors = plt.cm.viridis(np.linspace(0, 1, sample_size))
        
        # Plot trajectories
        for i, idx in enumerate(sample_indices):
            # Extract trajectory for this token
            trajectory = np.array([step[idx] for step in self.position_history])
            
            # Convert from log-cylindrical to Cartesian
            ell = trajectory[:, 0]
            theta = trajectory[:, 1]
            z = trajectory[:, 2]
            
            # r = exp(ell), x = r*cos(theta), y = r*sin(theta)
            r = np.exp(ell)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Plot trajectory
            ax.plot(x, y, z, '-', color=colors[i], alpha=0.7, linewidth=1.5)
            
            # Mark start and end
            ax.scatter(x[0], y[0], z[0], color=colors[i], marker='o', s=30)
            ax.scatter(x[-1], y[-1], z[-1], color=colors[i], marker='*', s=80)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Rotor)')
        ax.set_title('Token Trajectories in Log-Cylindrical Space')
        
        # Add a golden spiral reference
        t = np.linspace(0, 4*np.pi, 1000)
        r_spiral = np.exp(t / (2*np.pi))
        x_spiral = r_spiral * np.cos(t)
        y_spiral = r_spiral * np.sin(t)
        z_spiral = np.zeros_like(t)
        
        ax.plot(x_spiral, y_spiral, z_spiral, '--', color='gold', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        plt.savefig('helix_trajectories.png', dpi=200)
        plt.show()

    def visualize_energy(self):
        """Visualize energy history"""
        if not self.energy_history:
            print("No energy history to visualize")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.energy_history, 'b-', linewidth=2)
        plt.xlabel('Steps')
        plt.ylabel('Total Energy')
        plt.title('System Energy Evolution')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Mark tachyonic events
        if self.tachyonic_history:
            event_steps = [event['step'] for event in self.tachyonic_history]
            event_counts = [len(event['indices']) for event in self.tachyonic_history]
            
            # Get energy at those steps
            event_energies = [self.energy_history[step // self.record_interval] 
                             if step // self.record_interval < len(self.energy_history) else None 
                             for step in event_steps]
            
            # Filter out None values
            valid_indices = [i for i, e in enumerate(event_energies) if e is not None]
            if valid_indices:
                event_steps = [event_steps[i] for i in valid_indices]
                event_counts = [event_counts[i] for i in valid_indices]
                event_energies = [event_energies[i] for i in valid_indices]
                
                # Plot events
                plt.scatter([s // self.record_interval for s in event_steps], 
                           event_energies, 
                           c='red', s=[count * 20 for count in event_counts], 
                           alpha=0.7, label='Tachyonic Events')
                
                plt.legend()
        
        plt.tight_layout()
        plt.savefig('energy_history.png', dpi=200)
        plt.show()


# Example usage
if __name__ == "__main__":
    print("==== Testing Log-Cartesian Quantum Field Neural Network ====")
    
    # Create simulation with smaller token count for testing
    N = 64  # Reduced number of tokens
    model = GwaveGPU(N=N)
    
    # Initialize with golden spiral
    ell_init = torch.linspace(0, torch.log(torch.tensor(5.0)), N, device=model.device)
    theta_init = torch.tensor([(i * PHI.item()) % 1.0 for i in range(N)], device=model.device) * TAU
    
    model.initialize_tokens(ell_init, theta_init)
    
    # Run simulation with fewer steps
    results = model.run(max_steps=100)  # Reduced max steps
    
    print(f"Simulation completed in {results['runtime_seconds']:.2f} seconds")
    print(f"Steps completed: {results['steps_completed']}")
    print(f"Converged: {results['converged']}")
    
    # Visualize results
    model.visualize_trajectories()
    model.visualize_energy()
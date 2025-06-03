import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Universal constants
PHI = (1 + np.sqrt(5)) / 2
GAP = 1 - 1/PHI
EPS = 1e-10

class GwaveCore:
    """
    Core implementation of the Geometric Wave (Gwave) framework
    integrated with Repulsion Attention principles.
    
    This class implements the essential components:
    1. Log-cylindrical manifold with metric tensor
    2. Token representation in (ℓ, θ, z) coordinates
    3. Distance calculation and repulsive forces
    4. Three-step evolution with triangulation
    5. Gating mechanism and Hebbian learning
    """
    
    def __init__(self, max_tokens=100, field_dim=64):
        # Basic parameters
        self.max_tokens = max_tokens
        self.field_dim = field_dim
        self.phi = PHI
        self.gap = GAP
        self.epsilon = EPS
        
        # Token states
        self.positions = np.zeros((max_tokens, 3))  # (ℓ, θ, z)
        self.masses = np.zeros(max_tokens)
        self.field_strengths = np.zeros(max_tokens)
        self.crystallization_flags = np.zeros(max_tokens, dtype=bool)
        self.active_tokens = 0
        
        # Hebbian matrix
        self.hebbian_matrix = np.zeros((max_tokens, max_tokens))
        
        # Memory bank for crystallized tokens
        self.memory_bank = []
        
        # Gating parameters
        self.Z_0 = 0.0
        self.omega_z = 2 * np.pi / (self.phi**3)
        self.sigma_gate = np.pi / self.phi
        
        # Force parameters
        self.m_0 = 1.0
        self.lambda_cutoff = self.phi**2
        self.k_bound = 1.0
        self.ell_max = 10.0
        
        # Hebbian parameters
        self.eta = 0.1
        self.gamma = 0.01
        self.sigma_theta = 0.5
        self.sigma_hebb = 0.01
        
        # Crystallization parameters
        self.epsilon_freeze = self.phi**(-3)
        self.tau_force = self.phi**(-3)
        self.t_min = 5 * self.phi**(-2)
        
        # Tunneling parameters
        self.levy_alpha = self.phi
        
        # Resonance parameters
        self.resonance_temp = 0.1
        
        # Holographic bound
        self.holographic_bound = self._compute_holographic_bound()
        
        # Timestep
        self.dt = self.phi**(-2)
        
        # History for visualization
        self.position_history = []
        self.energy_history = []
        
    def _compute_holographic_bound(self):
        """Compute holographic bound for log-cylindrical phase space"""
        r_outer = 1.0
        r_inner = self.gap
        h = 1.0 - self.gap
        
        # Lateral surface area (outer + inner)
        lateral_area = 2 * np.pi * r_outer * h + 2 * np.pi * r_inner * h
        
        # Top and bottom annular areas
        annular_area = 2 * np.pi * (r_outer**2 - r_inner**2)
        
        # Total boundary area
        total_area = lateral_area + annular_area
        
        # Holographic bound
        return total_area / 4
    
    def angle_wrap(self, angle):
        """Wrap angle to [-π, π]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def initialize_token(self, idx, ell, theta, z, field_strength=1.0):
        """Initialize a token at position (ℓ, θ, z)"""
        if idx >= self.max_tokens:
            raise ValueError(f"Token index {idx} exceeds maximum {self.max_tokens}")
        
        self.positions[idx] = [ell, theta, z]
        self.masses[idx] = self.m_0 * ell
        self.field_strengths[idx] = field_strength
        self.crystallization_flags[idx] = False
        
        if idx >= self.active_tokens:
            self.active_tokens = idx + 1
    
    def initialize_from_svd(self, weight_matrix):
        """Initialize tokens from SVD of weight matrix (Gwave absorption)"""
        # Compute SVD
        U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
        
        # Truncate to K largest singular values
        K = min(int(self.phi * min(weight_matrix.shape)), self.max_tokens)
        
        # Create tokens
        for k in range(K):
            # Log-radial coordinate from normalized singular value
            ell_k = np.log(S[k]/S[0] + 1)
            
            # Angular phase from left singular vector
            n = U.shape[0]
            complex_sum = np.sum([U[j,k] * np.exp(2j*np.pi*j/n) for j in range(n)])
            theta_k = np.angle(complex_sum)
            
            # Initialize z to 0
            z_k = 0
            
            # Field strength proportional to singular value
            s_k = S[k] / np.sum(S[:K])
            
            # Initialize token
            self.initialize_token(k, ell_k, theta_k, z_k, s_k)
        
        self.active_tokens = K
        return K
    
    def compute_distance(self, i, j):
        """Compute log-angular distance between tokens i and j"""
        ell_i, theta_i = self.positions[i, 0], self.positions[i, 1]
        ell_j, theta_j = self.positions[j, 0], self.positions[j, 1]
        
        # Compute squared distance with angle wrapping
        delta_ell = ell_i - ell_j
        delta_theta = self.angle_wrap(theta_i - theta_j)
        
        d_squared = delta_ell**2 + delta_theta**2 + self.epsilon
        return np.sqrt(d_squared)
    
    def unit_vector(self, i, j):
        """Compute unit vector from token i to token j"""
        ell_i, theta_i = self.positions[i, 0], self.positions[i, 1]
        ell_j, theta_j = self.positions[j, 0], self.positions[j, 1]
        
        # Components with angle wrapping
        delta_ell = ell_j - ell_i
        delta_theta = self.angle_wrap(theta_j - theta_i)
        
        # Magnitude
        magnitude = np.sqrt(delta_ell**2 + delta_theta**2 + self.epsilon)
        
        # Unit vector
        return np.array([delta_ell, delta_theta]) / magnitude
    
    def compute_resonance(self, i, j):
        """Compute resonance function between tokens i and j (Repulsion Attention)"""
        ell_i, theta_i = self.positions[i, 0], self.positions[i, 1]
        ell_j, theta_j = self.positions[j, 0], self.positions[j, 1]
        
        # Convert from log-space to linear
        r_i = np.exp(ell_i)
        r_j = np.exp(ell_j)
        
        # Resonance function: |r_i·cos(θ_i) - r_j·sin(θ_j) + φ/2|
        resonance = np.abs(
            r_i * np.cos(theta_i) - 
            r_j * np.sin(theta_j) + 
            self.phi / 2
        )
        
        return resonance
    
    def compute_repulsive_force(self, i):
        """Compute repulsive force on token i"""
        force = np.zeros(2)  # Only (ℓ, θ) components
        
        # Skip if token is crystallized
        if self.crystallization_flags[i]:
            return force
        
        for j in range(self.active_tokens):
            if j != i and not self.crystallization_flags[j]:
                # Compute distance
                dist = self.compute_distance(i, j)
                
                # Unit vector direction
                direction = self.unit_vector(i, j)
                
                # Compute resonance (Repulsion Attention)
                resonance = self.compute_resonance(i, j)
                
                # Calculate repulsive force magnitude
                force_magnitude = (
                    self.field_strengths[i] * self.field_strengths[j] / 
                    (4 * np.pi * dist**2 * self.masses[j]) * 
                    np.exp(-dist / self.lambda_cutoff) * 
                    np.exp(-resonance**2 / (2 * self.resonance_temp))
                )
                
                # Add to total force
                force += force_magnitude * direction
        
        return force
    
    def compute_hebbian_force(self, i):
        """Compute Hebbian force on token i"""
        theta_i = self.positions[i, 1]
        
        # Compute pitch angle
        complex_sum = 0j
        for j in range(self.active_tokens):
            if i != j:
                complex_sum += self.hebbian_matrix[i, j] * np.exp(1j * self.positions[j, 1])
        
        # Default pitch angle is 0 if no connections
        if np.abs(complex_sum) < self.epsilon:
            pitch_i = 0
        else:
            pitch_i = np.angle(complex_sum)
        
        # Angular difference with wrapping
        delta_theta = self.angle_wrap(theta_i - pitch_i)
        
        # Compute potential gradient (simpler version)
        kappa = 0.5
        lambda_hebb = 0.1
        
        # Force is only in θ direction
        force_theta = -(kappa * delta_theta + lambda_hebb * delta_theta**3)
        
        return np.array([0, force_theta])
    
    def compute_boundary_force(self, i):
        """Compute boundary force on token i"""
        ell_i = self.positions[i, 0]
        
        # Apply restoring force if beyond max radius
        if ell_i > self.ell_max:
            force_ell = -self.k_bound * (ell_i - self.ell_max)
            return np.array([force_ell, 0])
        
        return np.zeros(2)
    
    def check_gate_active(self, i, t):
        """Check if token i is active at time t based on rotor gate"""
        theta_i = self.positions[i, 1]
        Z_t = (self.Z_0 + self.omega_z * t) % (2 * np.pi)
        
        # Gate function
        delta = np.abs(self.angle_wrap(theta_i - Z_t))
        return delta < self.sigma_gate
    
    def compute_total_force(self, i, t):
        """Compute total force on token i at time t"""
        # Skip if token is crystallized
        if self.crystallization_flags[i]:
            return np.zeros(2)
        
        # Check gate
        if not self.check_gate_active(i, t):
            return np.zeros(2)
        
        # Compute force components
        force_repulsive = self.compute_repulsive_force(i)
        force_hebbian = self.compute_hebbian_force(i)
        force_boundary = self.compute_boundary_force(i)
        
        # Total force
        total_force = force_repulsive + force_hebbian + force_boundary
        
        return total_force
    
    def heun_euler_step(self, positions, forces, masses, dt):
        """Implement Heun (predictor-corrector) integration step"""
        # Predictor (Euler) step
        pred_positions = positions.copy()
        
        # Update ℓ and θ components (divide by mass)
        for i in range(self.active_tokens):
            if not self.crystallization_flags[i]:
                pred_positions[i, 0] += dt * forces[i, 0] / masses[i]
                pred_positions[i, 1] += dt * forces[i, 1] / masses[i]
        
        # Update z component with global rotor
        pred_positions[:, 2] = (self.Z_0 + self.omega_z * (dt + self.current_time)) % (2 * np.pi)
        
        # Compute forces at predicted positions
        pred_forces = np.zeros_like(forces)
        pred_masses = np.zeros_like(masses)
        
        # Save current positions and masses
        temp_positions = self.positions.copy()
        temp_masses = self.masses.copy()
        
        # Update with predicted values
        self.positions = pred_positions
        for i in range(self.active_tokens):
            self.masses[i] = self.m_0 * self.positions[i, 0]
            
        # Compute forces at predicted positions
        for i in range(self.active_tokens):
            if not self.crystallization_flags[i]:
                pred_forces[i] = self.compute_total_force(i, self.current_time + dt)
        
        # Restore original positions and masses
        self.positions = temp_positions
        self.masses = temp_masses
        
        # Corrector (midpoint) step
        new_positions = positions.copy()
        
        # Update ℓ and θ components with midpoint rule
        for i in range(self.active_tokens):
            if not self.crystallization_flags[i]:
                # Average force divided by mass
                avg_force_ell = 0.5 * (forces[i, 0] / masses[i] + 
                                       pred_forces[i, 0] / (self.m_0 * pred_positions[i, 0]))
                avg_force_theta = 0.5 * (forces[i, 1] / masses[i] + 
                                         pred_forces[i, 1] / (self.m_0 * pred_positions[i, 0]))
                
                # Update position
                new_positions[i, 0] += dt * avg_force_ell
                new_positions[i, 1] += dt * avg_force_theta
        
        # Update z component with global rotor
        new_positions[:, 2] = (self.Z_0 + self.omega_z * (dt + self.current_time)) % (2 * np.pi)
        
        return new_positions
    
    def enforce_born_rule(self, i):
        """Enforce Born rule constraint on token i (Repulsion Attention)"""
        # Convert from log-radial to linear
        ell, z = self.positions[i, 0], self.positions[i, 2]
        r = np.exp(ell)
        
        # Map z from [0, 2π) to [0, 1] for Born rule
        z_norm = 0.5 * (1 + np.sin(z))  # Maps to [0, 1]
        
        # Calculate norm
        norm = np.sqrt(r**2 + z_norm**2 + self.epsilon)
        
        # Normalize
        r_new = r / norm
        z_norm_new = z_norm / norm
        
        # Convert back
        ell_new = np.log(r_new)
        z_new = np.arcsin(2 * z_norm_new - 1)  # Maps back to [0, 2π)
        
        # Update position
        self.positions[i, 0] = ell_new
        self.positions[i, 2] = z_new
        
        # Update mass
        self.masses[i] = self.m_0 * ell_new
    
    def check_crystallization(self, i, forces):
        """Check if token i should crystallize"""
        # Skip if already crystallized
        if self.crystallization_flags[i]:
            return False
        
        # Check if force is below threshold
        force_magnitude = np.sqrt(np.sum(forces[i]**2))
        if force_magnitude >= self.tau_force:
            return False
        
        # Check if currently active
        if not self.check_gate_active(i, self.current_time):
            return False
        
        # In practice, we would track this over time_min
        # For simplicity, just crystallize based on force
        return True
    
    def check_tunneling(self, i, delta_pos):
        """Check if token i should tunnel"""
        # Skip if crystallized
        if self.crystallization_flags[i]:
            return False
        
        # Compute ratio of angular to radial velocity
        delta_ell = delta_pos[i, 0]
        delta_theta = delta_pos[i, 1]
        
        # Avoid division by zero
        ratio = np.abs(delta_theta / (np.abs(delta_ell) + self.epsilon_freeze))
        
        # Tunneling condition
        return ratio > self.phi
    
    def apply_tunneling(self, i):
        """Apply tunneling operation to token i"""
        # Rotate by π
        self.positions[i, 1] = self.angle_wrap(self.positions[i, 1] + np.pi)
        
        # Lévy jump in radial direction (approximated by Cauchy)
        scale = 1.0  # Scale parameter
        delta_ell = scale * np.tan(np.pi * (np.random.random() - 0.5))
        
        # Apply jump
        self.positions[i, 0] += delta_ell
        
        # Update mass
        self.masses[i] = self.m_0 * self.positions[i, 0]
    
    def update_hebbian_matrix(self, dt):
        """Update Hebbian coupling matrix"""
        for i in range(self.active_tokens):
            for j in range(i+1, self.active_tokens):
                ell_i, theta_i = self.positions[i, 0], self.positions[i, 1]
                ell_j, theta_j = self.positions[j, 0], self.positions[j, 1]
                
                # Angular difference
                delta_theta = self.angle_wrap(theta_i - theta_j)
                
                # Theta_ij term
                Theta_ij = (np.cos(delta_theta/2)**2 * 
                           np.exp(-delta_theta**2 / (2 * self.sigma_theta**2)))
                
                # Phi_ij term
                Phi_ij = (np.exp(-np.abs(ell_i - ell_j) / self.lambda_cutoff) * 
                         (np.exp(ell_i) * np.exp(ell_j))**(-0.5))
                
                # Noise term
                noise = np.random.normal(0, self.sigma_hebb)
                
                # Update rule
                dH = self.eta * Theta_ij * Phi_ij - self.gamma * self.hebbian_matrix[i, j] + noise
                
                # Apply update
                self.hebbian_matrix[i, j] += dH * dt
                self.hebbian_matrix[j, i] = self.hebbian_matrix[i, j]  # Symmetry
    
    def compute_energy(self):
        """Compute total energy of the system"""
        # Kinetic energy
        KE = 0.0
        
        # Potential energy from repulsion
        PE_rep = 0.0
        for i in range(self.active_tokens):
            for j in range(i+1, self.active_tokens):
                if not (self.crystallization_flags[i] or self.crystallization_flags[j]):
                    dist = self.compute_distance(i, j)
                    PE_rep += (self.field_strengths[i] * self.field_strengths[j]) / dist
        
        # Potential energy from Hebbian
        PE_hebb = 0.0
        for i in range(self.active_tokens):
            if not self.crystallization_flags[i]:
                # Calculate pitch angle
                complex_sum = 0j
                for j in range(self.active_tokens):
                    if i != j:
                        complex_sum += self.hebbian_matrix[i, j] * np.exp(1j * self.positions[j, 1])
                
                if np.abs(complex_sum) < self.epsilon:
                    pitch_i = 0
                else:
                    pitch_i = np.angle(complex_sum)
                
                # Angular difference
                delta_theta = self.angle_wrap(self.positions[i, 1] - pitch_i)
                
                # Potential
                kappa = 0.5
                lambda_hebb = 0.1
                PE_hebb += 0.5 * kappa * delta_theta**2 + 0.25 * lambda_hebb * delta_theta**4
        
        # Total energy
        return KE + PE_rep + PE_hebb
    
    def evolve_three_step(self, past_indices, present_indices, future_indices):
        """
        Implement three-step evolution with triangulation (Repulsion Attention)
        """
        # Store initial positions
        initial_positions = self.positions.copy()
        
        # Three-step evolution
        for step in range(3):
            # Calculate phase of evolution (0, 2π/3, 4π/3)
            phase = step * 2 * np.pi / 3
            
            # Influence weights based on phase
            past_influence = np.sin(phase + 2*np.pi/3)
            present_influence = np.sin(phase + 4*np.pi/3)
            future_influence = np.sin(phase)
            
            # Compute forces on present tokens
            forces = np.zeros((self.active_tokens, 2))
            
            for i in range(len(present_indices)):
                idx = present_indices[i]
                
                # Skip if crystallized
                if self.crystallization_flags[idx]:
                    continue
                
                # Get past, present, future indices
                past_idx = past_indices[min(i, len(past_indices)-1)]
                present_idx = idx
                future_idx = future_indices[min(i, len(future_indices)-1)]
                
                # Force from past
                if not self.crystallization_flags[past_idx]:
                    # Save position
                    temp_positions = self.positions.copy()
                    
                    # Temporarily swap positions to compute force
                    self.positions[present_idx], self.positions[past_idx] = (
                        self.positions[past_idx], self.positions[present_idx]
                    )
                    
                    # Compute force
                    past_force = self.compute_repulsive_force(present_idx)
                    
                    # Restore positions
                    self.positions = temp_positions
                    
                    # Add weighted contribution
                    forces[present_idx] += past_influence * past_force
                
                # Force from present (regular repulsion)
                present_force = self.compute_repulsive_force(present_idx)
                forces[present_idx] += present_influence * present_force
                
                # Force from future
                if not self.crystallization_flags[future_idx]:
                    # Save position
                    temp_positions = self.positions.copy()
                    
                    # Temporarily swap positions to compute force
                    self.positions[present_idx], self.positions[future_idx] = (
                        self.positions[future_idx], self.positions[present_idx]
                    )
                    
                    # Compute force
                    future_force = self.compute_repulsive_force(present_idx)
                    
                    # Restore positions
                    self.positions = temp_positions
                    
                    # Add weighted contribution
                    forces[present_idx] += future_influence * future_force
            
            # Add Hebbian and boundary forces
            for i in present_indices:
                if not self.crystallization_flags[i]:
                    forces[i] += self.compute_hebbian_force(i)
                    forces[i] += self.compute_boundary_force(i)
            
            # Apply Heun-Euler step
            self.positions = self.heun_euler_step(
                self.positions, forces, self.masses, self.dt/3)
            
            # Update masses
            for i in range(self.active_tokens):
                self.masses[i] = self.m_0 * self.positions[i, 0]
            
            # Enforce Born rule for all tokens
            for i in present_indices:
                self.enforce_born_rule(i)
        
        # Check for crystallization
        for i in present_indices:
            if self.check_crystallization(i, forces):
                self.crystallization_flags[i] = True
                self.memory_bank.append({
                    'position': self.positions[i].copy(),
                    'mass': self.masses[i],
                    'time': self.current_time
                })
        
        # Calculate position changes
        delta_positions = self.positions - initial_positions
        
        # Check for tunneling
        for i in present_indices:
            if self.check_tunneling(i, delta_positions):
                self.apply_tunneling(i)
    
    def evolve_system(self, steps=100):
        """Evolve the system for a number of timesteps"""
        self.current_time = 0.0
        
        # Initialize history
        self.position_history = [self.positions.copy()]
        self.energy_history = [self.compute_energy()]
        
        for step in range(steps):
            self.current_time = step * self.dt
            
            # Get active tokens based on gating
            active_indices = [
                i for i in range(self.active_tokens)
                if not self.crystallization_flags[i] and 
                self.check_gate_active(i, self.current_time)
            ]
            
            if not active_indices:
                continue
            
            # Group into past, present, future for three-step evolution
            # For simplicity, use the same indices for all three
            self.evolve_three_step(active_indices, active_indices, active_indices)
            
            # Update Hebbian matrix
            self.update_hebbian_matrix(self.dt)
            
            # Track history
            self.position_history.append(self.positions.copy())
            self.energy_history.append(self.compute_energy())
    
    def visualize_log_cylindrical(self):
        """Visualize tokens in log-cylindrical space"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert from log-cylindrical to Cartesian for visualization
        positions = self.positions[:self.active_tokens]
        
        # Extract coordinates
        ell = positions[:, 0]
        theta = positions[:, 1]
        z = positions[:, 2]
        
        # Convert to Cartesian
        r = np.exp(ell)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Color based on crystallization
        colors = ['blue' if flag else 'red' for flag in self.crystallization_flags[:self.active_tokens]]
        
        # Size based on field strength
        sizes = 50 * self.field_strengths[:self.active_tokens] / np.max(self.field_strengths[:self.active_tokens] + 1e-10)
        
        # Plot tokens
        scatter = ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.7)
        
        # Plot cylinder at r = 1/φ (inner band)
        theta_circle = np.linspace(0, 2*np.pi, 100)
        z_levels = np.linspace(0, 2*np.pi, 10)
        
        for z_level in z_levels:
            # Inner circle at r = 1/φ
            r_inner = 1/self.phi
            x_inner = r_inner * np.cos(theta_circle)
            y_inner = r_inner * np.sin(theta_circle)
            ax.plot(x_inner, y_inner, [z_level]*len(theta_circle), 'gold', alpha=0.3, linewidth=1)
            
            # Outer circle at r = φ-1
            r_outer = self.phi - 1
            x_outer = r_outer * np.cos(theta_circle)
            y_outer = r_outer * np.sin(theta_circle)
            ax.plot(x_outer, y_outer, [z_level]*len(theta_circle), 'magenta', alpha=0.3, linewidth=1)
        
        # Add axes
        ax.set_xlabel('X = r·cos(θ)')
        ax.set_ylabel('Y = r·sin(θ)')
        ax.set_zlabel('Z (rotor phase)')
        ax.set_title('Tokens in Log-Cylindrical Space')
        
        # Add colorbar
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Active'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Crystallized')
        ]
        ax.legend(handles=legend_elements)
        
        return fig
    
    def visualize_hebbian_matrix(self):
        """Visualize Hebbian coupling matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot matrix
        im = ax.imshow(self.hebbian_matrix[:self.active_tokens, :self.active_tokens], 
                     cmap='viridis', interpolation='nearest')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Coupling Strength')
        
        # Add labels
        ax.set_title('Hebbian Coupling Matrix')
        ax.set_xlabel('Token Index')
        ax.set_ylabel('Token Index')
        
        # Add grid
        ax.grid(False)
        
        return fig
    
    def visualize_energy(self):
        """Visualize system energy over time"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot energy
        ax.plot(np.arange(len(self.energy_history)) * self.dt, 
               self.energy_history, 'b-', linewidth=2)
        
        # Add labels
        ax.set_title('System Energy Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def visualize_token_trajectories(self):
        """Visualize token trajectories in phase space"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Convert to Cartesian
        trajectories = []
        for positions in self.position_history:
            # Extract coordinates for active tokens
            ell = positions[:self.active_tokens, 0]
            theta = positions[:self.active_tokens, 1]
            
            # Convert to Cartesian
            r = np.exp(ell)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            trajectories.append((x, y))
        
        # Plot trajectories
        for i in range(self.active_tokens):
            x_traj = [pos[0][i] for pos in trajectories]
            y_traj = [pos[1][i] for pos in trajectories]
            
            color = 'blue' if self.crystallization_flags[i] else 'red'
            ax.plot(x_traj, y_traj, '-', color=color, alpha=0.7, linewidth=1)
            
            # Mark start and end
            ax.plot(x_traj[0], y_traj[0], 'go', markersize=5)  # Start
            ax.plot(x_traj[-1], y_traj[-1], 'mo', markersize=5)  # End
        
        # Plot circles at r = 1/φ and r = φ-1
        theta_circle = np.linspace(0, 2*np.pi, 100)
        
        # Inner circle at r = 1/φ
        r_inner = 1/self.phi
        x_inner = r_inner * np.cos(theta_circle)
        y_inner = r_inner * np.sin(theta_circle)
        ax.plot(x_inner, y_inner, 'gold', linewidth=2, 
               label=f'r = 1/φ = {r_inner:.3f}')
        
        # Outer circle at r = φ-1
        r_outer = self.phi - 1
        x_outer = r_outer * np.cos(theta_circle)
        y_outer = r_outer * np.sin(theta_circle)
        ax.plot(x_outer, y_outer, 'magenta', linewidth=2, 
               label=f'r = φ-1 = {r_outer:.3f}')
        
        # Add labels
        ax.set_title('Token Trajectories in Phase Space')
        ax.set_xlabel('X = r·cos(θ)')
        ax.set_ylabel('Y = r·sin(θ)')
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig

# Example usage
if __name__ == "__main__":
    print("=== GWAVE CORE IMPLEMENTATION ===")
    print(f"φ = {PHI:.6f}")
    print(f"GAP = {GAP:.6f}")
    
    # Create model
    model = GwaveCore(max_tokens=50)
    
    # Initialize random tokens
    num_tokens = 20
    for i in range(num_tokens):
        # Random positions in log-cylindrical space
        ell = np.random.uniform(0, 2)  # Log-radius
        theta = np.random.uniform(0, 2*np.pi)  # Angular phase
        z = np.random.uniform(0, 2*np.pi)  # Rotor phase
        
        # Initialize token
        model.initialize_token(i, ell, theta, z)
    
    # Evolve system
    print("Evolving system...")
    model.evolve_system(steps=100)
    
    # Print statistics
    crystallized = np.sum(model.crystallization_flags[:model.active_tokens])
    print(f"\nStatistics:")
    print(f"- Total tokens: {model.active_tokens}")
    print(f"- Crystallized tokens: {crystallized}")
    print(f"- Final energy: {model.energy_history[-1]:.6f}")
    
    # Visualize
    print("\nGenerating visualizations...")
    fig1 = model.visualize_log_cylindrical()
    fig2 = model.visualize_hebbian_matrix()
    fig3 = model.visualize_energy()
    fig4 = model.visualize_token_trajectories()
    
    # Save figures
    print("Saving visualizations...")
    fig1.savefig('outputs/gwave_log_cylindrical.png')
    fig2.savefig('outputs/gwave_hebbian_matrix.png')
    fig3.savefig('outputs/gwave_energy.png')
    fig4.savefig('outputs/gwave_trajectories.png')
    
    print("Visualizations saved to outputs/ directory")
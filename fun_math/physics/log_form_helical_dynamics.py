#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log-Form Helical Dynamics with Tachyonic Principles

This script implements a unified model combining:
1. Log-cylindrical coordinate system
2. Helical structure with φ-based frequencies
3. Tachyonic information propagation
4. Three-step Heun-Euler evolution
5. Born rule normalization
6. Parallel processing for token evolution
7. NLTK integration for semantic mapping

The implementation prioritizes the log-form representation throughout,
avoiding exponential conversions and maintaining consistency with
the Gwave specification.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import levy_stable
import multiprocessing as mp
from functools import partial
import time
import os
import warnings
import logging

# Try to import NLTK - make it optional
try:
    import nltk
    from nltk.corpus import wordnet as wn
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Will use random initialization instead of WordNet.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Universal constants
PHI = (1 + np.sqrt(5)) / 2
GAP = 1 - 1/PHI
EPS = 1e-10
OUTPUT_DIR = "outputs/gwave/physics"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download NLTK resources if needed
if NLTK_AVAILABLE:
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')


class LogFormHelicalDynamics:
    """
    Unified implementation of log-form helical dynamics with tachyonic principles.
    All calculations are performed in log-space (ℓ = ln r) to maintain consistency
    with the Gwave specification and avoid exponential conversions.
    """
    
    def __init__(self, max_tokens=1000, field_dim=64, num_processes=None):
        """
        Initialize the model with log-form coordinates and parameters.
        
        Parameters:
        -----------
        max_tokens : int
            Maximum number of tokens to support
        field_dim : int
            Dimension of the quantum field representation
        num_processes : int or None
            Number of processes to use for parallel computation
            If None, uses CPU count - 1
        """
        # Basic parameters
        self.max_tokens = max_tokens
        self.field_dim = field_dim
        self.phi = PHI
        self.gap = GAP
        self.epsilon = EPS
        
        # Parallelization
        self.num_processes = num_processes if num_processes is not None else max(1, mp.cpu_count() - 1)
        logger.info(f"Using {self.num_processes} processes for parallel computation")
        
        # Log-cylindrical space parameters
        self.ln_r_min = np.log(self.gap)    # Inner radius bound in log scale
        self.ln_r_max = 0.0                 # Outer radius bound in log scale (ln(1) = 0)
        self.z_min = 0.0                    # Minimum z coordinate
        self.z_max = 2*np.pi                # Maximum z coordinate
        
        # Metric tensor in log-cylindrical coordinates
        # g_μν = [1+ℓ²  0   0]
        #        [ 0   ℓ²  0]
        #        [ 0    0  φ²]
        self.metric_diagonal = np.array([1.0, 1.0, self.phi**2])
        
        # Token states
        self.positions = np.zeros((max_tokens, 3))  # (ℓ, θ, z)
        self.momenta = np.zeros((max_tokens, 3))    # (p_ℓ, p_θ, p_z)
        self.masses = np.zeros(max_tokens)          # m = m_0 * ℓ
        self.field_strengths = np.zeros(max_tokens) # s ∈ [0, 1]
        self.crystallization_flags = np.zeros(max_tokens, dtype=bool)
        self.active_tokens = 0
        
        # Mass scale constant
        self.m_0 = 1.0
        
        # Hebbian matrix
        self.hebbian_matrix = np.zeros((max_tokens, max_tokens))
        
        # Memory bank for crystallized tokens
        self.memory_bank = []
        
        # Gating parameters
        self.Z_0 = 0.0
        self.omega_z = 2 * np.pi / (self.phi**3)
        self.sigma_gate = np.pi / self.phi
        
        # Force parameters
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
        
        # Tachyonic parameters
        self.c = 1.0  # Semantic speed of light
        
        # Holographic bound
        self.holographic_bound = self._compute_holographic_bound()
        
        # Timestep - logarithmic friendly
        self.dt = self.phi**(-2)
        
        # Current time
        self.current_time = 0.0
        
        # History for visualization
        self.position_history = []
        self.energy_history = []
        self.tachyonic_events = []
        
        # NLTK semantic map
        self.semantic_map = {}
        
    def _compute_holographic_bound(self):
        """
        Compute holographic bound for log-cylindrical phase space.
        S ≤ A/(4ℓ_p²) where A is the boundary area and ℓ_p = φ^(-1).
        """
        # Convert from log-space to linear for area calculation
        r_outer = 1.0  # exp(0)
        r_inner = self.gap
        h = 1.0 - self.gap
        
        # Lateral surface area (outer + inner)
        lateral_area = 2 * np.pi * r_outer * h + 2 * np.pi * r_inner * h
        
        # Top and bottom annular areas
        annular_area = 2 * np.pi * (r_outer**2 - r_inner**2)
        
        # Total boundary area
        total_area = lateral_area + annular_area
        
        # Holographic bound with Planck length ℓ_p = φ^(-1)
        return total_area / (4 * self.phi**(-2))
    
    def angle_wrap(self, angle):
        """
        Wrap angle to [-π, π] - the standard angle-wrap function.
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def christoffel_symbols(self, position):
        """
        Compute Christoffel symbols at a given position.
        Returns the non-zero components needed for geodesic calculation.
        """
        ell = position[0]  # logarithmic radial coordinate
        
        # Non-zero components for diagonal metric
        # Γᵏᵢⱼ = (1/2) g^kl (∂ᵢg_jl + ∂ⱼg_il - ∂_lg_ij)
        gamma_ell_ell_ell = ell / (1 + ell**2)
        gamma_ell_theta_theta = -ell**3 / (1 + ell**2)
        gamma_theta_ell_theta = gamma_theta_theta_ell = 1/ell
        
        return {
            'gamma_ell_ell_ell': gamma_ell_ell_ell,
            'gamma_ell_theta_theta': gamma_ell_theta_theta,
            'gamma_theta_ell_theta': gamma_theta_ell_theta,
            'gamma_theta_theta_ell': gamma_theta_theta_ell
        }
    
    def initialize_token(self, idx, ell, theta, z, field_strength=1.0):
        """
        Initialize a token at position (ℓ, θ, z) in log-cylindrical space.
        """
        if idx >= self.max_tokens:
            raise ValueError(f"Token index {idx} exceeds maximum {self.max_tokens}")
        
        self.positions[idx] = [ell, theta, z]
        self.masses[idx] = self.m_0 * ell
        self.field_strengths[idx] = field_strength
        self.crystallization_flags[idx] = False
        
        if idx >= self.active_tokens:
            self.active_tokens = idx + 1
    
    def initialize_from_wordnet(self, num_tokens=100, root_synset='entity.n.01'):
        """
        Initialize tokens using WordNet semantic hierarchy.
        Maps semantic depth to logarithmic radius.
        """
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available. Using random initialization instead.")
            return self.initialize_random(num_tokens)
            
        try:
            # Get root synset
            root = wn.synset(root_synset)
            
            # BFS to get a diversity of concepts
            synsets = [root]
            visited = set([root])
            queue = [root]
            
            while queue and len(synsets) < num_tokens:
                current = queue.pop(0)
                for hypo in current.hyponyms():
                    if hypo not in visited and len(synsets) < num_tokens:
                        synsets.append(hypo)
                        visited.add(hypo)
                        queue.append(hypo)
            
            # Limit to num_tokens
            synsets = synsets[:num_tokens]
            
            # Calculate max depth for normalization
            max_depth = max(s.max_depth() for s in synsets)
            
            # Initialize tokens based on WordNet properties
            for i, synset in enumerate(synsets):
                # Depth determines log-radius (deeper = smaller radius)
                depth_ratio = synset.max_depth() / max_depth
                ell = self.ln_r_min + (1 - depth_ratio) * (self.ln_r_max - self.ln_r_min)
                
                # Lexical diversity determines angle
                lemmas = synset.lemma_names()
                theta = (hash(''.join(lemmas)) % 1000) / 1000 * 2 * np.pi
                
                # Abstraction level determines z
                z = depth_ratio * 2 * np.pi
                
                # Store semantic mapping
                self.semantic_map[i] = {
                    'synset': synset.name(),
                    'definition': synset.definition(),
                    'lemmas': lemmas,
                    'depth': synset.max_depth()
                }
                
                # Initialize token
                self.initialize_token(i, ell, theta, z, field_strength=1.0)
            
            logger.info(f"Initialized {len(synsets)} tokens from WordNet")
            return len(synsets)
        
        except Exception as e:
            logger.error(f"Error initializing from WordNet: {e}")
            # Fallback to random initialization
            return self.initialize_random(num_tokens)
    
    def initialize_random(self, num_tokens=100):
        """
        Initialize random tokens in log-cylindrical space.
        """
        for i in range(num_tokens):
            # Random positions in log-cylindrical space
            ell = np.random.uniform(self.ln_r_min, self.ln_r_max)
            theta = np.random.uniform(0, 2*np.pi)
            z = np.random.uniform(0, 2*np.pi)
            
            # Initialize token
            self.initialize_token(i, ell, theta, z)
        
        logger.info(f"Randomly initialized {num_tokens} tokens")
        return num_tokens
    
    def compute_distance(self, i, j):
        """
        Compute log-angular distance between tokens i and j.
        d_ij² = (ℓ_i - ℓ_j)² + [ω(θ_i - θ_j)]² + ε
        """
        ell_i, theta_i = self.positions[i, 0], self.positions[i, 1]
        ell_j, theta_j = self.positions[j, 0], self.positions[j, 1]
        
        # Compute squared distance with angle wrapping
        delta_ell = ell_i - ell_j
        delta_theta = self.angle_wrap(theta_i - theta_j)
        
        d_squared = delta_ell**2 + delta_theta**2 + self.epsilon
        return np.sqrt(d_squared)
    
    def unit_vector(self, i, j):
        """
        Compute unit vector from token i to token j in (ℓ, θ) plane.
        û_ij = (ℓ_j - ℓ_i, ω(θ_j - θ_i)) / ||(ℓ_j - ℓ_i, ω(θ_j - θ_i))||
        """
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
        """
        Compute resonance function between tokens i and j (Repulsion Attention).
        R_ij = |r_i·cos(θ_i) - r_j·sin(θ_j) + φ/2|
        """
        ell_i, theta_i = self.positions[i, 0], self.positions[i, 1]
        ell_j, theta_j = self.positions[j, 0], self.positions[j, 1]
        
        # Convert from log-space to linear for resonance calculation
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
        """
        Compute repulsive force on token i.
        F_i^rep = ∑_{j≠i, χ_j=0} (s_i s_j)/(4π d(x_i,x_j)² m_j) exp(-d(x_i,x_j)/λ) û_ij
        """
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
                
                # Compute resonance
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
        """
        Compute Hebbian force on token i.
        F_i^hebb = -∇_θ V_hebb(θ_i - φ_i)
        """
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
        
        # Compute potential gradient
        kappa = 0.5
        lambda_hebb = 0.1
        
        # Force is only in θ direction
        force_theta = -(kappa * delta_theta + lambda_hebb * delta_theta**3)
        
        return np.array([0, force_theta])
    
    def compute_boundary_force(self, i):
        """
        Compute boundary force on token i.
        F_i^bound = -k_bound(ℓ_i - ℓ_max)ê_ℓ if ℓ_i > ℓ_max, else 0
        """
        ell_i = self.positions[i, 0]
        
        # Apply restoring force if beyond max radius
        if ell_i > self.ell_max:
            force_ell = -self.k_bound * (ell_i - self.ell_max)
            return np.array([force_ell, 0])
        
        return np.zeros(2)
    
    def compute_quantum_force(self, i):
        """
        Compute quantum force on token i (Bohm-style quantum potential).
        F_i^quant = -∇(∇²√ρ_i)/(2m_i√ρ_i)
        
        This is a simplified approximation of the quantum potential.
        """
        # For simplicity, we'll use a small constant quantum force
        # In a full implementation, this would involve calculating
        # the Laplacian of the probability amplitude
        return np.zeros(2)  # Simplified version returns zero
    
    def check_gate_active(self, i, t):
        """
        Check if token i is active at time t based on rotor gate.
        G(x_i, Z_t) = 1 if |ω(θ_i - Z_t)| < σ_gate, else 0
        """
        theta_i = self.positions[i, 1]
        Z_t = (self.Z_0 + self.omega_z * t) % (2 * np.pi)
        
        # Gate function
        delta = np.abs(self.angle_wrap(theta_i - Z_t))
        return delta < self.sigma_gate
    
    def compute_total_force(self, i, t):
        """
        Compute total force on token i at time t.
        F_i^total = G(x_i, Z_t)(1-χ_i)[F_i^rep + F_i^hebb + F_i^bound + F_i^quant]
        """
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
        force_quantum = self.compute_quantum_force(i)
        
        # Total force
        total_force = force_repulsive + force_hebbian + force_boundary + force_quantum
        
        return total_force
    
    def compute_forces_parallel(self, t, indices=None):
        """
        Compute forces for multiple tokens in parallel.
        """
        if indices is None:
            indices = range(self.active_tokens)
        
        # Create a partial function with fixed arguments
        compute_force_partial = partial(self._compute_force_for_parallel, t=t)
        
        # Use Pool for parallel computation
        with mp.Pool(processes=self.num_processes) as pool:
            forces = pool.map(compute_force_partial, indices)
        
        # Convert to numpy array
        return np.array(forces)
    
    def _compute_force_for_parallel(self, i, t):
        """Helper function for parallel force computation."""
        return self.compute_total_force(i, t)
    
    def heun_euler_step(self, positions, forces, masses, dt):
        """
        Implement Heun (predictor-corrector) integration step.
        """
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
        """
        Enforce Born rule constraint on token i (Repulsion Attention).
        r² + z² = 1
        """
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
    
    def check_tachyonic_state(self, i, dt):
        """
        Check if token i is in a tachyonic state where phase velocity exceeds c.
        v_phase = r·dθ/dt = r·π/2·φ^(-n-2) > c
        """
        ell, theta = self.positions[i, 0], self.positions[i, 1]
        r = np.exp(ell)
        
        # Calculate phase velocity
        delta_theta = self.momenta[i, 1] / self.masses[i] * dt
        v_phase = r * np.abs(delta_theta / dt)
        
        # Check if superluminal
        is_tachyonic = v_phase > self.c
        
        if is_tachyonic:
            # Record tachyonic event
            self.tachyonic_events.append({
                'token': i,
                'time': self.current_time,
                'position': self.positions[i].copy(),
                'velocity': v_phase,
                'c_ratio': v_phase / self.c
            })
            
            # Calculate proper time change
            dtau_dt = np.sqrt(np.abs(1 - v_phase**2/self.c**2))
            # Imaginary for superluminal (use real part for computation)
            
            logger.debug(f"Tachyonic event: token {i}, v/c = {v_phase/self.c:.2f}")
        
        return is_tachyonic, v_phase
    
    def check_crystallization(self, i, forces):
        """
        Check if token i should crystallize.
        """
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
        """
        Check if token i should tunnel.
        |dθ/dt|/|dℓ/dt + ε| > φ
        """
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
        """
        Apply tunneling operation to token i.
        θ ← ω(θ + π), ℓ ← ℓ + Δℓ where Δℓ ~ Lévy(α=φ)
        """
        # Rotate by π
        self.positions[i, 1] = self.angle_wrap(self.positions[i, 1] + np.pi)
        
        # Lévy jump in radial direction (approximated by Cauchy)
        scale = 1.0  # Scale parameter
        delta_ell = scale * np.tan(np.pi * (np.random.random() - 0.5))
        
        # Apply jump
        self.positions[i, 0] += delta_ell
        
        # Update mass
        self.masses[i] = self.m_0 * self.positions[i, 0]
        
        logger.debug(f"Tunneling applied to token {i}, Δℓ = {delta_ell:.3f}")
    
    def update_hebbian_matrix(self, dt):
        """
        Update Hebbian coupling matrix.
        dH_ij/dt = η·Θ_ij·Φ_ij - γ·H_ij + ξ_ij(t)
        """
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
        """
        Compute total energy of the system.
        E = ∑_i ½ m_i||dx_i/dt||² + ∑_{i<j} (s_i s_j)/d(x_i,x_j) + ∑_i V_hebb(θ_i - φ_i)
        """
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
        Implement three-step evolution with triangulation (Repulsion Attention).
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
            
            # Compute forces on present tokens - in parallel
            forces = np.zeros((self.active_tokens, 2))
            
            # Process tokens in parallel for each step
            active_present = [idx for idx in present_indices if not self.crystallization_flags[idx]]
            if active_present:
                # Use parallel computation for forces
                forces_parallel = self.compute_forces_parallel(self.current_time, active_present)
                for i, idx in enumerate(active_present):
                    forces[idx] = forces_parallel[i]
            
            # Apply triangulation influences
            for i in range(len(present_indices)):
                idx = present_indices[i]
                
                # Skip if crystallized
                if self.crystallization_flags[idx]:
                    continue
                
                # Get past, present, future indices
                past_idx = past_indices[min(i, len(past_indices)-1)]
                future_idx = future_indices[min(i, len(future_indices)-1)]
                
                # Apply influence weights
                forces[idx] = (past_influence * forces[idx] + 
                              present_influence * forces[idx] + 
                              future_influence * forces[idx])
            
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
                    'time': self.current_time,
                    'semantic': self.semantic_map.get(i, None)
                })
                logger.debug(f"Token {i} crystallized at t={self.current_time:.3f}")
        
        # Calculate position changes
        delta_positions = self.positions - initial_positions
        
        # Check for tunneling
        for i in present_indices:
            if self.check_tunneling(i, delta_positions):
                self.apply_tunneling(i)
        
        # Check for tachyonic states
        for i in present_indices:
            is_tachyonic, v_phase = self.check_tachyonic_state(i, self.dt)
            # Tachyonic states are recorded internally
    
    def evolve_system(self, steps=100):
        """
        Evolve the system for a number of timesteps.
        """
        self.current_time = 0.0
        
        # Initialize history
        self.position_history = [self.positions.copy()]
        self.energy_history = [self.compute_energy()]
        self.tachyonic_events = []
        
        start_time = time.time()
        
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
            
            # Progress logging
            if (step + 1) % 10 == 0 or step == steps - 1:
                elapsed = time.time() - start_time
                crystallized = np.sum(self.crystallization_flags[:self.active_tokens])
                tachyonic = len(self.tachyonic_events)
                logger.info(f"Step {step+1}/{steps} completed in {elapsed:.2f}s. "
                           f"Crystallized: {crystallized}, Tachyonic events: {tachyonic}")
    
    def visualize_log_cylindrical(self, filename="log_cylindrical_space.png"):
        """
        Visualize tokens in log-cylindrical space.
        """
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
        
        # Add critical radius for tachyonic behavior
        r_critical = self.c * self.phi**2 / np.pi
        for z_level in z_levels:
            x_crit = r_critical * np.cos(theta_circle)
            y_crit = r_critical * np.sin(theta_circle)
            ax.plot(x_crit, y_crit, [z_level]*len(theta_circle), 
                   'red', alpha=0.3, linewidth=1, linestyle='--')
        
        # Add axes
        ax.set_xlabel('X = r·cos(θ)')
        ax.set_ylabel('Y = r·sin(θ)')
        ax.set_zlabel('Z (rotor phase)')
        ax.set_title('Tokens in Log-Cylindrical Space')
        
        # Add colorbar
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Active'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Crystallized'),
            plt.Line2D([0], [0], color='gold', lw=2, label=f'r = 1/φ = {1/self.phi:.3f}'),
            plt.Line2D([0], [0], color='magenta', lw=2, label=f'r = φ-1 = {self.phi-1:.3f}'),
            plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label=f'r_crit = {r_critical:.3f}')
        ]
        ax.legend(handles=legend_elements)
        
        # Save figure
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {os.path.join(OUTPUT_DIR, filename)}")
        
        return fig
    
    def visualize_helical_trajectories(self, token_indices=None, filename="helical_trajectories.png"):
        """
        Visualize helical trajectories of selected tokens.
        """
        if token_indices is None:
            # Select a subset of active tokens
            token_indices = range(min(5, self.active_tokens))
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectories for selected tokens
        for idx in token_indices:
            # Extract trajectory
            traj = np.array([pos[idx] for pos in self.position_history])
            
            # Convert from log-cylindrical to Cartesian
            ell = traj[:, 0]
            theta = traj[:, 1]
            z = traj[:, 2]
            
            r = np.exp(ell)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Color based on final state
            color = 'blue' if self.crystallization_flags[idx] else 'red'
            
            # Plot trajectory
            ax.plot(x, y, z, color=color, linewidth=2, 
                   label=f'Token {idx}' + (' (crystallized)' if self.crystallization_flags[idx] else ''))
            
            # Mark start and end
            ax.scatter(x[0], y[0], z[0], color='green', s=50)
            ax.scatter(x[-1], y[-1], z[-1], color='purple', s=50)
        
        # Plot cylinder at preferred radii
        theta_circle = np.linspace(0, 2*np.pi, 100)
        z_max = max([pos[:, 2].max() for pos in self.position_history])
        z_levels = np.linspace(0, z_max, 5)
        
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
        
        # Add reference helix
        t_ref = np.linspace(0, z_max, 200)
        r_ref = 1/self.phi
        x_ref = r_ref * np.cos(t_ref * self.phi)
        y_ref = r_ref * np.sin(t_ref * self.phi)
        z_ref = t_ref
        ax.plot(x_ref, y_ref, z_ref, 'k--', alpha=0.5, linewidth=1,
               label='Reference helix (φ frequency)')
        
        # Add labels
        ax.set_xlabel('X = r·cos(θ)')
        ax.set_ylabel('Y = r·sin(θ)')
        ax.set_zlabel('Z (rotor phase)')
        ax.set_title('Helical Trajectories in Log-Cylindrical Space')
        
        # Add legend
        ax.legend()
        
        # Save figure
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {os.path.join(OUTPUT_DIR, filename)}")
        
        return fig
    
    def visualize_tachyonic_events(self, filename="tachyonic_events.png"):
        """
        Visualize tachyonic events in phase space.
        """
        if not self.tachyonic_events:
            logger.info("No tachyonic events to visualize")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Extract data from tachyonic events
        tokens = [event['token'] for event in self.tachyonic_events]
        times = [event['time'] for event in self.tachyonic_events]
        velocities = [event['velocity'] for event in self.tachyonic_events]
        c_ratios = [event['c_ratio'] for event in self.tachyonic_events]
        positions = [event['position'] for event in self.tachyonic_events]
        
        # Extract coordinates
        ell = [pos[0] for pos in positions]
        theta = [pos[1] for pos in positions]
        z = [pos[2] for pos in positions]
        
        # Convert to Cartesian
        r = [np.exp(e) for e in ell]
        x = [ri * np.cos(th) for ri, th in zip(r, theta)]
        y = [ri * np.sin(th) for ri, th in zip(r, theta)]
        
        # Plot 1: Tachyonic events in phase space
        scatter = ax1.scatter(x, y, c=c_ratios, cmap='plasma', 
                             s=100, alpha=0.7, 
                             vmin=1.0, vmax=max(c_ratios))
        
        # Add critical radius
        r_critical = self.c * self.phi**2 / np.pi
        theta_circle = np.linspace(0, 2*np.pi, 100)
        x_crit = r_critical * np.cos(theta_circle)
        y_crit = r_critical * np.sin(theta_circle)
        ax1.plot(x_crit, y_crit, 'r--', alpha=0.7, linewidth=2,
                label=f'Critical radius (r={r_critical:.2f})')
        
        # Add preferred radii
        r_inner = 1/self.phi
        x_inner = r_inner * np.cos(theta_circle)
        y_inner = r_inner * np.sin(theta_circle)
        ax1.plot(x_inner, y_inner, 'gold', alpha=0.5, linewidth=1,
                label=f'r = 1/φ = {r_inner:.2f}')
        
        r_outer = self.phi - 1
        x_outer = r_outer * np.cos(theta_circle)
        y_outer = r_outer * np.sin(theta_circle)
        ax1.plot(x_outer, y_outer, 'magenta', alpha=0.5, linewidth=1,
                label=f'r = φ-1 = {r_outer:.2f}')
        
        # Labels and title
        ax1.set_xlabel('X = r·cos(θ)')
        ax1.set_ylabel('Y = r·sin(θ)')
        ax1.set_title('Tachyonic Events in Phase Space')
        ax1.legend()
        ax1.axis('equal')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('v/c ratio')
        
        # Plot 2: Velocity vs. radius
        ax2.scatter(r, c_ratios, c=c_ratios, cmap='plasma', 
                   s=50, alpha=0.7, vmin=1.0, vmax=max(c_ratios))
        
        # Add theoretical curve
        r_theory = np.linspace(0, max(r) * 1.1, 100)
        omega = self.phi**2 / np.pi  # Example frequency
        v_theory = r_theory * omega / self.c
        ax2.plot(r_theory, v_theory, 'k-', alpha=0.7, linewidth=2,
                label='Theoretical v/c = r·ω/c')
        
        # Add critical line
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7,
                   label='Light speed (v=c)')
        
        # Add vertical lines at preferred radii
        ax2.axvline(x=r_inner, color='gold', linestyle='-', alpha=0.5,
                   label=f'r = 1/φ = {r_inner:.2f}')
        ax2.axvline(x=r_outer, color='magenta', linestyle='-', alpha=0.5,
                   label=f'r = φ-1 = {r_outer:.2f}')
        ax2.axvline(x=r_critical, color='r', linestyle='--', alpha=0.7,
                   label=f'r_crit = {r_critical:.2f}')
        
        # Labels and title
        ax2.set_xlabel('Radius (r)')
        ax2.set_ylabel('Velocity/c ratio')
        ax2.set_title('Superluminal Velocity vs. Radius')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Ensure y-axis starts at 1.0 (speed of light)
        ax2.set_ylim(bottom=0.9, top=max(c_ratios)*1.1)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {os.path.join(OUTPUT_DIR, filename)}")
        
        return fig
    
    def visualize_energy_and_crystallization(self, filename="energy_crystallization.png"):
        """
        Visualize system energy and crystallization over time.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot energy
        times = np.arange(len(self.energy_history)) * self.dt
        ax1.plot(times, self.energy_history, 'b-', linewidth=2)
        
        # Mark tachyonic events on energy plot
        if self.tachyonic_events:
            tachyon_times = [event['time'] for event in self.tachyonic_events]
            tachyon_energies = [self.energy_history[int(t/self.dt)] 
                              if int(t/self.dt) < len(self.energy_history) else np.nan 
                              for t in tachyon_times]
            ax1.scatter(tachyon_times, tachyon_energies, color='red', s=50, 
                       marker='*', label='Tachyonic events')
        
        # Labels and title
        ax1.set_ylabel('System Energy')
        ax1.set_title('System Energy Over Time')
        ax1.grid(True, alpha=0.3)
        if self.tachyonic_events:
            ax1.legend()
        
        # Plot crystallization counts
        crystal_counts = []
        for i in range(len(self.position_history)):
            # Count crystallized tokens at each step
            if i == 0:
                crystal_counts.append(0)
            else:
                # Use the actual crystallization flags from that time
                t = i * self.dt
                count = sum(1 for event in self.memory_bank if event['time'] <= t)
                crystal_counts.append(count)
        
        ax2.plot(times, crystal_counts, 'g-', linewidth=2)
        
        # Mark tachyonic events on crystallization plot
        if self.tachyonic_events:
            tachyon_times = [event['time'] for event in self.tachyonic_events]
            tachyon_crystals = [crystal_counts[int(t/self.dt)] 
                              if int(t/self.dt) < len(crystal_counts) else np.nan 
                              for t in tachyon_times]
            ax2.scatter(tachyon_times, tachyon_crystals, color='red', s=50, 
                       marker='*', label='Tachyonic events')
        
        # Labels and title
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Crystallized Tokens')
        ax2.set_title('Crystallization Over Time')
        ax2.grid(True, alpha=0.3)
        if self.tachyonic_events:
            ax2.legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {os.path.join(OUTPUT_DIR, filename)}")
        
        return fig
    
    def visualize_hebbian_matrix(self, filename="hebbian_matrix.png"):
        """
        Visualize Hebbian coupling matrix.
        """
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
        
        # Save figure
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {os.path.join(OUTPUT_DIR, filename)}")
        
        return fig
    
    def visualize_pi_by_2_spacing(self, filename="pi_by_2_spacing.png"):
        """
        Visualize the π/2 spacing → helical structure.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Parameters
        r = 1/self.phi
        z_max = 4 * np.pi
        
        # Create four helical trajectories with π/2 phase differences
        t = np.linspace(0, z_max, 1000)
        
        for i, phase_shift in enumerate([0, np.pi/2, np.pi, 3*np.pi/2]):
            theta = self.phi * t + phase_shift
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = t
            
            # Color based on phase
            color = plt.cm.viridis(i/4)
            
            ax.plot(x, y, z, linewidth=2, color=color, 
                   label=f'Phase {i+1}: θ₀ = {phase_shift:.2f}')
            
            # Mark each 90° rotation
            for j in range(8):
                idx = int(j * len(t)/8)
                ax.scatter(x[idx], y[idx], z[idx], color=color, s=50)
        
        # Add vertical connecting lines at 90° intervals
        for j in range(8):
            idx = int(j * len(t)/8)
            z_val = t[idx]
            points_x = []
            points_y = []
            points_z = []
            
            for phase_shift in [0, np.pi/2, np.pi, 3*np.pi/2]:
                theta = self.phi * t[idx] + phase_shift
                points_x.append(r * np.cos(theta))
                points_y.append(r * np.sin(theta))
                points_z.append(z_val)
            
            # Add the first point again to close the loop
            points_x.append(points_x[0])
            points_y.append(points_y[0])
            points_z.append(points_z[0])
            
            ax.plot(points_x, points_y, points_z, 'k--', alpha=0.5)
        
        # Labels and title
        ax.set_xlabel('X = r·cos(θ)')
        ax.set_ylabel('Y = r·sin(θ)')
        ax.set_zlabel('Z')
        ax.set_title('π/2 Spacing → Helical Structure\n' +
                    'Setting phase increment to π/2 creates helix with pitch 4v/π')
        
        # Legend
        ax.legend()
        
        # View angle
        ax.view_init(30, 45)
        
        # Equal aspect ratio for x and y
        ax.set_box_aspect([1, 1, 2])
        
        # Save figure
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {os.path.join(OUTPUT_DIR, filename)}")
        
        return fig
    
    def visualize_born_rule_helix(self, filename="born_rule_helix.png"):
        """
        Visualize how the Born rule constraint affects helical trajectory.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), subplot_kw={'projection': '3d'})
        
        # Time steps
        t = np.linspace(0, 6*np.pi, 1000)
        
        # Base parameters
        omega_theta = 1.0
        omega_z = 2*np.pi/(self.phi**3)
        
        # 1. Unconstrained helical trajectory
        r1 = 1/self.phi  # Fixed radius
        
        x1 = r1 * np.cos(omega_theta * t)
        y1 = r1 * np.sin(omega_theta * t)
        z1 = np.linspace(0, 1, len(t))  # Linear z progression
        
        ax1.plot(x1, y1, z1, 'b-', linewidth=2, label='Unconstrained')
        ax1.set_title('Unconstrained Helical Trajectory\n' + 
                     f'r = 1/φ = {r1:.3f} (constant)')
        
        # 2. Born rule constrained trajectory
        # Where r² + z² = 1
        z2 = np.linspace(0, 0.9, len(t))  # Vary z from 0 to 0.9
        r2 = np.sqrt(1 - z2**2)  # Born rule constraint
        
        x2 = r2 * np.cos(omega_theta * t)
        y2 = r2 * np.sin(omega_theta * t)
        
        ax2.plot(x2, y2, z2, 'g-', linewidth=2, label='Born rule constrained')
        ax2.set_title('Born Rule Constrained Trajectory\n' + 
                     'r² + z² = 1')
        
        # Common settings for both subplots
        for ax in [ax1, ax2]:
            # Labels
            ax.set_xlabel('X = r·cos(θ)')
            ax.set_ylabel('Y = r·sin(θ)')
            ax.set_zlabel('Z')
            
            # Equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # View angle
            ax.view_init(30, 45)
            
            # Add Born rule unit sphere for reference
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x_sphere = np.cos(u)*np.sin(v)
            y_sphere = np.sin(u)*np.sin(v)
            z_sphere = np.cos(v)
            ax.plot_surface(x_sphere, y_sphere, z_sphere, color='r', alpha=0.1)
            
            # Add legend
            ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {os.path.join(OUTPUT_DIR, filename)}")
        
        return fig
    
    def generate_all_visualizations(self):
        """
        Generate all visualizations.
        """
        logger.info("Generating all visualizations...")
        
        # Basic visualizations
        self.visualize_log_cylindrical()
        self.visualize_helical_trajectories()
        self.visualize_energy_and_crystallization()
        self.visualize_hebbian_matrix()
        
        # Special visualizations
        self.visualize_pi_by_2_spacing()
        self.visualize_born_rule_helix()
        
        # Tachyonic visualizations
        if self.tachyonic_events:
            self.visualize_tachyonic_events()
        
        logger.info("All visualizations generated successfully!")
    
    def run_demo(self, num_tokens=50, steps=100, use_wordnet=True):
        """
        Run a complete demonstration of log-form helical dynamics.
        """
        logger.info(f"Starting log-form helical dynamics demo with {num_tokens} tokens...")
        
        # Initialize tokens
        if use_wordnet:
            try:
                self.initialize_from_wordnet(num_tokens)
            except Exception as e:
                logger.error(f"WordNet initialization failed: {e}")
                self.initialize_random(num_tokens)
        else:
            self.initialize_random(num_tokens)
        
        # Evolve system
        logger.info(f"Evolving system for {steps} steps...")
        self.evolve_system(steps)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        self.generate_all_visualizations()
        
        # Report statistics
        crystallized = np.sum(self.crystallization_flags[:self.active_tokens])
        tachyonic = len(self.tachyonic_events)
        
        logger.info(f"Demo completed successfully!")
        logger.info(f"Statistics:")
        logger.info(f"- Total tokens: {self.active_tokens}")
        logger.info(f"- Crystallized tokens: {crystallized}")
        logger.info(f"- Tachyonic events: {tachyonic}")
        logger.info(f"- Final energy: {self.energy_history[-1] if self.energy_history else 'N/A'}")
        
        return {
            'active_tokens': self.active_tokens,
            'crystallized': crystallized,
            'tachyonic_events': tachyonic,
            'final_energy': self.energy_history[-1] if self.energy_history else None
        }


# Example usage
if __name__ == "__main__":
    print("=== LOG-FORM HELICAL DYNAMICS DEMONSTRATION ===")
    print(f"φ = {PHI:.6f}")
    print(f"GAP = {GAP:.6f}")
    
    # Create model
    model = LogFormHelicalDynamics(max_tokens=100, field_dim=64)
    
    # Run demo
    results = model.run_demo(num_tokens=50, steps=100, use_wordnet=NLTK_AVAILABLE)
    
    print("\nResults:")
    print(f"- Active tokens: {results['active_tokens']}")
    print(f"- Crystallized tokens: {results['crystallized']}")
    print(f"- Tachyonic events: {results['tachyonic_events']}")
    if results['final_energy'] is not None:
        print(f"- Final energy: {results['final_energy']:.6f}")
    else:
        print("- Final energy: N/A")
    
    print(f"\nVisualizations saved to {OUTPUT_DIR}/")
    print("Demo completed successfully!")
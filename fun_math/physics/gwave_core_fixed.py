# gwave_core_fixed.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Any, Union
from pathlib import Path
import time
import os
import multiprocessing as mp
from functools import partial
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- universal constants ----------
PHI = (1 + np.sqrt(5)) / 2
DT  = PHI ** -2             # Δt = φ⁻²
EPS = 1e-10
GOLDEN_THR = PHI            # tunnelling threshold
OUTPUT_DIR = "outputs/gwave/physics"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- configuration ----------
@dataclass
class GwaveConfig:
    max_tokens      : int   = 256
    m0              : float = 1.0
    lambda_cutoff   : float = PHI**2
    ell_max         : float = 10.0
    k_bound         : float = 0.5
    eta_hebb        : float = 0.01
    gamma_hebb      : float = 0.001
    sigma_theta     : float = np.pi / PHI
    sigma_gate      : float = np.pi / PHI
    sigma_hebb      : float = 0.001
    energy_clip     : float = 1e6      # hard safety cap
    # Advanced features
    levy_alpha      : float = PHI      # Lévy stable distribution parameter
    c               : float = 1.0      # Semantic speed of light
    omega_z         : float = 2 * np.pi / (PHI**3)  # z-rotation frequency
    resonance_temp  : float = 0.1      # Temperature for resonance
    field_dim       : int   = 16       # Quantum field dimension
    num_processes   : int   = max(1, mp.cpu_count() - 1)  # Parallel processes
    track_tachyonic : bool  = True     # Track tachyonic events
    track_phi_n     : int   = 8        # Number of φⁿ layers to track
    mass_min        : float = 0.01     # Minimum mass to prevent divide by zero
    
# ---------- helper functions ----------
def wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]"""
    return (a + np.pi) % (2 * np.pi) - np.pi

def stable_levy(alpha: float = PHI, scale: float = 1.0) -> float:
    """Cauchy-like draw approximating Lévy(α=φ) with safety bounds."""
    # Use arctan for stability, limiting the input to avoid numerical issues
    u = np.clip(np.random.rand(), 0.01, 0.99) - 0.5  # Avoid exact 0.5
    return scale * np.tan(np.pi * u)

def fibonacci_sequence(n: int) -> List[int]:
    """Generate Fibonacci sequence of length n."""
    fib = [1, 1]
    for i in range(n - 2):
        fib.append(fib[-1] + fib[-2])
    return fib

def safe_divide(a, b, default=0.0):
    """Safely divide a by b, returning default if b is too close to zero."""
    if np.abs(b) < EPS:
        return default
    return a / b

def safe_sqrt(x):
    """Safely calculate square root, handling negative values."""
    return np.sqrt(np.maximum(x, EPS))

def safe_log(x):
    """Safely calculate logarithm, handling non-positive values."""
    return np.log(np.maximum(x, EPS))

def safe_exp(x, max_exp=50):
    """Safely calculate exponential, preventing overflow."""
    return np.exp(np.clip(x, -max_exp, max_exp))

def is_valid(x):
    """Check if value is valid (not NaN or Inf)."""
    if np.isscalar(x):
        return np.isfinite(x)
    return np.all(np.isfinite(x))

# ---------- core engine ----------
class GwaveCore:
    def __init__(self, cfg: GwaveConfig):
        self.cfg = cfg
        n = cfg.max_tokens
        # state tensors
        self.pos   = np.zeros((n, 3))        # (ℓ, θ, z)
        self.mass  = np.zeros(n)
        self.s     = np.zeros(n)
        self.froz  = np.zeros(n, dtype=bool)
        self.H     = np.zeros((n, n))
        self.N_act = 0
        # time
        self.t     = 0.0
        # history (optional)
        self.traj  : List[np.ndarray] = []
        self.energy_history : List[float] = []
        
        # Advanced features
        self.vortex_field = np.zeros((n, 3))  # Vorticity field
        self.tachyonic_events : List[Dict[str, Any]] = []
        
        # Phi^n layer tracking
        self.phi_n_layers = [PHI**i for i in range(1, cfg.track_phi_n + 1)]
        self.phi_n_counts = np.zeros((cfg.track_phi_n, 2))  # [tokens in band, crystallized tokens]
        
        # Quantum field
        self.field = self._initialize_quantum_field()
        
        # Debug counters for numerical stability
        self.nan_counts = {
            'forces': 0,
            'positions': 0,
            'energy': 0
        }
        
        logger.info(f"Initialized GwaveCore with {cfg.max_tokens} max tokens")
        logger.info(f"Using {cfg.num_processes} processes for parallel computation")

    def _initialize_quantum_field(self) -> Dict[str, np.ndarray]:
        """Initialize quantum field with bounded Lévy distribution."""
        n = self.cfg.max_tokens
        field_dim = self.cfg.field_dim
        
        # Initialize complex wave function
        psi_real = np.zeros((n, field_dim))
        psi_imag = np.zeros((n, field_dim))
        
        # Bounded Lévy samples
        for i in range(n):
            for j in range(field_dim):
                amp = stable_levy(alpha=self.cfg.levy_alpha, scale=0.5)  # Reduced scale for stability
                # Soft clipping
                amp = np.tanh(amp)
                phase = np.random.uniform(0, 2 * np.pi)
                psi_real[i, j] = amp * np.cos(phase)
                psi_imag[i, j] = amp * np.sin(phase)
        
        # Normalize
        norm = safe_sqrt(np.sum(psi_real**2 + psi_imag**2))
        psi_real = safe_divide(psi_real, norm)
        psi_imag = safe_divide(psi_imag, norm)
        
        # Initialize other field components
        momentum = np.zeros((n, field_dim, 2))  # Real and imaginary components
        energy = np.zeros(n)
        entropy = np.zeros(n)
        
        return {
            'psi_real': psi_real,
            'psi_imag': psi_imag,
            'momentum': momentum,
            'energy': energy,
            'entropy': entropy
        }

    # ---------- token I/O ----------
    def add_token(self, ell: float, theta: float, z: float = 0.0,
                  strength: float = 1.0) -> int:
        """Add a token at position (ℓ, θ, z) with specified field strength."""
        idx = self.N_act
        if idx >= self.cfg.max_tokens:
            raise RuntimeError("token cap reached")
        
        # Ensure values are finite
        if not (is_valid(ell) and is_valid(theta) and is_valid(z) and is_valid(strength)):
            logger.warning(f"Invalid token values: ell={ell}, theta={theta}, z={z}, strength={strength}")
            ell = 0.0
            theta = 0.0
            z = 0.0
            strength = 1.0
        
        self.pos[idx]  = (ell, theta, z)
        # Ensure mass is never zero
        self.mass[idx] = max(self.cfg.m0 * ell, self.cfg.mass_min)
        self.s[idx]    = strength
        self.N_act += 1
        return idx

    # ---------- distance helpers ----------
    def _d2(self, i: int, j: int) -> float:
        """Compute squared distance between tokens i and j."""
        dℓ = self.pos[i, 0] - self.pos[j, 0]
        dθ = wrap_angle(self.pos[i, 1] - self.pos[j, 1])
        return dℓ*dℓ + dθ*dθ + EPS  # Add EPS to avoid division by zero

    def _u_vec(self, i: int, j: int) -> np.ndarray:
        """Compute unit vector from token i to token j."""
        dℓ = self.pos[j, 0] - self.pos[i, 0]
        dθ = wrap_angle(self.pos[j, 1] - self.pos[i, 1])
        norm = safe_sqrt(dℓ*dℓ + dθ*dθ)
        return np.array([dℓ, dθ]) / (norm + EPS)  # Safe division

    # ---------- force components ----------
    def _F_rep(self, i: int) -> np.ndarray:
        """Compute repulsive force on token i."""
        F = np.zeros(2)
        for j in range(self.N_act):
            if i == j or self.froz[j]:
                continue
            d2 = self._d2(i, j)
            
            # Add resonance term (from Repulsion Attention)
            r_i = safe_exp(self.pos[i, 0])
            r_j = safe_exp(self.pos[j, 0])
            
            # Safe trig calculations
            if is_valid(self.pos[i, 1]) and is_valid(self.pos[j, 1]):
                cos_i = np.cos(self.pos[i, 1])
                sin_j = np.sin(self.pos[j, 1])
            else:
                cos_i, sin_j = 0.0, 0.0
                
            resonance = np.abs(r_i * cos_i - r_j * sin_j + PHI / 2)
            
            # Safe mass check
            mass_j = max(self.mass[j], self.cfg.mass_min)
            
            # Force magnitude with resonance modulation (with safety)
            Fmag = (self.s[i] * self.s[j] / (4 * np.pi * d2 * mass_j) * 
                   safe_exp(-safe_sqrt(d2) / self.cfg.lambda_cutoff) * 
                   safe_exp(-resonance**2 / (2 * self.cfg.resonance_temp)))
            
            # Clip to prevent extreme forces
            Fmag = np.clip(Fmag, -1e3, 1e3)
            
            # Check for validity before applying
            if is_valid(Fmag):
                u_vec = self._u_vec(i, j)
                F += Fmag * u_vec
            else:
                logger.debug(f"Invalid force magnitude: {Fmag} between tokens {i} and {j}")
                
        # Final validity check
        if not is_valid(F):
            logger.warning(f"Invalid repulsive force for token {i}: {F}")
            F = np.zeros(2)
            self.nan_counts['forces'] += 1
            
        return F

    def _pitch(self, i: int) -> float:
        """Compute pitch angle for token i based on Hebbian connections."""
        # Calculate complex value representing pitch direction
        c = 0
        for j in range(self.N_act):
            # Skip invalid angles
            if not is_valid(self.pos[j, 1]):
                continue
                
            connection = self.H[i, j]
            if connection > 0:
                c += connection * np.exp(1j * self.pos[j, 1])
        
        if np.abs(c) > EPS:
            return np.angle(c)
        else:
            return self.pos[i, 1]  # Default to current angle

    def _F_hebb(self, i: int) -> np.ndarray:
        """Compute Hebbian force on token i."""
        # Check for valid angle
        if not is_valid(self.pos[i, 1]):
            return np.zeros(2)
            
        pitch = self._pitch(i)
        Δθ = wrap_angle(self.pos[i, 1] - pitch)
        
        # Limit extreme values
        Δθ = np.clip(Δθ, -np.pi/2, np.pi/2)
        
        k1, k2 = 0.5, 0.1
        F_theta = -(k1 * Δθ + k2 * Δθ**3)
        
        # Clip to prevent extreme forces
        F_theta = np.clip(F_theta, -10.0, 10.0)
        
        return np.array([0.0, F_theta])

    def _F_bound(self, i: int) -> np.ndarray:
        """Compute boundary force on token i."""
        if not is_valid(self.pos[i, 0]):
            return np.zeros(2)
            
        if self.pos[i, 0] > self.cfg.ell_max:
            force = -self.cfg.k_bound * (self.pos[i, 0] - self.cfg.ell_max)
            # Clip to prevent extreme forces
            force = np.clip(force, -10.0, 10.0)
            return np.array([force, 0.0])
        return np.zeros(2)

    def _F_quantum(self, i: int) -> np.ndarray:
        """Compute quantum force on token i (simplified version)."""
        # For now, just return zero
        return np.zeros(2)

    def _F_total(self, i: int) -> np.ndarray:
        """Compute total force on token i."""
        if self.froz[i] or not self._gate(i):
            return np.zeros(2)
        
        # Get individual force components
        F_rep = self._F_rep(i)
        F_hebb = self._F_hebb(i)
        F_bound = self._F_bound(i)
        F_quantum = self._F_quantum(i)
        
        # Sum all forces
        F_total = F_rep + F_hebb + F_bound + F_quantum
        
        # Validate and clip extreme values
        if not is_valid(F_total):
            logger.warning(f"Invalid total force for token {i}. Components: rep={F_rep}, hebb={F_hebb}, bound={F_bound}")
            F_total = np.zeros(2)
            self.nan_counts['forces'] += 1
        else:
            # Apply safety clipping to total force magnitude
            mag = np.linalg.norm(F_total)
            if mag > 100:
                F_total = F_total * (100 / mag)
        
        return F_total

    # ---------- gating ----------
    def _gate(self, i: int) -> bool:
        """Check if token i is active based on rotor gate."""
        if not is_valid(self.pos[i, 1]) or not is_valid(self.t):
            return False
            
        Z = (self.t * self.cfg.omega_z) % (2*np.pi)
        return (not self.froz[i] and
                abs(wrap_angle(self.pos[i, 1] - Z)) < self.cfg.sigma_gate)

    # ---------- parallel force computation ----------
    def _compute_forces_parallel(self) -> np.ndarray:
        """Compute forces for all tokens in parallel."""
        active_indices = [i for i in range(self.N_act) if self._gate(i)]
        
        if not active_indices:
            return np.zeros((self.N_act, 2))
        
        # Just use serial computation for simplicity
        forces = np.zeros((self.N_act, 2))
        for i in active_indices:
            forces[i] = self._F_total(i)
        return forces

    # ---------- integrator ----------
    def _step_heun(self):
        """Implement Heun (predictor-corrector) integration step."""
        n = self.N_act
        
        # Compute forces - potentially in parallel
        F0 = self._compute_forces_parallel()
        
        # Predictor
        pred = self.pos.copy()
        
        # Calculate acceleration with safe division
        accel = np.zeros((n, 2))
        for i in range(n):
            # Ensure mass is never zero or negative
            mass_i = max(self.mass[i], self.cfg.mass_min)
            accel[i] = F0[i] / mass_i
            
            # Clip extreme acceleration values
            accel[i] = np.clip(accel[i], -100.0, 100.0)
        
        # Apply predictor step
        pred[:n, :2] += accel * DT
        
        # Validate predicted positions
        for i in range(n):
            if not is_valid(pred[i, 0]) or not is_valid(pred[i, 1]):
                logger.warning(f"Invalid predicted position for token {i}: {pred[i]}")
                pred[i] = self.pos[i]  # Revert to original position
                self.nan_counts['positions'] += 1
        
        # Update z with global rotor
        pred[:n, 2] = (self.t * self.cfg.omega_z + DT) % (2*np.pi)
        
        # Recalc forces at predictor
        pos_backup = self.pos.copy()
        self.pos = pred
        F1 = self._compute_forces_parallel()
        self.pos = pos_backup  # restore
        
        # Corrector
        avg_accel = np.zeros((n, 2))
        for i in range(n):
            # Ensure mass is never zero or negative
            mass_i = max(self.mass[i], self.cfg.mass_min)
            
            # Average forces from predictor and initial
            avg_force = 0.5 * (F0[i] + F1[i])
            
            # Calculate acceleration
            avg_accel[i] = avg_force / mass_i
            
            # Clip extreme acceleration values
            avg_accel[i] = np.clip(avg_accel[i], -100.0, 100.0)
        
        # Apply corrector step
        new_pos = self.pos.copy()
        new_pos[:n, :2] += avg_accel * DT
        
        # Validate new positions
        for i in range(n):
            if not is_valid(new_pos[i, 0]) or not is_valid(new_pos[i, 1]):
                logger.warning(f"Invalid corrected position for token {i}: {new_pos[i]}")
                new_pos[i] = self.pos[i]  # Keep old position
                self.nan_counts['positions'] += 1
            else:
                # Clip ell to reasonable bounds
                new_pos[i, 0] = np.clip(new_pos[i, 0], -10.0, 10.0)
        
        # Update positions
        self.pos = new_pos
        
        # Update z with global rotor
        self.pos[:n, 2] = (self.t * self.cfg.omega_z + DT) % (2*np.pi)
        
        # Update mass
        for i in range(n):
            # Ensure mass is never zero or negative
            self.mass[i] = max(self.cfg.m0 * self.pos[i, 0], self.cfg.mass_min)
        
        # Update vortex field - compute curl of velocity field
        self._update_vortex_field(F0, F1)
        
        # Check for tachyonic states
        if self.cfg.track_tachyonic:
            self._check_tachyonic_states(F0)

    def _update_vortex_field(self, F0: np.ndarray, F1: np.ndarray):
        """Update vortex field by computing curl of velocity field."""
        n = self.N_act
        
        # Average force over predictor-corrector
        F_avg = 0.5 * (F0 + F1)
        
        # Calculate velocity field
        v = np.zeros((n, 2))
        for i in range(n):
            # Safe division
            v[i] = safe_divide(F_avg[i], self.mass[i])
        
        # For each active token, compute approximate curl
        for i in range(n):
            if self.froz[i]:
                continue
                
            # Check for valid position and velocity
            if not (is_valid(self.pos[i]) and is_valid(v[i])):
                self.vortex_field[i] = [0, 0, 0]
                continue
                
            # Convert to Cartesian for curl calculation
            r = safe_exp(self.pos[i, 0])
            theta = self.pos[i, 1]
            
            # Radial and angular velocity components
            v_r = v[i, 0]
            v_theta = v[i, 1]
            
            # Safe trig calculations
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            # Convert to Cartesian velocity
            v_x = v_r * cos_theta - r * v_theta * sin_theta
            v_y = v_r * sin_theta + r * v_theta * cos_theta
            
            # Simplified curl calculation in 2D
            curl_z = r * v_theta  # Main component of vorticity
            
            # Validate curl value
            if is_valid(curl_z):
                # Clip to reasonable bounds
                curl_z = np.clip(curl_z, -100.0, 100.0)
                self.vortex_field[i] = [0, 0, curl_z]  # Only z-component for 2D flow
            else:
                self.vortex_field[i] = [0, 0, 0]

    def _check_tachyonic_states(self, forces: np.ndarray):
        """Check for tachyonic states where phase velocity exceeds c."""
        n = self.N_act
        
        for i in range(n):
            if self.froz[i]:
                continue
                
            # Check for valid position
            if not is_valid(self.pos[i]):
                continue
                
            # Calculate phase velocity
            r = safe_exp(self.pos[i, 0])
            
            # Safe division
            v_theta = safe_divide(forces[i, 1], self.mass[i])
            
            v_phase = r * abs(v_theta)
            
            # Check if valid and superluminal
            if is_valid(v_phase) and v_phase > self.cfg.c:
                # Record tachyonic event
                self.tachyonic_events.append({
                    'token': i,
                    'time': self.t,
                    'position': self.pos[i].copy(),
                    'velocity': v_phase,
                    'c_ratio': v_phase / self.cfg.c
                })
                
                # Calculate proper time change (imaginary for superluminal)
                dtau_dt = safe_sqrt(abs(1 - v_phase**2/self.cfg.c**2))
                
                logger.debug(f"Tachyonic event: token {i}, v/c = {v_phase/self.cfg.c:.2f}")

    def _enforce_born_rule(self):
        """Enforce Born rule constraint: r² + z² = 1."""
        n = self.N_act
        
        for i in range(n):
            if self.froz[i]:
                continue
                
            # Check for valid position
            if not is_valid(self.pos[i]):
                continue
                
            # Convert from log-radial to linear
            ell, z = self.pos[i, 0], self.pos[i, 2]
            r = safe_exp(ell)
            
            # Map z from [0, 2π) to [0, 1] for Born rule
            z_norm = 0.5 * (1 + np.sin(z))
            
            # Calculate norm
            norm = safe_sqrt(r**2 + z_norm**2)
            
            # Normalize
            r_new = safe_divide(r, norm)
            z_norm_new = safe_divide(z_norm, norm)
            
            # Convert back
            ell_new = safe_log(r_new)
            z_new = np.arcsin(np.clip(2 * z_norm_new - 1, -1.0, 1.0))
            
            # Validate new position
            if is_valid(ell_new) and is_valid(z_new):
                # Update position
                self.pos[i, 0] = ell_new
                self.pos[i, 2] = z_new
                
                # Update mass
                self.mass[i] = max(self.cfg.m0 * ell_new, self.cfg.mass_min)
            else:
                logger.debug(f"Invalid Born rule result for token {i}: ell_new={ell_new}, z_new={z_new}")

    # ---------- public evolve ----------
    def evolve(self, steps: int = 100):
        """Evolve the system for a number of timesteps."""
        start_time = time.time()
        self.traj.append(self.pos.copy())
        
        # Calculate and store initial energy
        energy = self._compute_energy()
        self.energy_history.append(energy)
        
        for step in range(steps):
            self._step_heun()
            self._enforce_born_rule()
            self._update_hebb()
            self._check_freeze_and_tunnel()
            self._update_phi_n_tracking()
            
            # Check energy safety
            try:
                self._energy_safety()
            except RuntimeError as e:
                logger.warning(f"Energy safety triggered: {e}")
                break
            
            # Store trajectory and energy
            self.traj.append(self.pos.copy())
            energy = self._compute_energy()
            self.energy_history.append(energy)
            
            # Progress logging
            if (step + 1) % 10 == 0 or step == steps - 1:
                elapsed = time.time() - start_time
                crystallized = np.sum(self.froz[:self.N_act])
                tachyonic = len(self.tachyonic_events)
                nan_forces = self.nan_counts['forces']
                nan_positions = self.nan_counts['positions']
                
                logger.info(f"Step {step+1}/{steps} completed in {elapsed:.2f}s. "
                           f"Crystallized: {crystallized}, Tachyonic events: {tachyonic}, "
                           f"Numerical issues: forces={nan_forces}, positions={nan_positions}")
            
            self.t += DT

    def _update_phi_n_tracking(self):
        """Update tracking of tokens in φⁿ layers."""
        n = self.N_act
        
        # Reset counts
        self.phi_n_counts = np.zeros((self.cfg.track_phi_n, 2))
        
        for i in range(n):
            # Skip invalid positions
            if not is_valid(self.pos[i, 0]):
                continue
                
            r = safe_exp(self.pos[i, 0])
            
            # Check each φⁿ layer
            for j, phi_n in enumerate(self.phi_n_layers):
                # Check if token is within band around φⁿ
                band_width = 0.1 * phi_n
                if abs(r - phi_n) < band_width:
                    # Increment total count
                    self.phi_n_counts[j, 0] += 1
                    
                    # If crystallized, increment crystallized count
                    if self.froz[i]:
                        self.phi_n_counts[j, 1] += 1

    # ---------- Hebb, freeze, tunnel ----------
    def _update_hebb(self):
        """Update Hebbian coupling matrix."""
        n = self.N_act
        for i in range(n):
            for j in range(i+1, n):
                # Skip invalid positions
                if not (is_valid(self.pos[i]) and is_valid(self.pos[j])):
                    continue
                    
                dθ = wrap_angle(self.pos[i, 1] - self.pos[j, 1])
                Θ  = np.cos(dθ/2)**2 * safe_exp(-dθ**2/(2*self.cfg.sigma_theta**2))
                dℓ = abs(self.pos[i, 0] - self.pos[j, 0])
                Φ  = safe_exp(-dℓ/self.cfg.lambda_cutoff) * safe_exp(-(self.pos[i, 0] + self.pos[j, 0])/2)
                
                dH = (self.cfg.eta_hebb * Θ * Φ
                      - self.cfg.gamma_hebb * self.H[i, j]
                      + np.random.normal(0, self.cfg.sigma_hebb))
                
                # Clip to prevent extreme values
                dH = np.clip(dH, -0.1, 0.1)
                
                # Update matrix
                self.H[i, j] += dH * DT
                self.H[j, i]  = self.H[i, j]
                
                # Clip to reasonable bounds
                self.H[i, j] = np.clip(self.H[i, j], 0.0, 1.0)
                self.H[j, i] = self.H[i, j]

    def _check_freeze_and_tunnel(self):
        """Check for crystallization and tunneling conditions."""
        n = self.N_act
        for i in range(n):
            if self.froz[i] or not self._gate(i):
                continue
                
            # Skip invalid positions
            if not is_valid(self.pos[i]):
                continue
                
            # tiny motion ⇒ freeze
            force_mag = np.linalg.norm(self._F_rep(i))
            if is_valid(force_mag) and force_mag < PHI**-3:
                self.froz[i] = True
                logger.debug(f"Token {i} crystallized at t={self.t:.3f}")
                continue
                
            # tunnelling
            if len(self.traj) > 1:  # Need at least two trajectory points
                prev_pos = self.traj[-2][i]
                
                # Skip invalid previous positions
                if not is_valid(prev_pos):
                    continue
                    
                dpos = self.pos[i] - prev_pos
                
                # Safe division
                ratio = safe_divide(abs(dpos[1]), abs(dpos[0]) + EPS)
                
                if ratio > GOLDEN_THR:
                    # Tunnel to opposite angle
                    self.pos[i, 1] = wrap_angle(self.pos[i, 1] + np.pi)
                    
                    # Apply Lévy jump to radius (with safety)
                    levy_jump = stable_levy(alpha=self.cfg.levy_alpha, scale=0.5)  # Reduced scale
                    levy_jump = np.clip(levy_jump, -2.0, 2.0)  # Limit extreme jumps
                    
                    self.pos[i, 0] += levy_jump
                    
                    # Ensure ell is within reasonable bounds
                    self.pos[i, 0] = np.clip(self.pos[i, 0], -5.0, 5.0)
                    
                    # Update mass
                    self.mass[i] = max(self.cfg.m0 * self.pos[i, 0], self.cfg.mass_min)
                    
                    logger.debug(f"Token {i} tunneled at t={self.t:.3f}")

    def _compute_energy(self) -> float:
        """Compute total energy of the system."""
        # Kinetic energy
        KE = 0.0
        
        # Potential energy from repulsion
        PE_rep = 0.0
        for i in range(self.N_act):
            for j in range(i+1, self.N_act):
                # Skip frozen tokens
                if self.froz[i] or self.froz[j]:
                    continue
                    
                # Skip invalid positions
                if not (is_valid(self.pos[i]) and is_valid(self.pos[j])):
                    continue
                    
                dist = safe_sqrt(self._d2(i, j))
                PE_term = safe_divide(self.s[i] * self.s[j], dist)
                
                # Validate and add
                if is_valid(PE_term):
                    PE_rep += PE_term
        
        # Potential energy from Hebbian
        PE_hebb = 0.0
        for i in range(self.N_act):
            if self.froz[i]:
                continue
                
            # Skip invalid positions
            if not is_valid(self.pos[i]):
                continue
                
            # Calculate pitch angle
            pitch_i = self._pitch(i)
            
            # Angular difference
            delta_theta = wrap_angle(self.pos[i, 1] - pitch_i)
            
            # Limit extreme values
            delta_theta = np.clip(delta_theta, -np.pi/2, np.pi/2)
            
            # Potential
            kappa = 0.5
            lambda_hebb = 0.1
            PE_term = 0.5 * kappa * delta_theta**2 + 0.25 * lambda_hebb * delta_theta**4
            
            # Validate and add
            if is_valid(PE_term):
                PE_hebb += PE_term
        
        # Total energy
        energy = KE + PE_rep + PE_hebb
        
        # Final validation
        if not is_valid(energy):
            logger.warning(f"Invalid energy calculated: {energy}")
            self.nan_counts['energy'] += 1
            return 0.0
            
        return energy

    def _energy_safety(self):
        """Safety check to prevent energy runaway."""
        E = self._compute_energy()
        if E > self.cfg.energy_clip:
            raise RuntimeError(f"energy runaway – params unstable: E = {E}")

    # ---------- API utility ----------
    def dump(self, out: Path):
        """Save model state to file."""
        np.savez(out,
                 cfg=asdict(self.cfg),
                 pos=self.pos[:self.N_act],
                 mass=self.mass[:self.N_act],
                 H=self.H[:self.N_act, :self.N_act],
                 froz=self.froz[:self.N_act],
                 vortex=self.vortex_field[:self.N_act],
                 tachyonic=self.tachyonic_events,
                 phi_n_counts=self.phi_n_counts)

    # ---------- Visualization methods ----------
    def visualize_vortex_field(self, filename="vortex_field.png"):
        """
        Visualize the vortex field in phase space.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Convert to Cartesian
        r = np.exp(self.pos[:self.N_act, 0])
        theta = self.pos[:self.N_act, 1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Plot tokens
        scatter = ax.scatter(x, y, 
                           c=['blue' if f else 'red' for f in self.froz[:self.N_act]], 
                           s=50, alpha=0.7)
        
        # Plot vortex field
        vorticity = self.vortex_field[:self.N_act, 2]
        # Normalize for better visualization
        if np.any(vorticity != 0):
            max_vorticity = np.max(np.abs(vorticity[np.isfinite(vorticity)]))
            if max_vorticity > 0:
                vorticity_norm = np.clip(vorticity / max_vorticity, -1.0, 1.0)
                
                # Draw arrows proportional to vorticity
                for i in range(self.N_act):
                    if self.froz[i] or not is_valid(vorticity[i]):
                        continue
                        
                    # Create tangential vector to show rotation
                    v_scale = vorticity_norm[i] * 0.2 * r[i]
                    dx = -v_scale * np.sin(theta[i])
                    dy = v_scale * np.cos(theta[i])
                    
                    # Check for valid vector
                    if is_valid(dx) and is_valid(dy):
                        ax.arrow(x[i], y[i], dx, dy, 
                               head_width=0.05, head_length=0.08, 
                               fc='green', ec='green', alpha=0.6)
        
        # Add preferred radii
        theta_circle = np.linspace(0, 2*np.pi, 100)
        
        # Inner band at r = 1/φ
        r_inner = 1/PHI
        x_inner = r_inner * np.cos(theta_circle)
        y_inner = r_inner * np.sin(theta_circle)
        ax.plot(x_inner, y_inner, 'gold', alpha=0.5, linewidth=1,
               label=f'r = 1/φ = {r_inner:.3f}')
        
        # Outer band at r = φ-1
        r_outer = PHI - 1
        x_outer = r_outer * np.cos(theta_circle)
        y_outer = r_outer * np.sin(theta_circle)
        ax.plot(x_outer, y_outer, 'magenta', alpha=0.5, linewidth=1,
               label=f'r = φ-1 = {r_outer:.3f}')
        
        # φⁿ bands
        for i, phi_n in enumerate(self.phi_n_layers):
            if i % 2 == 0:  # Plot every other layer to avoid clutter
                x_phi_n = phi_n * np.cos(theta_circle)
                y_phi_n = phi_n * np.sin(theta_circle)
                ax.plot(x_phi_n, y_phi_n, 'cyan', alpha=0.3, linewidth=1,
                       label=f'r = φ^{i+1} = {phi_n:.3f}')
        
        # Critical radius for tachyonic behavior
        r_critical = self.cfg.c * PHI**2 / np.pi
        x_crit = r_critical * np.cos(theta_circle)
        y_crit = r_critical * np.sin(theta_circle)
        ax.plot(x_crit, y_crit, 'red', alpha=0.5, linewidth=1, linestyle='--',
               label=f'r_crit = {r_critical:.3f}')
        
        # Labels and title
        ax.set_xlabel('X = r·cos(θ)')
        ax.set_ylabel('Y = r·sin(θ)')
        ax.set_title('Vortex Field in Phase Space')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {os.path.join(OUTPUT_DIR, filename)}")
        
        return fig

    def visualize_tachyonic_trajectories(self, filename="tachyonic_trajectories.png"):
        """
        Visualize tachyonic helical trajectories in phase space.
        """
        import matplotlib.pyplot as plt
        
        if not self.tachyonic_events:
            logger.info("No tachyonic events to visualize")
            return None
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get unique tokens that experienced tachyonic events
        tachyonic_tokens = set(event['token'] for event in self.tachyonic_events)
        
        # Track all trajectories but highlight tachyonic ones
        for i in range(self.N_act):
            # Extract trajectory
            traj = np.array([pos[i] for pos in self.traj])
            
            # Filter out invalid positions
            valid_mask = np.all(np.isfinite(traj), axis=1)
            if not np.any(valid_mask):
                continue
                
            traj = traj[valid_mask]
            
            # Convert from log-cylindrical to Cartesian
            ell = traj[:, 0]
            theta = traj[:, 1]
            z = traj[:, 2]
            
            r = np.exp(ell)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Determine color and style based on whether token is tachyonic
            if i in tachyonic_tokens:
                color = 'red'
                linewidth = 2
                label = f'Tachyonic token {i}'
                
                # Find tachyonic events for this token
                token_events = [event for event in self.tachyonic_events if event['token'] == i]
                
                # Mark tachyonic events
                for event in token_events:
                    # Find closest trajectory point
                    t_idx = int(event['time'] / DT)
                    if t_idx < len(x) and is_valid(x[t_idx]) and is_valid(y[t_idx]) and is_valid(z[t_idx]):
                        # Mark with sphere
                        ax.scatter([x[t_idx]], [y[t_idx]], [z[t_idx]], 
                                  color='purple', s=100, alpha=0.7)
                        
                        # Add velocity vector
                        v_phase = event['velocity']
                        scale = 0.2 * v_phase / self.cfg.c
                        dx = -scale * np.sin(theta[t_idx])
                        dy = scale * np.cos(theta[t_idx])
                        dz = 0  # Assuming z-direction is not affected
                        
                        # Check for valid vector
                        if is_valid(dx) and is_valid(dy) and is_valid(dz):
                            ax.quiver(x[t_idx], y[t_idx], z[t_idx], 
                                     dx, dy, dz, 
                                     color='purple', alpha=0.7, 
                                     arrow_length_ratio=0.3)
            else:
                color = 'blue' if self.froz[i] else 'gray'
                linewidth = 0.5
                label = None
                alpha = 0.3
                
                # Only plot non-tachyonic tokens if crystallized
                if not self.froz[i]:
                    continue
                    
                # Plot trajectory
                ax.plot(x, y, z, color=color, linewidth=linewidth, 
                       alpha=alpha, label=label)
                continue
            
            # Plot tachyonic trajectory
            ax.plot(x, y, z, color=color, linewidth=linewidth, label=label)
            
            # Mark start and end
            if len(x) > 0 and is_valid(x[0]) and is_valid(y[0]) and is_valid(z[0]):
                ax.scatter(x[0], y[0], z[0], color='green', s=50)
            if len(x) > 0 and is_valid(x[-1]) and is_valid(y[-1]) and is_valid(z[-1]):
                ax.scatter(x[-1], y[-1], z[-1], color='orange', s=50)
        
        # Plot cylinder at r = 1/φ (inner band)
        theta_circle = np.linspace(0, 2*np.pi, 100)
        z_levels = np.linspace(0, 2*np.pi, 10)
        
        for z_level in z_levels:
            # Inner circle at r = 1/φ
            r_inner = 1/PHI
            x_inner = r_inner * np.cos(theta_circle)
            y_inner = r_inner * np.sin(theta_circle)
            ax.plot(x_inner, y_inner, [z_level]*len(theta_circle), 'gold', alpha=0.3, linewidth=1)
            
            # Outer circle at r = φ-1
            r_outer = PHI - 1
            x_outer = r_outer * np.cos(theta_circle)
            y_outer = r_outer * np.sin(theta_circle)
            ax.plot(x_outer, y_outer, [z_level]*len(theta_circle), 'magenta', alpha=0.3, linewidth=1)
            
            # Critical radius for tachyonic behavior
            r_critical = self.cfg.c * PHI**2 / np.pi
            x_crit = r_critical * np.cos(theta_circle)
            y_crit = r_critical * np.sin(theta_circle)
            ax.plot(x_crit, y_crit, [z_level]*len(theta_circle), 
                   'red', alpha=0.3, linewidth=1, linestyle='--')
        
        # Add reference helix
        t_ref = np.linspace(0, 4*np.pi, 200)
        r_ref = 1/PHI
        x_ref = r_ref * np.cos(PHI * t_ref)
        y_ref = r_ref * np.sin(PHI * t_ref)
        z_ref = t_ref
        ax.plot(x_ref, y_ref, z_ref, 'k--', alpha=0.5, linewidth=1,
               label='Reference helix (φ frequency)')
        
        # Add labels
        ax.set_xlabel('X = r·cos(θ)')
        ax.set_ylabel('Y = r·sin(θ)')
        ax.set_zlabel('Z (rotor phase)')
        ax.set_title('Tachyonic Helical Trajectories in Phase Space')
        
        # Add legend for unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        # Save figure
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {os.path.join(OUTPUT_DIR, filename)}")
        
        return fig
    
    def visualize_phi_n_tracking(self, filename="phi_n_tracking.png"):
        """
        Visualize token distribution across φⁿ layers.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        layers = [f"φ^{i+1}" for i in range(self.cfg.track_phi_n)]
        total_counts = self.phi_n_counts[:, 0]
        crystallized_counts = self.phi_n_counts[:, 1]
        
        # Plot bars
        width = 0.35
        x = np.arange(len(layers))
        ax.bar(x - width/2, total_counts, width, label='Total tokens', color='blue', alpha=0.7)
        ax.bar(x + width/2, crystallized_counts, width, label='Crystallized tokens', color='green', alpha=0.7)
        
        # Add actual φⁿ values as text
        for i, phi_n in enumerate(self.phi_n_layers):
            ax.text(i, max(total_counts[i], crystallized_counts[i]) + 0.5, 
                   f"{phi_n:.3f}", ha='center')
        
        # Add labels and title
        ax.set_xlabel('φⁿ Layer')
        ax.set_ylabel('Token Count')
        ax.set_title('Token Distribution Across φⁿ Layers')
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {os.path.join(OUTPUT_DIR, filename)}")
        
        return fig

    def generate_all_visualizations(self):
        """Generate all visualizations."""
        import matplotlib.pyplot as plt
        
        logger.info("Generating all visualizations...")
        
        # Basic token visualization
        self.visualize_tokens()
        
        # Vortex field
        self.visualize_vortex_field()
        
        # Tachyonic trajectories
        if self.tachyonic_events:
            self.visualize_tachyonic_trajectories()
        
        # Phi^n tracking
        self.visualize_phi_n_tracking()
        
        # Energy history
        self.visualize_energy_history()
        
        logger.info("All visualizations generated successfully!")
    
    def visualize_tokens(self, filename="tokens.png"):
        """Visualize tokens in phase space."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Convert to Cartesian
        r = np.exp(self.pos[:self.N_act, 0])
        theta = self.pos[:self.N_act, 1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Filter out invalid positions
        valid_mask = np.isfinite(x) & np.isfinite(y)
        
        # Color based on crystallization
        colors = ['blue' if f else 'red' for f in self.froz[:self.N_act]]
        
        # Size based on field strength
        max_s = np.max(self.s[:self.N_act]) if self.N_act > 0 else 1.0
        sizes = 50 * self.s[:self.N_act] / (max_s + EPS)
        
        # Plot tokens
        scatter = ax.scatter(x[valid_mask], y[valid_mask], 
                           c=[colors[i] for i in range(len(colors)) if valid_mask[i]], 
                           s=[sizes[i] for i in range(len(sizes)) if valid_mask[i]], 
                           alpha=0.7)
        
        # Add preferred radii
        theta_circle = np.linspace(0, 2*np.pi, 100)
        
        # Inner band at r = 1/φ
        r_inner = 1/PHI
        x_inner = r_inner * np.cos(theta_circle)
        y_inner = r_inner * np.sin(theta_circle)
        ax.plot(x_inner, y_inner, 'gold', alpha=0.5, linewidth=2,
               label=f'r = 1/φ = {r_inner:.3f}')
        
        # Outer band at r = φ-1
        r_outer = PHI - 1
        x_outer = r_outer * np.cos(theta_circle)
        y_outer = r_outer * np.sin(theta_circle)
        ax.plot(x_outer, y_outer, 'magenta', alpha=0.5, linewidth=2,
               label=f'r = φ-1 = {r_outer:.3f}')
        
        # Critical radius for tachyonic behavior
        r_critical = self.cfg.c * PHI**2 / np.pi
        x_crit = r_critical * np.cos(theta_circle)
        y_crit = r_critical * np.sin(theta_circle)
        ax.plot(x_crit, y_crit, 'red', alpha=0.5, linewidth=2, linestyle='--',
               label=f'r_crit = {r_critical:.3f}')
        
        # Add legend
        legend1 = ax.legend(loc='upper left')
        
        # Add second legend for token types
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Active'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Crystallized')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.add_artist(legend1)
        
        # Labels and title
        ax.set_xlabel('X = r·cos(θ)')
        ax.set_ylabel('Y = r·sin(θ)')
        ax.set_title('Tokens in Phase Space')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {os.path.join(OUTPUT_DIR, filename)}")
        
        return fig

    def visualize_energy_history(self, filename="energy_history.png"):
        """Visualize energy history."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter out invalid energy values
        times = np.arange(len(self.energy_history)) * DT
        energies = np.array(self.energy_history)
        valid_mask = np.isfinite(energies)
        
        if not np.any(valid_mask):
            logger.warning("No valid energy values to plot")
            plt.text(0.5, 0.5, "No valid energy data", 
                    ha='center', va='center', transform=ax.transAxes)
        else:
            # Plot energy
            ax.plot(times[valid_mask], energies[valid_mask], 'b-', linewidth=2)
            
            # Mark tachyonic events
            if self.tachyonic_events:
                tachyon_times = np.array([event['time'] for event in self.tachyonic_events])
                tachyon_energies = []
                
                for t in tachyon_times:
                    idx = int(t/DT)
                    if idx < len(energies) and np.isfinite(energies[idx]):
                        tachyon_energies.append(energies[idx])
                    else:
                        tachyon_energies.append(np.nan)
                
                # Filter out NaN values
                valid_tachyon = ~np.isnan(tachyon_energies)
                if np.any(valid_tachyon):
                    ax.scatter(tachyon_times[valid_tachyon], 
                              np.array(tachyon_energies)[valid_tachyon], 
                              color='red', s=50, marker='*', label='Tachyonic events')
                    ax.legend()
        
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.set_title('System Energy Over Time')
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {os.path.join(OUTPUT_DIR, filename)}")
        
        return fig

# ---------- NLTK loader ----------
def text_to_tokens(text: str, core: GwaveCore):
    """
    Load tokens from text using NLTK for tokenization.
    Maps distinct words to tokens in log-cylindrical space.
    """
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text.lower())
    except ImportError:
        # Fallback if NLTK not available
        tokens = text.lower().split()
    
    # Get unique tokens
    vocab = sorted(set(tokens))[:core.cfg.max_tokens]
    
    for i, w in enumerate(vocab):
        # Calculate log-radius based on position in vocabulary
        ell = np.log(1 + i / (len(vocab) + 1))
        
        # Calculate angle - distribute evenly
        theta = 2*np.pi * i / len(vocab)
        
        # z coordinate - based on word length (just a heuristic)
        z = 2*np.pi * len(w) / max(len(word) for word in vocab)
        
        # Add token
        core.add_token(ell, theta, z, 1.0/len(vocab))
    
    logger.info(f"Loaded {len(vocab)} tokens from text")
    return len(vocab)

# ---------- helper function to place tokens at critical radius ----------
def place_tokens_at_critical(core: GwaveCore, num_tokens=16):
    """
    Place tokens at the critical radius for tachyonic events.
    """
    # Calculate critical radius
    r_critical = core.cfg.c * PHI**2 / np.pi
    ell_critical = safe_log(r_critical)
    
    # Distribute tokens evenly around critical radius
    for i in range(num_tokens):
        # Calculate angle - distribute evenly
        theta = 2*np.pi * i / num_tokens
        
        # z coordinate - distribute in range [0, 2π)
        z = 2*np.pi * i / num_tokens
        
        # Add token with slightly varying radius to encourage interaction
        variation = 0.05 * (np.random.random() - 0.5)
        core.add_token(ell_critical + variation, theta, z, 1.0)
    
    logger.info(f"Placed {num_tokens} tokens at critical radius ell = {ell_critical}")
    return num_tokens

# ---------- demo entry ----------
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(
        description="Run Gwave demonstration with advanced features")
    parser.add_argument("--text", type=str,
                        default="the quick brown fox jumps over the lazy dog",
                        help="raw sentence or paragraph")
    parser.add_argument("--steps", type=int, default=100,
                        help="number of evolution steps")
    parser.add_argument("--max_tokens", type=int, default=64,
                        help="maximum number of tokens")
    parser.add_argument("--visualize", action="store_true",
                        help="generate visualizations")
    parser.add_argument("--critical_tokens", type=int, default=0,
                        help="number of tokens to place at critical radius (0 = use text only)")
    args = parser.parse_args()
    
    print("=== GWAVE DEMONSTRATION WITH ADVANCED FEATURES ===")
    print(f"φ = {PHI:.6f}")
    
    # Create config with advanced features
    cfg = GwaveConfig(
        max_tokens=args.max_tokens,
        levy_alpha=PHI,
        track_phi_n=8,
        track_tachyonic=True,
        # Additional parameters for stability
        mass_min=0.01,
        energy_clip=1e8  # Increased to allow more exploration
    )
    
    # Initialize model
    gw = GwaveCore(cfg)
    
    # Decide whether to use text tokens or critical radius tokens
    if args.critical_tokens > 0:
        # Place tokens at critical radius for tachyonic events
        place_tokens_at_critical(gw, args.critical_tokens)
    else:
        # Load tokens from text
        text_to_tokens(args.text, gw)
    
    print(f"Loaded {gw.N_act} tokens")
    
    # Evolve system
    print(f"Evolving system for {args.steps} steps...")
    gw.evolve(args.steps)
    
    # Statistics
    crystallized = np.sum(gw.froz[:gw.N_act])
    tachyonic_events = len(gw.tachyonic_events)
    
    print("\nResults:")
    print(f"- Active tokens: {gw.N_act}")
    print(f"- Crystallized tokens: {crystallized}")
    print(f"- Tachyonic events: {tachyonic_events}")
    
    # Report final energy (if valid)
    if gw.energy_history and np.isfinite(gw.energy_history[-1]):
        print(f"- Final energy: {gw.energy_history[-1]:.6f}")
    else:
        print("- Final energy: [Not available - numerical issues]")
    
    # Report numerical issues
    print(f"- Numerical issues: forces={gw.nan_counts['forces']}, positions={gw.nan_counts['positions']}, energy={gw.nan_counts['energy']}")
    
    # Track phi^n distributions
    print("\nPhi^n Layer Distribution:")
    for i, phi_n in enumerate(gw.phi_n_layers):
        total = gw.phi_n_counts[i, 0]
        crystallized = gw.phi_n_counts[i, 1]
        if total > 0:
            print(f"- φ^{i+1} = {phi_n:.3f}: {total} tokens ({crystallized} crystallized)")
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        try:
            gw.generate_all_visualizations()
            print(f"Visualizations saved to {OUTPUT_DIR}/")
        except ImportError as e:
            print(f"Could not generate visualizations: {e}")
    
    print("\nDemo completed successfully!")
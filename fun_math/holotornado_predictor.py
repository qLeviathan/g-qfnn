import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.special import jv, yv  # Bessel functions
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

class HolographicTornadoPredictor:
    def __init__(self, grid_size=128, domain_radius=50.0):
        self.grid_size = grid_size
        self.domain_radius = domain_radius
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Verify golden ratio property: φ² = 1 + φ
        assert abs(self.phi**2 - (1 + self.phi)) < 1e-10
        
        # Create radial grid
        self.r = torch.linspace(0.1, domain_radius, grid_size)
        self.theta = torch.linspace(0, 2*np.pi, grid_size)
        self.R, self.THETA = torch.meshgrid(self.r, self.theta, indexing='ij')
        
        # Physical constants (tornado-black hole correspondence)
        self.G = 1.0  # Gravitational analog
        self.c = 1.0  # Speed analog (max wind speed scaling)
        self.hbar = 1.0  # Quantum analog (vorticity quantum)
        
        # Holographic surface at r = domain_radius
        self.holographic_boundary = domain_radius
        
    def fokker_planck_radial(self, psi, t, D=1.0, V_potential=None):
        """
        Solve radial Fokker-Planck equation:
        ∂ψ/∂t = (1/r)∂/∂r[r D ∂ψ/∂r] - (1/r)∂/∂r[r ψ ∂V/∂r]
        
        With golden ratio annealing: D(r) = D₀ * φ^(-r/r₀)
        """
        # Golden ratio diffusion coefficient
        D_r = D * self.phi**(-self.r / self.domain_radius)
        
        # Calculate derivatives using sin/cos relationship
        # d(sin(θ))/dθ = cos(θ), d²(sin(θ))/dθ² = -sin(θ)
        
        # First radial derivative
        dpsi_dr = torch.gradient(psi, spacing=(self.r,))[0]
        
        # Second order term (diffusion)
        d2psi_dr2 = torch.gradient(dpsi_dr, spacing=(self.r,))[0]
        
        # Fokker-Planck evolution with cylindrical correction
        drift_term = 0
        if V_potential is not None:
            dV_dr = torch.gradient(V_potential(self.r), spacing=(self.r,))[0]
            drift_term = -torch.gradient(psi * dV_dr, spacing=(self.r,))[0] / self.r
        
        diffusion_term = D_r * (d2psi_dr2 + dpsi_dr / self.r)
        
        return diffusion_term + drift_term
    
    def schwarzschild_tornado_metric(self, r, M):
        """
        Tornado metric inspired by Schwarzschild black hole:
        ds² = -(1-2GM/rc²)c²dt² + (1-2GM/rc²)⁻¹dr² + r²dθ²
        
        Where M is the "mass" (intensity) of tornado vortex
        """
        r_s = 2 * self.G * M / self.c**2  # Schwarzschild radius analog
        
        # Metric components
        g_tt = -(1 - r_s / r)
        g_rr = 1 / (1 - r_s / r)
        g_theta_theta = r**2
        
        return g_tt, g_rr, g_theta_theta
    
    def holographic_entropy(self, r, M):
        """
        Bekenstein-Hawking entropy analog for tornado:
        S = A/(4l_p²) where A is area of "event horizon"
        
        Using φ-modified Planck length: l_p = √(ħG/c³) * φ
        """
        r_horizon = 2 * self.G * M / self.c**2
        A = 4 * np.pi * r_horizon**2
        l_planck_phi = np.sqrt(self.hbar * self.G / self.c**3) * self.phi
        
        return A / (4 * l_planck_phi**2)
    
    def feynman_path_integral(self, r_initial, r_final, n_paths=1000, n_steps=100):
        """
        Calculate quantum amplitude using Feynman path integrals
        with Heun's method for numerical integration
        
        Action S = ∫[½m(dr/dt)² - V(r)]dt
        Amplitude = Σ exp(iS/ħ) over all paths
        """
        amplitudes = []
        
        for _ in range(n_paths):
            # Generate random path using φ-weighted random walk
            path = self._generate_phi_path(r_initial, r_final, n_steps)
            
            # Calculate action using Heun's method
            action = self._calculate_action_heun(path)
            
            # Add quantum amplitude
            amplitude = np.exp(1j * action / self.hbar)
            amplitudes.append(amplitude)
        
        # Return probability amplitude
        return np.mean(amplitudes)
    
    def _generate_phi_path(self, r_start, r_end, n_steps):
        """Generate path weighted by golden ratio"""
        path = np.zeros(n_steps)
        path[0] = r_start
        path[-1] = r_end
        
        # Use φ-weighted interpolation
        for i in range(1, n_steps-1):
            t = i / (n_steps - 1)
            # Brownian bridge with φ-weighting
            mean = r_start * (1 - t) + r_end * t
            variance = t * (1 - t) * self.phi
            path[i] = mean + np.sqrt(variance) * np.random.randn()
        
        return path
    
    def _calculate_action_heun(self, path):
        """Calculate action using Heun's method (improved Euler)"""
        dt = 1.0 / len(path)
        action = 0
        
        for i in range(len(path) - 1):
            r_i = path[i]
            r_next = path[i + 1]
            
            # Velocity at current point
            v_i = (r_next - r_i) / dt
            
            # Kinetic energy
            T_i = 0.5 * v_i**2
            
            # Potential energy (tornado vortex potential)
            V_i = -self.G / r_i if r_i > 0.1 else -self.G / 0.1
            
            # Heun's method: use average of start and end point
            if i < len(path) - 2:
                r_next_next = path[i + 2]
                v_next = (r_next_next - r_next) / dt
                T_next = 0.5 * v_next**2
                V_next = -self.G / r_next if r_next > 0.1 else -self.G / 0.1
                
                # Average Lagrangian
                L_avg = 0.5 * ((T_i - V_i) + (T_next - V_next))
            else:
                L_avg = T_i - V_i
            
            action += L_avg * dt
        
        return action
    
    def predict_vortex_path_annealing(self, initial_state, n_iterations=1000, beta_schedule=None):
        """
        Predict tornado path using simulated annealing with φ-based cooling
        
        Temperature schedule: T(k) = T₀ / φ^(k/k₀)
        """
        if beta_schedule is None:
            # Golden ratio cooling schedule
            beta_schedule = lambda k: self.phi**(k / 100)
        
        current_state = initial_state.copy()
        current_energy = self._calculate_energy(current_state)
        
        best_state = current_state.copy()
        best_energy = current_energy
        
        trajectory = [current_state]
        
        for k in range(n_iterations):
            # Temperature using φ-annealing
            T = 1.0 / beta_schedule(k)
            
            # Propose new state using φ-weighted perturbation
            new_state = self._propose_state_phi(current_state, T)
            new_energy = self._calculate_energy(new_state)
            
            # Metropolis criterion
            delta_E = new_energy - current_energy
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                current_state = new_state
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_state = current_state.copy()
                    best_energy = current_energy
            
            trajectory.append(current_state.copy())
        
        return best_state, trajectory
    
    def _propose_state_phi(self, state, T):
        """Propose new state using φ-based perturbation"""
        # Perturbation size scales with φ^(-1/T)
        perturbation_scale = self.phi**(-1/T)
        
        new_state = state.copy()
        new_state['r'] += perturbation_scale * np.random.randn()
        new_state['theta'] += perturbation_scale * np.random.randn() / new_state['r']
        
        # Keep within bounds
        new_state['r'] = np.clip(new_state['r'], 0.1, self.domain_radius)
        new_state['theta'] = new_state['theta'] % (2 * np.pi)
        
        return new_state
    
    def _calculate_energy(self, state):
        """Calculate energy using tornado-black hole correspondence"""
        r = state['r']
        theta = state['theta']
        
        # Gravitational analog energy
        E_grav = -self.G / r
        
        # Rotational energy (using derivative relations: d(sin)/d = cos)
        v_theta = state.get('v_theta', 1.0)
        E_rot = 0.5 * v_theta**2 * r**2
        
        # Holographic contribution (information on boundary)
        E_holo = np.log(self.phi) * r / self.holographic_boundary
        
        return E_grav + E_rot + E_holo
    
    def rho_gravity_potential(self, r, rho_0=1.0, lambda_param=1.0):
        """
        ρ-gravity potential inspired by econophysics/game theory
        V(r) = -ρ₀ * log(1 + λ/r) * φ
        
        This creates attraction that weakens logarithmically
        """
        return -rho_0 * np.log(1 + lambda_param / r) * self.phi
    
    def solve_heun_differential(self, y0, t_span, tornado_params):
        """
        Solve tornado dynamics using Heun's method
        d²r/dt² = -∂V/∂r + L²/r³
        
        With V including ρ-gravity and holographic corrections
        """
        def dynamics(t, y):
            r, dr_dt, theta, dtheta_dt = y
            
            # Angular momentum
            L = r**2 * dtheta_dt
            
            # ρ-gravity force
            F_rho = -np.gradient(self.rho_gravity_potential(np.array([r])), r)[0]
            
            # Centrifugal force
            F_centrifugal = L**2 / r**3
            
            # Holographic correction (information flow from boundary)
            F_holo = -self.phi * (r - self.holographic_boundary) / self.holographic_boundary**2
            
            # Equations of motion
            d2r_dt2 = F_rho + F_centrifugal + F_holo
            d2theta_dt2 = -2 * dr_dt * dtheta_dt / r  # Conservation of angular momentum
            
            return [dr_dt, d2r_dt2, dtheta_dt, d2theta_dt2]
        
        # Solve using scipy's solve_ivp with RK45 (similar to Heun)
        solution = solve_ivp(dynamics, t_span, y0, method='RK45', dense_output=True)
        
        return solution
    
    def visualize_holographic_prediction(self, tornado_path, predicted_path):
        """Visualize tornado path with holographic boundary and predictions"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                       subplot_kw=dict(projection='polar'))
        
        # Left plot: Actual path with holographic encoding
        ax1.plot(tornado_path[:, 1], tornado_path[:, 0], 'r-', linewidth=2, 
                label='Actual Path')
        ax1.plot(predicted_path[:, 1], predicted_path[:, 0], 'b--', linewidth=2,
                label='Predicted Path')
        
        # Holographic boundary
        theta_boundary = np.linspace(0, 2*np.pi, 100)
        r_boundary = np.ones_like(theta_boundary) * self.holographic_boundary
        ax1.plot(theta_boundary, r_boundary, 'gold', linewidth=3, 
                label='Holographic Boundary')
        
        # Show information encoding on boundary
        n_info_points = 20
        for i in range(n_info_points):
            theta_i = i * 2 * np.pi / n_info_points
            # Information density follows φ-distribution
            info_density = self.phi**(i % 5)
            ax1.scatter(theta_i, self.holographic_boundary, 
                       s=info_density*50, c='orange', alpha=0.6)
        
        ax1.set_title('Tornado Path with Holographic Boundary')
        ax1.legend()
        ax1.set_ylim(0, self.domain_radius * 1.1)
        
        # Right plot: Phase space with φ-spirals
        ax2.set_theta_zero_location('N')
        
        # Plot φ-spiral attractors
        theta_spiral = np.linspace(0, 6*np.pi, 1000)
        r_spiral = self.domain_radius * np.exp(-theta_spiral / (2*np.pi*self.phi))
        ax2.plot(theta_spiral % (2*np.pi), r_spiral, 'gold', alpha=0.3, 
                label='φ-Spiral Attractor')
        
        # Plot energy contours
        R_grid, THETA_grid = np.meshgrid(np.linspace(0.1, self.domain_radius, 50),
                                        np.linspace(0, 2*np.pi, 50))
        
        # Calculate energy field
        E_field = np.zeros_like(R_grid)
        for i in range(R_grid.shape[0]):
            for j in range(R_grid.shape[1]):
                state = {'r': R_grid[i, j], 'theta': THETA_grid[i, j]}
                E_field[i, j] = self._calculate_energy(state)
        
        contours = ax2.contour(THETA_grid, R_grid, E_field, levels=15, alpha=0.5)
        ax2.clabel(contours, inline=True, fontsize=8)
        
        ax2.set_title('Phase Space with φ-Spiral Dynamics')
        ax2.set_ylim(0, self.domain_radius * 1.1)
        
        plt.tight_layout()
        return fig

# Example usage demonstrating the unified framework
def demonstrate_holographic_tornado_prediction():
    # Initialize predictor
    predictor = HolographicTornadoPredictor(grid_size=128, domain_radius=50.0)
    
    # Initial tornado state
    initial_state = {
        'r': 10.0,
        'theta': 0.0,
        'v_r': 0.5,
        'v_theta': 2.0,
        'M': 5.0  # Tornado "mass" (intensity)
    }
    
    # 1. Calculate holographic entropy
    S_holo = predictor.holographic_entropy(initial_state['r'], initial_state['M'])
    print(f"Holographic Entropy: {S_holo:.3f}")
    
    # 2. Solve Fokker-Planck for probability distribution
    print("\nSolving Fokker-Planck equation...")
    psi_0 = torch.exp(-0.5 * (predictor.r - initial_state['r'])**2 / 2.0)
    psi_0 = psi_0 / torch.trapz(psi_0, predictor.r)
    
    # Time evolution
    dt = 0.01
    psi = psi_0.clone()
    for t in range(100):
        dpsi_dt = predictor.fokker_planck_radial(psi, t*dt, D=1.0)
        psi = psi + dt * dpsi_dt
        psi = torch.abs(psi)  # Ensure positivity
        psi = psi / torch.trapz(psi, predictor.r)  # Normalize
    
    # 3. Calculate Feynman path integral amplitude
    print("\nCalculating Feynman path integral...")
    amplitude = predictor.feynman_path_integral(
        initial_state['r'], 
        predictor.domain_radius * 0.8, 
        n_paths=500
    )
    probability = np.abs(amplitude)**2
    print(f"Transition probability: {probability:.4f}")
    
    # 4. Predict path using φ-annealing
    print("\nPredicting tornado path with φ-annealing...")
    predicted_state, trajectory = predictor.predict_vortex_path_annealing(
        initial_state, 
        n_iterations=500
    )
    
    # 5. Solve dynamics with Heun's method
    print("\nSolving dynamics with Heun's method...")
    y0 = [initial_state['r'], initial_state['v_r'], 
          initial_state['theta'], initial_state['v_theta']]
    t_span = (0, 10)
    
    solution = predictor.solve_heun_differential(y0, t_span, initial_state)
    
    # Extract trajectory
    t_eval = np.linspace(0, 10, 100)
    trajectory_heun = solution.sol(t_eval).T
    
    # Convert to path format
    actual_path = np.column_stack([trajectory_heun[:, 0], trajectory_heun[:, 2]])
    predicted_path = np.array([[s['r'], s['theta']] for s in trajectory[:100]])
    
    # 6. Visualize results
    fig = predictor.visualize_holographic_prediction(actual_path, predicted_path)
    
    # Plot probability distribution evolution
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(predictor.r.numpy(), psi_0.numpy(), 'b-', label='Initial')
    ax.plot(predictor.r.numpy(), psi.numpy(), 'r-', label='Final')
    ax.fill_between(predictor.r.numpy(), psi.numpy(), alpha=0.3, color='red')
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Fokker-Planck Evolution of Tornado Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Show second-order derivative relationship
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Generate test function showing sin/cos relationship
    theta_test = np.linspace(0, 4*np.pi, 200)
    f = np.sin(theta_test)
    df = np.cos(theta_test)  # First derivative
    d2f = -np.sin(theta_test)  # Second derivative
    
    ax1.plot(theta_test, f, 'b-', label='f = sin(θ)')
    ax1.plot(theta_test, df, 'r-', label="f' = cos(θ)")
    ax1.plot(theta_test, d2f, 'g-', label="f'' = -sin(θ)")
    ax1.set_title('Derivative Relationships in Circular Motion')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Show φ² = 1 + φ visually
    phi_powers = [predictor.phi**i for i in range(-3, 4)]
    fibonacci = [0, 1]
    for i in range(2, 10):
        fibonacci.append(fibonacci[-1] + fibonacci[-2])
    
    ax2.plot(range(-3, 4), phi_powers, 'go-', markersize=8, label='φⁿ')
    ax2.axhline(y=1 + predictor.phi, color='r', linestyle='--', 
                label=f'1 + φ = {1 + predictor.phi:.3f}')
    ax2.axhline(y=predictor.phi**2, color='b', linestyle='--', 
                label=f'φ² = {predictor.phi**2:.3f}')
    ax2.set_xlabel('Power n')
    ax2.set_ylabel('Value')
    ax2.set_title('Golden Ratio Property: φ² = 1 + φ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return predictor

if __name__ == "__main__":
    predictor = demonstrate_holographic_tornado_prediction()
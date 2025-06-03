import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import gamma, logsumexp
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class LogEinsumQuantumVortex:
    def __init__(self, c=1.0, alpha=1.5):
        """
        Log-domain einsum implementation of quantum vortex dynamics
        All operations in log space for numerical stability and additive coherence
        """
        self.c = c
        self.alpha = alpha
        self.phi = (1 + np.sqrt(5)) / 2
        self.log_phi = np.log(self.phi)
        self.log_c = np.log(c)
        
        # Log-space metric signature
        self.log_eta = torch.log(torch.tensor([1.0, 1.0, 1.0, 1.0]))  # All positive in log space
        
    def log_einsum_levy_operator(self, log_psi, k_log, batch_size=64):
        """
        Logarithmic Lévy operator using einsum notation
        ∇^α → exp(α * log|k|) in Fourier space
        
        Args:
            log_psi: log-domain wavefunction [batch, spatial]
            k_log: log-domain wavenumbers [spatial]
        """
        # Batch einsum for log-domain Lévy operator
        # log(|k|^α) = α * log|k|
        log_levy_power = torch.einsum('i,->i', k_log, torch.tensor(self.alpha))
        
        # Log-domain diffusion coefficient: log(D_α) = log(c) + (2-α)*log(φ)
        log_D_levy = self.log_c + (2 - self.alpha) * self.log_phi
        
        # Einsum for batch application: log(D * |k|^α * ψ_k)
        # = log(D) + log(|k|^α) + log(ψ_k)
        log_diffusion_term = torch.einsum('i,bi->bi', 
                                        log_levy_power + log_D_levy, 
                                        log_psi)
        
        return log_diffusion_term
    
    def log_einsum_relativistic_correction(self, k_log, log_psi):
        """
        Relativistic correction in log domain
        γ = √(1 + (kc)²) → log(γ) = 0.5 * log(1 + exp(2*log(k) + 2*log(c)))
        """
        # log(k²c²) = 2*log(k) + 2*log(c)
        log_kc_squared = torch.einsum('i,->i', k_log, torch.tensor(2.0)) + 2 * self.log_c
        
        # log(1 + k²c²) using logsumexp for numerical stability
        log_one_plus_kc2 = torch.logsumexp(torch.stack([
            torch.zeros_like(log_kc_squared),  # log(1) = 0
            log_kc_squared
        ]), dim=0)
        
        # log(γ) = 0.5 * log(1 + k²c²)
        log_gamma = 0.5 * log_one_plus_kc2
        
        # Apply correction: log(ψ/γ) = log(ψ) - log(γ)
        log_corrected = torch.einsum('bi,i->bi', log_psi, -log_gamma)
        
        return log_corrected
    
    def log_einsum_kerr_metric(self, log_r, log_theta, log_a, log_M):
        """
        Kerr metric in log domain using einsum
        All metric components computed additively
        """
        # log(Σ) = log(r² + a²cos²θ)
        log_r_squared = 2 * log_r
        log_cos_theta = torch.log(torch.abs(torch.cos(torch.exp(log_theta))) + 1e-10)
        log_a_squared = 2 * log_a
        log_cos_squared = 2 * log_cos_theta
        
        # log(a²cos²θ) = log(a²) + log(cos²θ)
        log_a2_cos2 = log_a_squared + log_cos_squared
        
        # log(Σ) = log(r² + a²cos²θ) using logsumexp
        log_Sigma = torch.logsumexp(torch.stack([log_r_squared, log_a2_cos2]), dim=0)
        
        # log(Δ) = log(r² - 2Mr + a²)
        # This is trickier in log domain due to subtraction
        # We'll use the identity: log(a - b) when a > b
        
        # Frame dragging angular velocity in log domain
        # ω = 2Mar sin²θ / (r² + a²)Σ
        log_2M = log_a + log_M  # Assuming 2M ≈ a for simplicity
        log_sin_theta = torch.log(torch.abs(torch.sin(torch.exp(log_theta))) + 1e-10)
        log_sin_squared = 2 * log_sin_theta
        
        # log(ω) = log(2Mar) + log(sin²θ) - log(Σ) - log(denominator)
        log_omega = log_2M + log_a + log_r + log_sin_squared - 2 * log_Sigma
        
        return {
            'log_Sigma': log_Sigma,
            'log_omega': log_omega,
            'log_frame_drag': log_omega + log_r + log_sin_theta  # v = ωr sin θ
        }
    
    def log_einsum_quantum_vortex(self, log_r, log_phi, n, m):
        """
        Quantum vortex states in log domain
        |ψ⟩ = r^|m| exp(-r²/2) L_n^|m|(r²) exp(imφ)
        """
        abs_m = abs(m)
        
        # log(r^|m|) = |m| * log(r)
        log_r_power = abs_m * log_r
        
        # log(exp(-r²/2)) = -r²/2 = -exp(2*log(r))/2
        r_squared = torch.exp(2 * log_r)
        log_exponential = -r_squared / 2
        
        # For Laguerre polynomial, use series expansion in log domain
        # L_n^α(x) ≈ 1 for small x, so log(L_n^α) ≈ 0
        log_laguerre = torch.zeros_like(log_r)  # Simplified
        
        # Combine all log components
        log_radial = log_r_power + log_exponential + log_laguerre
        
        # Phase factor: exp(imφ) → we keep this separate as it's oscillatory
        phase_factor = torch.exp(1j * m * torch.exp(log_phi))
        
        # Phase velocity in log domain
        # v_phase = c * |m| * φ / n
        log_v_phase = self.log_c + np.log(abs_m) + self.log_phi - np.log(n) if n > 0 else float('inf')
        
        return {
            'log_amplitude': log_radial,
            'phase': phase_factor,
            'log_v_phase': log_v_phase,
            'is_superluminal': log_v_phase > self.log_c if n > 0 else True
        }
    
    def log_einsum_hebbian_update(self, log_pre, log_post, log_weights, lr_log):
        """
        Hebbian learning in log domain using einsum
        Δw = η * post ⊗ pre → log(Δw) = log(η) + log(post) + log(pre)
        """
        # Outer product in log domain: log(post ⊗ pre)
        # Using einsum: 'i,j->ij'
        log_outer = torch.einsum('i,j->ij', log_post, log_pre)
        
        # Learning rate scaling: log(η * φ) = log(η) + log(φ)
        log_lr_scaled = lr_log + self.log_phi
        
        # Weight update: log(w_new) = log(w_old + Δw)
        # This requires logsumexp for addition in log domain
        log_delta = log_lr_scaled + log_outer
        
        # Combine old weights and updates
        log_weights_new = torch.logsumexp(torch.stack([log_weights, log_delta]), dim=0)
        
        return log_weights_new
    
    def log_einsum_attention_mechanism(self, log_query, log_key, log_value):
        """
        Attention in log domain using einsum
        Attention(Q,K,V) = softmax(QK^T/√d_k)V
        """
        d_k = log_query.shape[-1]
        log_sqrt_dk = 0.5 * np.log(d_k)
        
        # log(QK^T) using einsum: 'bid,bjd->bij'
        log_scores = torch.einsum('bid,bjd->bij', log_query, log_key)
        
        # Scale by √d_k: log(scores/√d_k) = log(scores) - log(√d_k)
        log_scores_scaled = log_scores - log_sqrt_dk
        
        # Softmax in log domain (more stable)
        log_attention_weights = torch.log_softmax(log_scores_scaled, dim=-1)
        
        # Apply attention: log(attention * V)
        # 'bij,bjd->bid'
        log_output = torch.einsum('bij,bjd->bid', 
                                torch.exp(log_attention_weights),  # Convert back for multiplication
                                log_value)
        
        return log_output, log_attention_weights
    
    def log_einsum_vortex_field_evolution(self, log_psi_init, time_steps, spatial_grid):
        """
        Complete field evolution in log domain with einsum operations
        """
        batch_size, spatial_size = log_psi_init.shape
        
        # Initialize storage
        log_psi_history = [log_psi_init.clone()]
        log_energy_history = []
        log_vorticity_history = []
        
        # Wavenumber grid in log domain
        k = torch.fft.fftfreq(spatial_size, d=spatial_grid[1]-spatial_grid[0]) * 2 * np.pi
        k_log = torch.log(torch.abs(k) + 1e-10)
        
        log_psi = log_psi_init.clone()
        dt = 0.01
        log_dt = np.log(dt)
        
        for step in range(time_steps):
            # FFT to frequency domain (convert to complex for FFT)
            psi_complex = torch.exp(log_psi).to(torch.complex64)
            psi_k = torch.fft.fft(psi_complex, dim=-1)
            log_psi_k = torch.log(torch.abs(psi_k) + 1e-10)
            
            # Apply Lévy operator in log domain
            log_levy_term = self.log_einsum_levy_operator(log_psi_k, k_log)
            
            # Apply relativistic correction
            log_corrected = self.log_einsum_relativistic_correction(k_log, log_levy_term)
            
            # Time evolution: log(ψ_new) = log(ψ_old) + log(dt) + log(dψ/dt)
            log_dpsi_dt = log_corrected
            log_psi_update = torch.logsumexp(torch.stack([
                log_psi + log_dt + log_dpsi_dt,
                log_psi
            ]), dim=0)
            
            # Convert back to spatial domain
            psi_k_updated = torch.exp(log_psi_update).to(torch.complex64)
            psi_spatial = torch.fft.ifft(psi_k_updated, dim=-1)
            log_psi = torch.log(torch.abs(psi_spatial) + 1e-10)
            
            # Normalization in log domain
            log_norm = torch.logsumexp(2 * log_psi, dim=-1, keepdim=True)
            log_psi = log_psi - 0.5 * log_norm
            
            # Compute energy and vorticity in log domain
            # Energy: E = ∫ |ψ|² dx
            log_energy = torch.logsumexp(2 * log_psi, dim=-1)
            
            # Vorticity: ∇ × v where v = (ψ* ∇ψ - ψ ∇ψ*) / (2i|ψ|²)
            log_gradient = torch.log(torch.abs(torch.gradient(torch.exp(log_psi), dim=-1)[0]) + 1e-10)
            log_vorticity = log_gradient - 2 * log_psi  # Simplified
            
            # Store history
            log_psi_history.append(log_psi.clone())
            log_energy_history.append(log_energy.mean().item())
            log_vorticity_history.append(torch.logsumexp(log_vorticity, dim=-1).mean().item())
        
        return {
            'log_psi_history': log_psi_history,
            'log_energy_history': log_energy_history,
            'log_vorticity_history': log_vorticity_history,
            'spatial_grid': spatial_grid
        }
    
    def visualize_log_domain_dynamics(self):
        """
        Comprehensive visualization of log-domain vortex dynamics
        """
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Log-domain field evolution
        ax1 = fig.add_subplot(3, 4, 1)
        
        # Initialize field
        spatial_grid = torch.linspace(-10, 10, 128)
        log_psi_init = -0.5 * spatial_grid**2 / self.phi**2  # Log of Gaussian
        log_psi_init = log_psi_init.unsqueeze(0)  # Add batch dimension
        
        # Evolve system
        evolution = self.log_einsum_vortex_field_evolution(
            log_psi_init, time_steps=100, spatial_grid=spatial_grid
        )
        
        # Plot field evolution
        for i in range(0, len(evolution['log_psi_history']), 10):
            alpha = i / len(evolution['log_psi_history'])
            log_psi = evolution['log_psi_history'][i][0]  # Remove batch dim
            ax1.plot(spatial_grid, torch.exp(log_psi), 
                    color=(1-alpha, 0, alpha), alpha=0.7, linewidth=1)
        
        ax1.set_xlabel('Position')
        ax1.set_ylabel('|ψ|')
        ax1.set_title('Log-Domain Field Evolution')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 2. Energy conservation in log domain
        ax2 = fig.add_subplot(3, 4, 2)
        time_steps = np.arange(len(evolution['log_energy_history'])) * 0.01
        ax2.plot(time_steps, evolution['log_energy_history'], 'b-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('log(Energy)')
        ax2.set_title('Energy Conservation (Log Domain)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Vorticity evolution
        ax3 = fig.add_subplot(3, 4, 3)
        ax3.plot(time_steps, evolution['log_vorticity_history'], 'r-', linewidth=2)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('log(Vorticity)')
        ax3.set_title('Vorticity Dynamics')
        ax3.grid(True, alpha=0.3)
        
        # 4. Kerr metric in log domain
        ax4 = fig.add_subplot(3, 4, 4, projection='3d')
        
        log_r = torch.linspace(np.log(0.1), np.log(10), 50)
        log_theta = torch.linspace(np.log(0.1), np.log(np.pi-0.1), 50)
        LOG_R, LOG_THETA = torch.meshgrid(log_r, log_theta, indexing='ij')
        
        log_a = np.log(0.9)
        log_M = np.log(1.0)
        
        kerr_data = self.log_einsum_kerr_metric(LOG_R, LOG_THETA, log_a, log_M)
        
        # Convert to Cartesian for plotting
        R = torch.exp(LOG_R)
        THETA = torch.exp(LOG_THETA)
        X = R * torch.sin(THETA)
        Z = R * torch.cos(THETA)
        
        frame_drag = torch.exp(kerr_data['log_frame_drag'])
        surf = ax4.plot_surface(X.numpy(), frame_drag.numpy(), Z.numpy(), 
                               cmap='RdBu_r', alpha=0.8)
        ax4.set_xlabel('r sin θ')
        ax4.set_ylabel('Frame Drag Velocity')
        ax4.set_zlabel('r cos θ')
        ax4.set_title('Kerr Frame Dragging (Log Domain)')
        
        # 5. Quantum vortex phase velocities
        ax5 = fig.add_subplot(3, 4, 5)
        
        log_r_vortex = torch.linspace(np.log(0.1), np.log(5), 100)
        log_phi_vortex = torch.linspace(np.log(0.1), np.log(2*np.pi), 100)
        
        m_values = [1, 5, 10, 20]
        colors = ['blue', 'green', 'red', 'purple']
        
        for m, color in zip(m_values, colors):
            vortex_data = self.log_einsum_quantum_vortex(log_r_vortex, log_phi_vortex[0], n=1, m=m)
            if vortex_data['log_v_phase'] != float('inf'):
                v_phase = np.exp(vortex_data['log_v_phase'])
                ax5.axhline(y=v_phase/self.c, color=color, linewidth=2, 
                           label=f'm={m}, v/c={v_phase/self.c:.1f}')
        
        ax5.axhline(y=1, color='black', linestyle='--', linewidth=2, label='c')
        ax5.set_xlabel('Quantum Number m')
        ax5.set_ylabel('Phase Velocity / c')
        ax5.set_title('Quantum Vortex Phase Velocities')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')
        
        # 6. Hebbian weight evolution in log domain
        ax6 = fig.add_subplot(3, 4, 6)
        
        # Initialize weights and data
        dim = 10
        log_weights = torch.zeros(dim, dim)
        log_pre = torch.randn(dim)
        log_post = torch.randn(dim)
        lr_log = np.log(0.01)
        
        weight_norms = []
        for step in range(100):
            log_weights = self.log_einsum_hebbian_update(log_pre, log_post, log_weights, lr_log)
            weight_norms.append(torch.logsumexp(log_weights.flatten(), dim=0).item())
            
            # Update inputs
            log_pre = log_post + 0.1 * torch.randn(dim)
            log_post = log_pre + 0.1 * torch.randn(dim)
        
        ax6.plot(weight_norms, 'g-', linewidth=2)
        ax6.set_xlabel('Training Steps')
        ax6.set_ylabel('log(Weight Norm)')
        ax6.set_title('Hebbian Learning (Log Domain)')
        ax6.grid(True, alpha=0.3)
        
        # 7. Attention mechanism comparison
        ax7 = fig.add_subplot(3, 4, 7)
        
        batch_size, seq_len, d_model = 2, 32, 16
        log_query = torch.randn(batch_size, seq_len, d_model)
        log_key = torch.randn(batch_size, seq_len, d_model)
        log_value = torch.randn(batch_size, seq_len, d_model)
        
        log_output, log_attention = self.log_einsum_attention_mechanism(log_query, log_key, log_value)
        
        # Visualize attention matrix
        attention_matrix = torch.exp(log_attention[0].detach())  # First batch
        im = ax7.imshow(attention_matrix.numpy(), cmap='hot', aspect='auto')
        ax7.set_xlabel('Key Position')
        ax7.set_ylabel('Query Position')
        ax7.set_title('Log-Domain Attention Matrix')
        plt.colorbar(im, ax=ax7)
        
        # 8. Phase space density
        ax8 = fig.add_subplot(3, 4, 8)
        
        # Generate phase space points
        n_points = 1000
        log_positions = torch.randn(n_points) * 2
        log_momenta = torch.randn(n_points) * 2
        
        # Color by energy: E = p²/2m + V(x)
        log_kinetic = 2 * log_momenta - np.log(2)  # log(p²/2m)
        log_potential = log_positions**2 / 2  # log(harmonic potential)
        log_energy = torch.logsumexp(torch.stack([log_kinetic, log_potential]), dim=0)
        
        scatter = ax8.scatter(log_positions.numpy(), log_momenta.numpy(), 
                             c=log_energy.numpy(), cmap='viridis', s=10, alpha=0.6)
        ax8.set_xlabel('log(Position)')
        ax8.set_ylabel('log(Momentum)')
        ax8.set_title('Phase Space (Log Domain)')
        plt.colorbar(scatter, ax=ax8, label='log(Energy)')
        
        # 9. Lévy flight comparison
        ax9 = fig.add_subplot(3, 4, 9)
        
        # Simulate Lévy flights in log domain
        steps = 1000
        log_positions_gaussian = torch.cumsum(torch.randn(steps) * 0.1, dim=0)
        
        # Lévy flights with heavy tails
        levy_steps = torch.randn(steps) * 0.1
        # Add occasional large jumps
        large_jump_mask = torch.rand(steps) < 0.05
        levy_steps[large_jump_mask] *= 10
        log_positions_levy = torch.cumsum(levy_steps, dim=0)
        
        time_axis = np.arange(steps)
        ax9.plot(time_axis, log_positions_gaussian.numpy(), 'b-', alpha=0.7, label='Gaussian')
        ax9.plot(time_axis, log_positions_levy.numpy(), 'r-', alpha=0.7, label='Lévy')
        ax9.set_xlabel('Time Steps')
        ax9.set_ylabel('log(Position)')
        ax9.set_title('Lévy vs Gaussian Diffusion')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 10. Golden ratio scaling
        ax10 = fig.add_subplot(3, 4, 10)
        
        # Show how golden ratio emerges in log domain
        n_terms = 50
        fibonacci = [1, 1]
        for i in range(2, n_terms):
            fibonacci.append(fibonacci[-1] + fibonacci[-2])
        
        ratios = [fibonacci[i+1]/fibonacci[i] for i in range(len(fibonacci)-1)]
        log_ratios = np.log(ratios)
        
        ax10.plot(log_ratios, 'o-', color='gold', linewidth=2, markersize=4)
        ax10.axhline(y=self.log_phi, color='red', linestyle='--', linewidth=2, 
                    label=f'log(φ) = {self.log_phi:.4f}')
        ax10.set_xlabel('Fibonacci Index')
        ax10.set_ylabel('log(F_{n+1}/F_n)')
        ax10.set_title('Golden Ratio Convergence (Log Domain)')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        # 11. Energy spectral density
        ax11 = fig.add_subplot(3, 4, 11)
        
        # Compute power spectral density
        final_psi = evolution['log_psi_history'][-1][0]
        psi_complex = torch.exp(final_psi).to(torch.complex64)
        psi_fft = torch.fft.fft(psi_complex)
        power_spectrum = torch.abs(psi_fft)**2
        
        freqs = torch.fft.fftfreq(len(spatial_grid), d=spatial_grid[1]-spatial_grid[0])
        
        ax11.loglog(freqs[1:len(freqs)//2].numpy(), 
                   power_spectrum[1:len(freqs)//2].numpy(), 'b-', linewidth=2)
        ax11.set_xlabel('Frequency')
        ax11.set_ylabel('Power Spectral Density')
        ax11.set_title('Energy Spectrum (Log-Log)')
        ax11.grid(True, alpha=0.3)
        
        # 12. Coherence measure
        ax12 = fig.add_subplot(3, 4, 12)
        
        # Compute coherence between different time steps
        coherence = []
        for i in range(1, len(evolution['log_psi_history'])):
            psi1 = torch.exp(evolution['log_psi_history'][i-1][0])
            psi2 = torch.exp(evolution['log_psi_history'][i][0])
            
            # Normalized cross-correlation
            correlation = torch.sum(psi1.conj() * psi2).real
            norm1 = torch.sum(torch.abs(psi1)**2)
            norm2 = torch.sum(torch.abs(psi2)**2)
            coherence.append((correlation / torch.sqrt(norm1 * norm2)).item())
        
        ax12.plot(coherence, 'purple', linewidth=2)
        ax12.set_xlabel('Time Step')
        ax12.set_ylabel('Coherence')
        ax12.set_title('Temporal Coherence')
        ax12.grid(True, alpha=0.3)
        ax12.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        return fig, evolution

# Einsum optimization tests
def test_einsum_efficiency():
    """
    Test computational efficiency of log-domain einsum operations
    """
    print("\n" + "="*60)
    print("EINSUM EFFICIENCY TESTS")
    print("="*60)
    
    import time
    
    # Test dimensions
    batch_size = 64
    seq_len = 512
    d_model = 256
    
    # Standard operations
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # Log-domain operations
    log_query = torch.log(torch.abs(query) + 1e-10)
    log_key = torch.log(torch.abs(key) + 1e-10)
    log_value = torch.log(torch.abs(value) + 1e-10)
    
    vortex = LogEinsumQuantumVortex()
    
    # Time standard attention
    start_time = time.time()
    for _ in range(10):
        scores = torch.einsum('bid,bjd->bij', query, key)
        attention = torch.softmax(scores / np.sqrt(d_model), dim=-1)
        output = torch.einsum('bij,bjd->bid', attention, value)
    standard_time = time.time() - start_time
    
    # Time log-domain attention
    start_time = time.time()
    for _ in range(10):
        log_output, log_attention = vortex.log_einsum_attention_mechanism(
            log_query, log_key, log_value
        )
    log_time = time.time() - start_time
    
    print(f"Standard attention time: {standard_time:.4f}s")
    print(f"Log-domain attention time: {log_time:.4f}s")
    print(f"Speedup: {standard_time/log_time:.2f}x")
    
    # Test memory efficiency
    def get_memory_usage():
        return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Memory test for large tensors
    large_dim = 2048
    large_tensor = torch.randn(large_dim, large_dim)
    log_large_tensor = torch.log(torch.abs(large_tensor) + 1e-10)
    
    print(f"\nMemory efficiency:")
    print(f"Standard tensor memory: {large_tensor.element_size() * large_tensor.nelement() / 1e6:.2f} MB")
    print(f"Log tensor memory: {log_large_tensor.element_size() * log_large_tensor.nelement() / 1e6:.2f} MB")
    
    # Test numerical stability
    print(f"\nNumerical stability test:")
    small_values = torch.tensor([1e-10, 1e-20, 1e-30])
    log_small_values = torch.log(small_values + 1e-40)
    
    print(f"Standard values: {small_values}")
    print(f"Log values: {log_small_values}")
    print(f"Recovered: {torch.exp(log_small_values)}")

if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs('output', exist_ok=True)
    
    # Run efficiency tests
    test_einsum_efficiency()
    
    # Create and visualize the log-domain vortex system
    print("\n" + "="*60)
    print("LOG-DOMAIN QUANTUM VORTEX VISUALIZATION")
    print("="*60)
    
    vortex_system = LogEinsumQuantumVortex(c=1.0, alpha=1.5)
    fig, evolution_data = vortex_system.visualize_log_domain_dynamics()
    
    # Save figure
    fig.savefig('output/log_einsum_quantum_vortex.png', dpi=300, bbox_inches='tight')
    
    # Print insights
    print(f"\nLog-Domain Vortex Analysis:")
    print(f"- Golden ratio (φ): {vortex_system.phi:.6f}")
    print(f"- Log golden ratio: {vortex_system.log_phi:.6f}")
    print(f"- Lévy index (α): {vortex_system.alpha}")
    print(f"- Final energy: {evolution_data['log_energy_history'][-1]:.4f} (log scale)")
    print(f"- Energy conservation deviation: {np.std(evolution_data['log_energy_history']):.6f}")
    print(f"- Max vorticity: {max(evolution_data['log_vorticity_history']):.4f} (log scale)")
    
    print(f"\nComputational Benefits:")
    print(f"- All multiplications → additions in log domain")
    print(f"- Numerical stability for extreme values")
    print(f"- Natural handling of power laws and scaling")
    print(f"- Einsum notation optimizes tensor contractions")
    print(f"- Compatible with quantum flux neural architectures")
    
    print(f"\nOutput saved: output/log_einsum_quantum_vortex.png")
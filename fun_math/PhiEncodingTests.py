import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, CheckButtons
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

# Golden ratio constants
PHI = 1.618033988749895
PHI_INV = 1/PHI  # 0.382...
PHI_CONJ = PHI - 1  # 0.618...

class QuantumFieldVisualizer:
    """
    Interactive visualizer for quantum field dynamics with phi-based encoding
    """
    
    def __init__(self, n_particles=50):
        self.n = n_particles
        self.t = 0.0
        self.dt = 0.01
        
        # Initialize with phi-based encoding
        self.initialize_field()
        
        # Track history for trails
        self.history = {'r': [], 'theta': [], 'z': [], 'coherence': [], 'energy': []}
        self.max_history = 100
        
        # Resonance pairs tracking
        self.resonance_pairs = []
        self.triangles = []
        
    def initialize_field(self):
        """Initialize field with phi-based harmonic encoding"""
        # Use 0.382 and 0.618 for natural phi-based coherence
        base_radii = np.random.choice([PHI_INV, PHI_CONJ], self.n)
        
        # Add small perturbations to break symmetry
        self.ln_r = np.log(base_radii + 0.1 * np.random.randn(self.n) * 0.01)
        
        # Distribute angles using golden angle
        golden_angle = 2 * np.pi / PHI**2
        self.theta = np.arange(self.n) * golden_angle + 0.1 * np.random.randn(self.n)
        self.theta = self.theta % (2 * np.pi)
        
        # Initialize z near ground state with small excitations
        self.z = 0.1 * np.random.rand(self.n)
        
        # Hebbian weights (log-domain)
        self.ln_W = np.log(0.01 * np.ones((self.n, self.n)))
        
        # Coherence matrix
        self.rho = np.zeros((self.n, self.n), dtype=complex)
        
    def calculate_resonance(self, i, j):
        """Calculate resonance between particles (without phi/2 offset)"""
        r_i = np.exp(self.ln_r[i])
        r_j = np.exp(self.ln_r[j])
        
        # Original formula without the artificial offset
        R_ij = np.abs(r_i * np.cos(self.theta[i]) - r_j * np.sin(self.theta[j]))
        
        return R_ij
    
    def calculate_coherence_matrix(self):
        """Calculate quantum coherence matrix"""
        for i in range(self.n):
            for j in range(self.n):
                # Wave function overlap in log domain
                psi_i = np.exp(self.ln_r[i]) * np.exp(1j * self.theta[i])
                psi_j = np.exp(self.ln_r[j]) * np.exp(1j * self.theta[j])
                
                self.rho[i, j] = psi_i * np.conj(psi_j) * np.exp(-self.z[i]**2 - self.z[j]**2)
        
        # L1 coherence
        self.C_L1 = np.sum(np.abs(self.rho[np.triu_indices(self.n, k=1)]))
        self.C_L1 /= (self.n * (self.n - 1) / 2)  # Normalize
        
        return self.C_L1
    
    def find_resonant_pairs(self, threshold=0.1):
        """Find pairs of particles in resonance"""
        self.resonance_pairs = []
        
        for i in range(self.n):
            for j in range(i+1, self.n):
                R_ij = self.calculate_resonance(i, j)
                
                if R_ij < threshold:
                    # Calculate coupling strength
                    S_ij = np.exp(-R_ij**2 / (2 * 0.1 * np.sin(2*np.pi*self.t/PHI**2)**2 + 0.01))
                    self.resonance_pairs.append((i, j, S_ij))
    
    def find_triangles(self):
        """Find strongly coupled triangles (3-body interactions)"""
        self.triangles = []
        threshold = 0.5
        
        for i in range(self.n):
            for j in range(i+1, self.n):
                for k in range(j+1, self.n):
                    # Check if all three pairs are coupled
                    R_ij = self.calculate_resonance(i, j)
                    R_jk = self.calculate_resonance(j, k)
                    R_ki = self.calculate_resonance(k, i)
                    
                    S_ij = np.exp(-R_ij**2 / 0.1)
                    S_jk = np.exp(-R_jk**2 / 0.1)
                    S_ki = np.exp(-R_ki**2 / 0.1)
                    
                    S_ijk = S_ij * S_jk * S_ki
                    
                    if S_ijk > threshold:
                        # Calculate centroid
                        r_i, r_j, r_k = np.exp(self.ln_r[[i,j,k]])
                        x_c = (r_i*np.cos(self.theta[i]) + r_j*np.cos(self.theta[j]) + r_k*np.cos(self.theta[k])) / 3
                        y_c = (r_i*np.sin(self.theta[i]) + r_j*np.sin(self.theta[j]) + r_k*np.sin(self.theta[k])) / 3
                        
                        self.triangles.append({
                            'indices': (i, j, k),
                            'strength': S_ijk,
                            'centroid': (x_c, y_c)
                        })
    
    def evolve_step(self):
        """Single evolution step with adaptive time step"""
        # Calculate coherence
        C_L1 = self.calculate_coherence_matrix()
        
        # Adaptive time step - slows down as system stabilizes (high coherence)
        # This addresses your question about quantum entropy stabilization
        adaptive_dt = self.dt * (1 - 0.5 * C_L1)
        
        # Calculate drift and diffusion parameters
        omega = 2 * np.pi / PHI**2
        sin_wt = np.sin(omega * self.t)
        
        # Stochastic evolution for ln(r)
        for i in range(self.n):
            # Drift term with phi-based decay
            mu_ln_r = -self.ln_r[i] / (PHI * 1.0)  # T_0 = 1.0
            
            # Diffusion modulated by coherence
            sigma = 0.1 * (1 - C_L1) * np.abs(sin_wt)
            
            # Levy flight with stability parameter
            alpha = 2 - 0.1 * C_L1  # Approaches Gaussian as coherence increases
            
            # Simple approximation of Levy flight
            if np.random.rand() < 0.01:  # Rare large jumps
                jump = np.random.randn() * sigma * 5
            else:
                jump = np.random.randn() * sigma
            
            self.ln_r[i] += (mu_ln_r * adaptive_dt + jump * np.sqrt(adaptive_dt))
        
        # Evolution for theta with coupling
        for i in range(self.n):
            # Local field from neighbors
            coupling = 0
            for j in range(self.n):
                if i != j:
                    R_ij = self.calculate_resonance(i, j)
                    if R_ij < 0.5:  # Only nearby particles couple
                        coupling += np.sin(self.theta[j] - self.theta[i] + omega * self.t)
            
            coupling /= self.n
            
            # Phase evolution with neighbor influence
            self.theta[i] += omega * (1 + 0.1 * coupling) * adaptive_dt
            self.theta[i] = self.theta[i] % (2 * np.pi)
        
        # Evolution for z (height/excitation)
        for i in range(self.n):
            # Decay toward ground state
            self.z[i] -= self.z[i] * np.abs(sin_wt) * adaptive_dt / 1.0
            
            # Diffusion maintaining bounds
            noise = np.random.randn() * 0.1 * (1 - C_L1) * (1 - self.z[i]**2) * np.abs(sin_wt)
            self.z[i] += noise * np.sqrt(adaptive_dt)
            self.z[i] = np.clip(self.z[i], 0, 1)
        
        # Hebbian weight updates
        self.update_hebbian_weights(C_L1)
        
        # Find resonances and triangles
        self.find_resonant_pairs()
        self.find_triangles()
        
        # Update history
        self.update_history()
        
        # Increment time
        self.t += adaptive_dt
    
    def update_hebbian_weights(self, C_L1):
        """Update weights using Hebbian learning rule"""
        omega = 2 * np.pi / PHI**2
        
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    # Phase difference with time modulation
                    phase_diff = self.theta[i] - self.theta[j] + omega * self.t
                    
                    # Resonance strength
                    R_ij = self.calculate_resonance(i, j)
                    S_ij = np.exp(-R_ij**2 / 0.1)
                    
                    # Coherence between states
                    rho_ij = np.abs(self.rho[i, j])
                    
                    # Learning rate modulated by system state
                    eta = 0.1 / PHI * (1 - C_L1)
                    
                    # Hebbian update in log domain
                    delta_ln_W = eta * rho_ij * np.sin(phase_diff) * S_ij
                    self.ln_W[i, j] += delta_ln_W
    
    def calculate_total_energy(self):
        """Calculate total system energy"""
        r = np.exp(self.ln_r)
        
        # Kinetic energy (from phase velocity)
        omega = 2 * np.pi / PHI**2
        E_kinetic = 0.5 * np.sum(r**2 * omega**2)
        
        # Potential energy (from radial displacement and interactions)
        E_potential = 0.5 * np.sum((r - PHI_CONJ)**2)  # Centered at 0.618
        
        # Interaction energy
        E_interaction = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                R_ij = self.calculate_resonance(i, j)
                E_interaction += R_ij**2
        
        return E_kinetic + E_potential + E_interaction
    
    def update_history(self):
        """Update particle history for trails"""
        r = np.exp(self.ln_r)
        
        self.history['r'].append(r.copy())
        self.history['theta'].append(self.theta.copy())
        self.history['z'].append(self.z.copy())
        self.history['coherence'].append(self.C_L1)
        self.history['energy'].append(self.calculate_total_energy())
        
        # Keep only recent history
        if len(self.history['r']) > self.max_history:
            for key in self.history:
                self.history[key] = self.history[key][-self.max_history:]


def create_interactive_visualization():
    """Create interactive quantum field visualization"""
    
    # Initialize the quantum field
    qf = QuantumFieldVisualizer(n_particles=30)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Main phase space plot
    ax1 = plt.subplot(2, 3, 1, projection='polar')
    ax1.set_title('Phase Space Dynamics')
    
    # 3D visualization
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    ax2.set_title('3D Quantum Field')
    
    # Coherence matrix
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title('Coherence Matrix |ρᵢⱼ|')
    
    # Energy and coherence evolution
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title('System Evolution')
    
    # Resonance network
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title('Resonance Network')
    
    # Control panel
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    ax6.text(0.1, 0.9, 'Controls', fontsize=14, weight='bold')
    
    # Animation state
    animation_state = {'running': True, 'speed': 1.0, 'show_trails': True, 
                      'show_resonances': True, 'show_triangles': True}
    
    def update_frame(frame):
        """Update animation frame"""
        if not animation_state['running']:
            return
        
        # Evolve the system
        for _ in range(int(animation_state['speed'])):
            qf.evolve_step()
        
        # Clear axes
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        
        # Get current state
        r = np.exp(qf.ln_r)
        
        # 1. Phase space plot
        ax1.set_title(f'Phase Space (t={qf.t:.2f}, C_L1={qf.C_L1:.3f})')
        
        # Plot particles with color based on height (z)
        scatter = ax1.scatter(qf.theta, r, c=qf.z, cmap='viridis', 
                            s=100, alpha=0.8, edgecolors='black')
        
        # Plot trails if enabled
        if animation_state['show_trails'] and len(qf.history['r']) > 1:
            for i in range(qf.n):
                trail_r = [h[i] for h in qf.history['r'][-20:]]
                trail_theta = [h[i] for h in qf.history['theta'][-20:]]
                alphas = np.linspace(0.1, 0.5, len(trail_r))
                
                for j in range(len(trail_r)-1):
                    ax1.plot([trail_theta[j], trail_theta[j+1]], 
                           [trail_r[j], trail_r[j+1]], 
                           'gray', alpha=alphas[j], linewidth=0.5)
        
        # Mark phi-based radii
        ax1.axhline(y=PHI_INV, color='gold', linestyle='--', alpha=0.5, label=f'1/φ = {PHI_INV:.3f}')
        ax1.axhline(y=PHI_CONJ, color='orange', linestyle='--', alpha=0.5, label=f'φ-1 = {PHI_CONJ:.3f}')
        
        ax1.set_ylim(0, 1.2)
        ax1.legend(loc='upper right', fontsize=8)
        
        # 2. 3D visualization
        x = r * np.cos(qf.theta)
        y = r * np.sin(qf.theta)
        
        ax2.scatter(x, y, qf.z, c=qf.z, cmap='plasma', s=50, alpha=0.8)
        
        # Plot resonance connections in 3D
        if animation_state['show_resonances']:
            for i, j, strength in qf.resonance_pairs:
                if strength > 0.3:
                    ax2.plot([x[i], x[j]], [y[i], y[j]], [qf.z[i], qf.z[j]], 
                           'cyan', alpha=strength*0.5, linewidth=strength*2)
        
        # Plot triangles
        if animation_state['show_triangles']:
            for tri in qf.triangles[:5]:  # Show top 5 strongest
                i, j, k = tri['indices']
                # Draw triangle
                triangle_x = [x[i], x[j], x[k], x[i]]
                triangle_y = [y[i], y[j], y[k], y[i]]
                triangle_z = [qf.z[i], qf.z[j], qf.z[k], qf.z[i]]
                ax2.plot(triangle_x, triangle_y, triangle_z, 
                       'yellow', alpha=tri['strength'], linewidth=2)
        
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_zlim(0, 1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z (Excitation)')
        
        # 3. Coherence matrix
        coherence_matrix = np.abs(qf.rho)
        im = ax3.imshow(coherence_matrix, cmap='hot', interpolation='nearest')
        ax3.set_xlabel('Particle i')
        ax3.set_ylabel('Particle j')
        
        # 4. Evolution plots
        if len(qf.history['coherence']) > 1:
            steps = range(len(qf.history['coherence']))
            
            # Coherence evolution
            ax4_twin = ax4.twinx()
            ax4.plot(steps, qf.history['coherence'], 'b-', label='Coherence')
            ax4.set_ylabel('Coherence', color='b')
            ax4.tick_params(axis='y', labelcolor='b')
            
            # Energy evolution
            ax4_twin.plot(steps, qf.history['energy'], 'r-', label='Energy')
            ax4_twin.set_ylabel('Energy', color='r')
            ax4_twin.tick_params(axis='y', labelcolor='r')
            
            ax4.set_xlabel('Time Steps')
            ax4.grid(True, alpha=0.3)
        
        # 5. Resonance network
        ax5.set_xlim(-1.5, 1.5)
        ax5.set_ylim(-1.5, 1.5)
        ax5.set_aspect('equal')
        
        # Plot particles
        ax5.scatter(x, y, c=qf.z, cmap='viridis', s=100, alpha=0.8, edgecolors='black')
        
        # Plot resonance connections
        if animation_state['show_resonances']:
            for i, j, strength in qf.resonance_pairs:
                if strength > 0.2:
                    ax5.plot([x[i], x[j]], [y[i], y[j]], 
                           color='blue', alpha=strength, linewidth=strength*3)
        
        # Highlight triangular structures
        if animation_state['show_triangles']:
            for tri in qf.triangles[:3]:
                i, j, k = tri['indices']
                triangle = patches.Polygon([(x[i], y[i]), (x[j], y[j]), (x[k], y[k])],
                                         alpha=0.3*tri['strength'], 
                                         facecolor='yellow',
                                         edgecolor='orange',
                                         linewidth=2)
                ax5.add_patch(triangle)
                
                # Mark centroid
                ax5.plot(tri['centroid'][0], tri['centroid'][1], 
                       'ro', markersize=5)
        
        plt.tight_layout()
    
    # Create animation
    anim = FuncAnimation(fig, update_frame, interval=50, blit=False)
    
    # Add control buttons
    def toggle_animation(event):
        animation_state['running'] = not animation_state['running']
    
    def toggle_trails(event):
        animation_state['show_trails'] = not animation_state['show_trails']
    
    def toggle_resonances(event):
        animation_state['show_resonances'] = not animation_state['show_resonances']
    
    def toggle_triangles(event):
        animation_state['show_triangles'] = not animation_state['show_triangles']
    
    def speed_up(event):
        animation_state['speed'] = min(5, animation_state['speed'] + 0.5)
    
    def speed_down(event):
        animation_state['speed'] = max(0.5, animation_state['speed'] - 0.5)
    
    # Button positions
    button_play = Button(plt.axes([0.7, 0.45, 0.1, 0.03]), 'Play/Pause')
    button_play.on_clicked(toggle_animation)
    
    button_trails = Button(plt.axes([0.7, 0.40, 0.1, 0.03]), 'Trails')
    button_trails.on_clicked(toggle_trails)
    
    button_resonances = Button(plt.axes([0.7, 0.35, 0.1, 0.03]), 'Resonances')
    button_resonances.on_clicked(toggle_resonances)
    
    button_triangles = Button(plt.axes([0.7, 0.30, 0.1, 0.03]), 'Triangles')
    button_triangles.on_clicked(toggle_triangles)
    
    button_faster = Button(plt.axes([0.82, 0.25, 0.05, 0.03]), '+')
    button_faster.on_clicked(speed_up)
    
    button_slower = Button(plt.axes([0.70, 0.25, 0.05, 0.03]), '-')
    button_slower.on_clicked(speed_down)
    
    plt.show()
    
    return qf, anim


# Additional analysis functions for Jupyter exploration
def analyze_phi_encoding():
    """Analyze the effect of phi-based initialization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Compare different initialization strategies
    strategies = {
        'Phi-based': [PHI_INV, PHI_CONJ],
        'Uniform': [0.5, 0.5],
        'Random': [np.random.rand(), np.random.rand()],
        'Golden spiral': [i * PHI_INV % 1 for i in range(2)]
    }
    
    results = {}
    
    for idx, (name, base_values) in enumerate(strategies.items()):
        qf = QuantumFieldVisualizer(n_particles=20)
        
        # Custom initialization
        n_half = qf.n // 2
        qf.ln_r[:n_half] = np.log(base_values[0] + 0.01 * np.random.randn(n_half))
        qf.ln_r[n_half:] = np.log(base_values[1] + 0.01 * np.random.randn(qf.n - n_half))
        
        # Evolve and track
        coherence_history = []
        energy_history = []
        
        for _ in range(200):
            qf.evolve_step()
            coherence_history.append(qf.C_L1)
            energy_history.append(qf.calculate_total_energy())
        
        results[name] = {
            'coherence': coherence_history,
            'energy': energy_history,
            'final_state': qf
        }
        
        # Plot results
        ax = axes[idx//2, idx%2]
        ax.plot(coherence_history, label='Coherence')
        ax.plot(np.array(energy_history)/np.max(energy_history), label='Energy (normalized)')
        ax.set_title(f'{name} Initialization')
        ax.set_xlabel('Time steps')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def explore_resonance_patterns():
    """Explore how resonance patterns form without the phi/2 offset"""
    qf = QuantumFieldVisualizer(n_particles=10)
    
    # Create interactive widget for parameter exploration
    @widgets.interact(
        time_steps=(0, 500, 10),
        resonance_threshold=(0.01, 0.5, 0.01),
        show_matrix=False
    )
    def update(time_steps, resonance_threshold, show_matrix):
        # Reset and evolve
        qf.initialize_field()
        
        for _ in range(time_steps):
            qf.evolve_step()
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Phase space with resonance connections
        r = np.exp(qf.ln_r)
        x = r * np.cos(qf.theta)
        y = r * np.sin(qf.theta)
        
        ax1.scatter(x, y, s=200, c=qf.z, cmap='viridis', edgecolors='black')
        
        # Draw resonance connections
        qf.find_resonant_pairs(threshold=resonance_threshold)
        for i, j, strength in qf.resonance_pairs:
            ax1.plot([x[i], x[j]], [y[i], y[j]], 
                   'blue', alpha=strength, linewidth=strength*3)
        
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_aspect('equal')
        ax1.set_title(f'Resonance Network (threshold={resonance_threshold:.2f})')
        
        # Resonance matrix
        R_matrix = np.zeros((qf.n, qf.n))
        for i in range(qf.n):
            for j in range(qf.n):
                if i != j:
                    R_matrix[i, j] = qf.calculate_resonance(i, j)
        
        im = ax2.imshow(R_matrix, cmap='hot_r', interpolation='nearest')
        ax2.set_title('Resonance Matrix R_ij')
        ax2.set_xlabel('Particle j')
        ax2.set_ylabel('Particle i')
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.show()
        
        if show_matrix:
            print(f"Coherence L1: {qf.C_L1:.4f}")
            print(f"Number of resonant pairs: {len(qf.resonance_pairs)}")
            print(f"Average resonance strength: {np.mean([s for _, _, s in qf.resonance_pairs]):.4f}")


# Run the interactive visualization
if __name__ == "__main__":
    print("=== Quantum Field Dynamics with Phi-Based Encoding ===\n")
    
    print("Key insights implemented:")
    print(f"1. Using 1/φ = {PHI_INV:.3f} and φ-1 = {PHI_CONJ:.3f} for natural harmonic encoding")
    print("2. Removed artificial φ/2 offset from resonance calculation")
    print("3. Time step slows down as coherence increases (quantum entropy stabilization)")
    print("4. Triangular interactions create higher-order correlations\n")
    
    print("Creating interactive visualization...")
    print("Controls: Play/Pause, Toggle trails/resonances/triangles, Speed +/-")
    
    # Create the main visualization
    qf, anim = create_interactive_visualization()
    
    # Additional analysis
    print("\nFor Jupyter notebook exploration:")
    print("1. analyze_phi_encoding() - Compare initialization strategies")
    print("2. explore_resonance_patterns() - Interactive parameter exploration")
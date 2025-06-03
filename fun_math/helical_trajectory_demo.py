import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Universal constants
PHI = (1 + np.sqrt(5)) / 2
GAP = 1 - 1/PHI
EPS = 1e-10

def helical_trajectory_visualization():
    """
    Demonstrate helical trajectories in log-cylindrical phase space
    with different parameters based on the golden ratio.
    """
    # Figure setup
    fig = plt.figure(figsize=(15, 12))
    
    # Four different helical patterns
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    
    # Time steps
    t = np.linspace(0, 8*np.pi, 1000)
    
    # 1. Standard helical trajectory with φ-based radius
    r1 = 1/PHI  # Inner band at r = 1/φ
    omega_theta1 = 1.0
    omega_z1 = 2*np.pi/(PHI**3)
    
    x1 = r1 * np.cos(omega_theta1 * t)
    y1 = r1 * np.sin(omega_theta1 * t)
    z1 = omega_z1 * t
    
    ax1.plot(x1, y1, z1, 'b-', linewidth=2)
    ax1.set_title('Standard Helical Trajectory\n' + 
                 f'r = 1/φ = {r1:.3f}, ω_z = 2π/φ³ = {omega_z1:.3f}')
    
    # 2. Fibonacci-modulated helical trajectory
    r2 = PHI - 1  # Outer band at r = φ-1
    omega_theta2 = 1.0
    
    # Fibonacci sequence for frequencies
    fib = [1, 1]
    for i in range(10):
        fib.append(fib[-1] + fib[-2])
    
    # Modulated position based on Fibonacci
    x2 = r2 * np.cos(omega_theta2 * t)
    y2 = r2 * np.sin(omega_theta2 * t)
    
    # Add Fibonacci modulation
    fib_mod = np.zeros_like(t)
    for i, f in enumerate(fib[:8]):
        fib_mod += np.sin(t/f) / (i+1)**(1/PHI)
    
    # Scale modulation
    fib_mod = fib_mod / np.max(np.abs(fib_mod)) * np.pi/2
    
    # Apply modulation
    z2 = omega_z1 * t + fib_mod
    
    ax2.plot(x2, y2, z2, 'g-', linewidth=2)
    ax2.set_title('Fibonacci-Modulated Helical Trajectory\n' + 
                 f'r = φ-1 = {r2:.3f}, Fibonacci frequencies')
    
    # 3. Three-step triangular evolution
    r3 = 1/PHI
    omega_theta3 = 2*np.pi/3  # Three steps per cycle
    omega_z3 = 2*np.pi/(PHI**3)
    
    # Three-step positions with discontinuities
    x3 = []
    y3 = []
    z3 = []
    
    for i in range(len(t)-1):
        # Current phase
        phase = i % 3
        
        # Calculate influence weights based on triangulation
        past_influence = np.sin(phase * 2*np.pi/3 + 2*np.pi/3)
        present_influence = np.sin(phase * 2*np.pi/3 + 4*np.pi/3)
        future_influence = np.sin(phase * 2*np.pi/3)
        
        # Base position
        theta = omega_theta3 * t[i]
        x_base = r3 * np.cos(theta)
        y_base = r3 * np.sin(theta)
        z_base = omega_z3 * t[i]
        
        # Apply influences
        theta_shift = 0.1 * (past_influence + present_influence + future_influence)
        x3.append(x_base + 0.05 * past_influence)
        y3.append(y_base + 0.05 * present_influence)
        z3.append(z_base + 0.05 * future_influence)
    
    ax3.plot(x3, y3, z3, 'r-', linewidth=2)
    ax3.set_title('Three-Step Triangular Evolution\n' + 
                 'Past, Present, Future influences')
    
    # 4. Tachyonic helical trajectory
    # Phase velocity exceeds c when r > 2c·φ^(n+2)/π
    c = 1.0  # Speed of light
    n = 0
    r_critical = 2*c*(PHI**(n+2))/np.pi
    r4 = r_critical * 1.1  # Slightly above critical
    omega_theta4 = 1.0
    omega_z4 = 2*np.pi/(PHI**3)
    
    # Position
    x4 = r4 * np.cos(omega_theta4 * t)
    y4 = r4 * np.sin(omega_theta4 * t)
    z4 = omega_z4 * t
    
    # Phase velocity
    v_phase = r4 * omega_theta4
    
    # Proper time (becomes imaginary when v > c)
    # For visualization, use real part only
    dtau_dt = np.sqrt(np.abs(1 - v_phase**2/c**2))
    tau = np.zeros_like(t)
    for i in range(1, len(t)):
        tau[i] = tau[i-1] + dtau_dt * (t[i] - t[i-1])
    
    # Color mapping based on proper time
    colors = plt.cm.plasma(tau/np.max(tau))
    
    # Plot with color gradient
    for i in range(len(t)-1):
        ax4.plot(x4[i:i+2], y4[i:i+2], z4[i:i+2], 
                 color=colors[i], linewidth=2)
    
    ax4.set_title('Tachyonic Helical Trajectory\n' + 
                 f'r = {r4:.3f} > r_crit = {r_critical:.3f}, v_phase = {v_phase:.3f}c')
    
    # Common settings for all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        # Labels
        ax.set_xlabel('X = r·cos(θ)')
        ax.set_ylabel('Y = r·sin(θ)')
        ax.set_zlabel('Z')
        
        # Equal aspect ratio for x and y
        ax.set_box_aspect([1, 1, 2])
        
        # Add cylinder at preferred radii
        theta_circle = np.linspace(0, 2*np.pi, 100)
        z_levels = np.linspace(0, np.max(z1), 5)
        
        # Inner band at r = 1/φ
        r_inner = 1/PHI
        for z_level in z_levels:
            x_inner = r_inner * np.cos(theta_circle)
            y_inner = r_inner * np.sin(theta_circle)
            ax.plot(x_inner, y_inner, [z_level]*len(theta_circle), 
                   'gold', alpha=0.3, linewidth=1)
        
        # Outer band at r = φ-1
        r_outer = PHI - 1
        for z_level in z_levels:
            x_outer = r_outer * np.cos(theta_circle)
            y_outer = r_outer * np.sin(theta_circle)
            ax.plot(x_outer, y_outer, [z_level]*len(theta_circle), 
                   'magenta', alpha=0.3, linewidth=1)
        
        # Critical radius for tachyonic behavior
        if ax == ax4:
            for z_level in z_levels:
                x_crit = r_critical * np.cos(theta_circle)
                y_crit = r_critical * np.sin(theta_circle)
                ax.plot(x_crit, y_crit, [z_level]*len(theta_circle), 
                       'red', alpha=0.3, linewidth=1, linestyle='--')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('outputs/helical_trajectories.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to outputs/helical_trajectories.png")
    
    return fig

def born_rule_helix_visualization():
    """
    Visualize how the Born rule constraint r² + z² = 1
    affects the helical trajectory in phase space.
    """
    # Figure setup
    fig = plt.figure(figsize=(15, 8))
    
    # Two subplots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Time steps
    t = np.linspace(0, 6*np.pi, 1000)
    
    # Base parameters
    omega_theta = 1.0
    omega_z = 2*np.pi/(PHI**3)
    
    # 1. Unconstrained helical trajectory
    r1 = 1/PHI  # Fixed radius
    
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
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('outputs/born_rule_helix.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to outputs/born_rule_helix.png")
    
    return fig

def pi_by_2_spacing_visualization():
    """
    Visualize the π/2 spacing → helical structure
    where phase increments by π/2 create a 4-step helical cycle.
    """
    # Figure setup
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Time steps
    t = np.linspace(0, 8*np.pi, 1000)
    
    # Base parameters
    r = 1/PHI
    omega_z = 2*np.pi/(PHI**3)
    
    # Create four helical trajectories with π/2 phase differences
    for i, phase_shift in enumerate([0, np.pi/2, np.pi, 3*np.pi/2]):
        theta = t + phase_shift
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = omega_z * t
        
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
        z_val = omega_z * t[idx]
        points_x = []
        points_y = []
        points_z = []
        
        for phase_shift in [0, np.pi/2, np.pi, 3*np.pi/2]:
            theta = t[idx] + phase_shift
            points_x.append(r * np.cos(theta))
            points_y.append(r * np.sin(theta))
            points_z.append(z_val)
        
        # Add the first point again to close the loop
        points_x.append(points_x[0])
        points_y.append(points_y[0])
        points_z.append(points_z[0])
        
        ax.plot(points_x, points_y, points_z, 'k--', alpha=0.5)
    
    # Cylinder at r = 1/φ
    theta_circle = np.linspace(0, 2*np.pi, 100)
    z_levels = np.linspace(0, np.max(z), 5)
    
    for z_level in z_levels:
        x_circle = r * np.cos(theta_circle)
        y_circle = r * np.sin(theta_circle)
        ax.plot(x_circle, y_circle, [z_level]*len(theta_circle), 
               'gold', alpha=0.3, linewidth=1)
    
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
    plt.savefig('outputs/pi_by_2_spacing_helix.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to outputs/pi_by_2_spacing_helix.png")
    
    return fig

def z_modulation_visualization():
    """
    Visualize how z-coordinate modulates the coordinate system,
    creating a binary oscillator controlling 90° rotations.
    """
    # Figure setup
    fig = plt.figure(figsize=(15, 8))
    
    # Two subplots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    # Time steps
    t = np.linspace(0, 4*np.pi, 1000)
    
    # Base parameters
    r = 1/PHI
    omega_theta = 1.0
    
    # Z modulation function: binary oscillator
    # Map z from [0, 2π) to {0, 1} with smooth transition
    z = (t * PHI) % (2*np.pi)
    z_binary = (np.sin(z) > 0).astype(float)
    
    # Coordinate system modulation
    # When z_binary = 0: use Cartesian
    # When z_binary = 1: use 90° rotated Cartesian
    x = r * np.cos(omega_theta * t) * (1 - z_binary) + r * np.sin(omega_theta * t) * z_binary
    y = r * np.sin(omega_theta * t) * (1 - z_binary) - r * np.cos(omega_theta * t) * z_binary
    
    # 3D plot of trajectory
    ax1.plot(x, y, z, 'b-', linewidth=2)
    
    # Mark points where z_binary changes
    change_points = np.where(np.abs(np.diff(z_binary)) > 0.5)[0]
    for idx in change_points:
        ax1.scatter(x[idx], y[idx], z[idx], color='r', s=50)
    
    # Cylinder at r = 1/φ
    theta_circle = np.linspace(0, 2*np.pi, 100)
    z_levels = np.linspace(0, 2*np.pi, 5)
    
    for z_level in z_levels:
        x_circle = r * np.cos(theta_circle)
        y_circle = r * np.sin(theta_circle)
        ax1.plot(x_circle, y_circle, [z_level]*len(theta_circle), 
                'gold', alpha=0.3, linewidth=1)
    
    # Plot coordinate system states
    ax2.plot(t, z_binary, 'g-', linewidth=2, label='Coordinate System State')
    ax2.plot(t, 0.5 + 0.5*np.sin(z), 'b--', linewidth=1, alpha=0.5, label='Z Modulation')
    
    # Mark transition points
    for idx in change_points:
        ax2.axvline(x=t[idx], color='r', linestyle='--', alpha=0.5)
    
    # Add reference lines
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax2.axhline(y=1, color='k', linestyle='-', alpha=0.2)
    
    # Labels and titles
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Z-Modulated Trajectory\nCoordinate System Switching')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Coordinate System State')
    ax2.set_title('Z-Coordinate as Topological Modulator\nBinary Oscillator Controlling 90° Rotations')
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Equal aspect ratio for 3D plot
    ax1.set_box_aspect([1, 1, 2])
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('outputs/z_modulation_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to outputs/z_modulation_visualization.png")
    
    return fig

def tachyonic_closed_timelike_curves():
    """
    Visualize tachyonic closed timelike curves in the helical structure
    where phase velocity exceeds c.
    """
    # Figure setup
    fig = plt.figure(figsize=(15, 8))
    
    # Two subplots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    # Parameters
    c = 1.0  # Speed of light
    omega = 1.0  # Angular frequency
    
    # Critical radius where phase velocity equals c
    r_critical = c/omega
    
    # Three radii: subcritical, critical, and supercritical
    radii = [0.8*r_critical, r_critical, 1.2*r_critical]
    labels = ['Subluminal', 'Critical', 'Superluminal']
    colors = ['blue', 'green', 'red']
    
    # Time parameter
    t = np.linspace(0, 4*np.pi, 1000)
    
    # Plot helical trajectories
    for i, r in enumerate(radii):
        # Helix coordinates
        x = r * np.cos(omega * t)
        y = r * np.sin(omega * t)
        z = t / (2*np.pi)  # One unit in z per revolution
        
        # Phase velocity
        v_phase = r * omega
        
        # Plot 3D helix
        ax1.plot(x, y, z, color=colors[i], linewidth=2, label=f'{labels[i]} (r={r:.2f})')
        
        # Compute proper time
        dtau_dt = np.sqrt(np.abs(1 - v_phase**2/c**2))
        tau = np.zeros_like(t)
        
        # Accumulate proper time (negative for superluminal case)
        sign = 1 if v_phase <= c else -1
        for j in range(1, len(t)):
            tau[j] = tau[j-1] + sign * dtau_dt * (t[j] - t[j-1])
        
        # Plot proper time vs. coordinate time
        ax2.plot(t, tau, color=colors[i], linewidth=2, label=f'{labels[i]} (v={v_phase:.2f}c)')
    
    # Add closed timelike curve for superluminal case
    r_ctc = 2.0 * r_critical
    v_ctc = r_ctc * omega
    dtau_dt_ctc = np.sqrt(np.abs(1 - v_ctc**2/c**2))
    
    # Create CTC by ensuring τ returns to starting point
    t_ctc = np.linspace(0, 2*np.pi, 500)
    x_ctc = r_ctc * np.cos(omega * t_ctc)
    y_ctc = r_ctc * np.sin(omega * t_ctc)
    z_ctc = t_ctc / (2*np.pi)
    
    tau_ctc = np.zeros_like(t_ctc)
    for j in range(1, len(t_ctc)):
        tau_ctc[j] = tau_ctc[j-1] - dtau_dt_ctc * (t_ctc[j] - t_ctc[j-1])
    
    # Plot CTC
    ax1.plot(x_ctc, y_ctc, z_ctc, color='purple', linewidth=3, 
            label=f'CTC (r={r_ctc:.2f}, v={v_ctc:.2f}c)')
    ax2.plot(t_ctc, tau_ctc, color='purple', linewidth=3, 
            label=f'CTC (v={v_ctc:.2f}c)')
    
    # Add critical cylinder
    theta_circle = np.linspace(0, 2*np.pi, 100)
    z_levels = np.linspace(0, np.max(z), 5)
    
    for z_level in z_levels:
        x_crit = r_critical * np.cos(theta_circle)
        y_crit = r_critical * np.sin(theta_circle)
        ax1.plot(x_crit, y_crit, [z_level]*len(theta_circle), 
               'red', alpha=0.3, linewidth=1, linestyle='--')
    
    # Labels and titles
    ax1.set_xlabel('X = r·cos(θ)')
    ax1.set_ylabel('Y = r·sin(θ)')
    ax1.set_zlabel('Z (proper time)')
    ax1.set_title('Helical Trajectories in Phase Space\n' +
                 f'Critical radius r_c = {r_critical:.2f} where v_phase = c')
    
    ax2.set_xlabel('Coordinate Time (t)')
    ax2.set_ylabel('Proper Time (τ)')
    ax2.set_title('Proper Time vs. Coordinate Time\n' +
                 'Negative slope for superluminal case creates CTCs')
    ax2.grid(True, alpha=0.3)
    
    # Legend
    ax1.legend(loc='upper left')
    ax2.legend()
    
    # Equal aspect ratio for 3D plot
    ax1.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('outputs/tachyonic_ctc_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to outputs/tachyonic_ctc_visualization.png")
    
    return fig

if __name__ == "__main__":
    print("=== HELICAL TRAJECTORY DEMONSTRATION ===")
    print(f"φ = {PHI:.6f}")
    print(f"GAP = {GAP:.6f}")
    
    print("\nGenerating helical trajectory visualizations...")
    
    # Create helical visualizations
    helical_trajectory_visualization()
    born_rule_helix_visualization()
    pi_by_2_spacing_visualization()
    z_modulation_visualization()
    tachyonic_closed_timelike_curves()
    
    print("\nAll visualizations completed!")
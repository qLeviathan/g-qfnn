import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Golden ratio and related constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI

def fibonacci(n):
    """Generate nth Fibonacci number"""
    if n <= 1:
        return 1
    return int((PHI**n - (-PHI_INV)**n) / np.sqrt(5))

def inverse_fibonacci_sequence(n_terms):
    """Generate inverse Fibonacci sequence"""
    fib_sequence = [fibonacci(i) for i in range(1, n_terms + 1)]
    return [1/f for f in fib_sequence]

def analyze_phi_regularization():
    """
    Create comprehensive analysis of phi regularization and its connection
    to inverse Fibonacci sequences
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========== 1. Singularity Structure Comparison ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    theta = np.linspace(-2*np.pi, 2*np.pi, 1000)
    
    # Original singular functions
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Regularized versions
    reg_cos = 1 / (cos_theta**2 + 1)
    reg_sin = 1 / (sin_theta**2 + 1)
    
    # Plot with careful handling of singularities
    ax1.plot(theta, np.where(np.abs(cos_theta) > 0.01, 1/cos_theta**2, np.nan), 
             'r--', alpha=0.5, label='1/cos²θ (singular)')
    ax1.plot(theta, np.where(np.abs(sin_theta) > 0.01, 1/sin_theta**2, np.nan), 
             'b--', alpha=0.5, label='1/sin²θ (singular)')
    
    ax1.plot(theta, reg_cos, 'r-', linewidth=2, label='1/(cos²θ + 1)')
    ax1.plot(theta, reg_sin, 'b-', linewidth=2, label='1/(sin²θ + 1)')
    
    # Mark singularity positions
    for k in range(-2, 3):
        ax1.axvline(k*np.pi, color='gray', alpha=0.3, linestyle=':')
        ax1.axvline(k*np.pi + np.pi/2, color='gray', alpha=0.3, linestyle=':')
    
    ax1.set_xlim(-2*np.pi, 2*np.pi)
    ax1.set_ylim(0, 5)
    ax1.set_xlabel('θ (radians)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title('Singularity Regularization: Before and After', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add pi labels
    ax1.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
    ax1.set_xticklabels(['-2π', '-π', '0', 'π', '2π'])
    
    # ========== 2. Inverse Fibonacci Connection ==========
    ax2 = fig.add_subplot(gs[1, 0])
    
    n_terms = 15
    inv_fib = inverse_fibonacci_sequence(n_terms)
    
    # Phi-based regularization values
    phi_reg_values = []
    for n in range(1, n_terms + 1):
        # Different powers of phi in denominator
        val = 1 / (PHI**n + 1)
        phi_reg_values.append(val)
    
    ax2.semilogy(range(1, n_terms + 1), inv_fib, 'bo-', label='1/F_n (Inverse Fibonacci)', markersize=8)
    ax2.semilogy(range(1, n_terms + 1), phi_reg_values, 'r^-', label='1/(φⁿ + 1)', markersize=8)
    
    # Show the ratio converges to phi
    ratios = [inv_fib[i] / phi_reg_values[i] for i in range(len(inv_fib))]
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(1, n_terms + 1), ratios, 'g--', label='Ratio', alpha=0.7)
    ax2_twin.axhline(PHI, color='gold', linestyle=':', label=f'φ = {PHI:.3f}')
    ax2_twin.set_ylabel('Ratio', color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    
    ax2.set_xlabel('n')
    ax2.set_ylabel('Value (log scale)', color='black')
    ax2.set_title('Inverse Fibonacci vs Phi Regularization', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2_twin.legend(loc='center right')
    
    # ========== 3. Phase Space Potential Energy ==========
    ax3 = fig.add_subplot(gs[1, 1:], projection='3d')
    
    # Create meshgrid
    r_vals = np.linspace(0.1, 2, 50)
    theta_vals = np.linspace(0, 2*np.pi, 100)
    R, THETA = np.meshgrid(r_vals, theta_vals)
    
    # Calculate regularized potential energy
    U_regularized = np.zeros_like(R)
    for i in range(len(theta_vals)):
        for j in range(len(r_vals)):
            # z = 0 case (cos-based)
            U_regularized[i, j] = -np.log(np.cos(theta_vals[i])**2 + 1) + \
                                  (np.log(r_vals[j]) - np.log(PHI/2))**2
    
    # Plot surface
    surf = ax3.plot_surface(R * np.cos(THETA), R * np.sin(THETA), U_regularized,
                           cmap='viridis', alpha=0.8, edgecolor='none')
    
    # Mark phi-related radii
    phi_circle = plt.Circle((0, 0), PHI/2, fill=False, color='gold', linewidth=3)
    ax3.add_patch(phi_circle)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Potential Energy')
    ax3.set_title('Regularized Potential Energy Landscape', fontsize=12, fontweight='bold')
    
    # ========== 4. Flow Dynamics Comparison ==========
    ax4 = fig.add_subplot(gs[2, :2])
    
    # Create vector field
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # Convert to cylindrical
    R_field = np.sqrt(X**2 + Y**2)
    THETA_field = np.arctan2(Y, X)
    
    # Calculate gradients with regularization
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(len(x)):
        for j in range(len(y)):
            r = R_field[j, i]
            theta = THETA_field[j, i]
            
            if r > 0.1:  # Avoid origin
                # Regularized gradient
                dU_dr = 2 * (np.log(r) - np.log(PHI/2)) / r
                dU_dtheta = 2 * np.cos(theta) * np.sin(theta) / (np.cos(theta)**2 + 1)
                
                # Convert to Cartesian
                U[j, i] = -dU_dr * np.cos(theta) + dU_dtheta * np.sin(theta) / r
                V[j, i] = -dU_dr * np.sin(theta) - dU_dtheta * np.cos(theta) / r
    
    # Normalize for visibility
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = U / (magnitude + 0.1)
    V_norm = V / (magnitude + 0.1)
    
    # Plot vector field
    ax4.quiver(X, Y, U_norm, V_norm, magnitude, cmap='plasma', scale=20)
    
    # Add phi circles
    circle1 = plt.Circle((0, 0), PHI/2, fill=False, color='gold', linewidth=2, label=f'r = φ/2')
    circle2 = plt.Circle((0, 0), PHI, fill=False, color='orange', linewidth=2, label=f'r = φ')
    circle3 = plt.Circle((0, 0), 1/PHI, fill=False, color='red', linewidth=2, label=f'r = 1/φ')
    
    ax4.add_patch(circle1)
    ax4.add_patch(circle2)
    ax4.add_patch(circle3)
    
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-2, 2)
    ax4.set_aspect('equal')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('Regularized Flow Field with Phi-Based Attractors', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ========== 5. Fibonacci Spiral in Phase Space ==========
    ax5 = fig.add_subplot(gs[2, 2])
    
    # Generate Fibonacci spiral points
    n_points = 144  # Fibonacci number
    golden_angle = 2 * np.pi / PHI**2
    
    spiral_r = []
    spiral_theta = []
    
    for i in range(n_points):
        # Radius grows with inverse Fibonacci
        r = 1 / np.sqrt(i + 1)  # Inverse square root for better visualization
        theta = i * golden_angle
        
        spiral_r.append(r)
        spiral_theta.append(theta)
    
    # Convert to Cartesian
    spiral_x = [r * np.cos(theta) for r, theta in zip(spiral_r, spiral_theta)]
    spiral_y = [r * np.sin(theta) for r, theta in zip(spiral_r, spiral_theta)]
    
    # Color by index to show progression
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    ax5.scatter(spiral_x, spiral_y, c=colors, s=20, alpha=0.6)
    
    # Connect points to show spiral
    ax5.plot(spiral_x, spiral_y, 'k-', alpha=0.2, linewidth=0.5)
    
    ax5.set_xlim(-1.5, 1.5)
    ax5.set_ylim(-1.5, 1.5)
    ax5.set_aspect('equal')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_title('Fibonacci Spiral with Inverse Scaling', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # ========== 6. Energy Well Profiles ==========
    ax6 = fig.add_subplot(gs[3, :])
    
    theta_profile = np.linspace(0, 2*np.pi, 500)
    
    # Different regularization parameters
    reg_params = [0, 0.1, 0.5, 1.0, PHI]
    
    for reg in reg_params:
        if reg == 0:
            # Singular case (clip for visualization)
            energy = np.clip(-np.log(np.abs(np.cos(theta_profile)**2) + 1e-10), -5, 5)
            ax6.plot(theta_profile, energy, '--', alpha=0.5, label=f'Singular')
        else:
            energy = -np.log(np.cos(theta_profile)**2 + reg)
            label = f'Reg = {reg:.2f}' if reg != PHI else f'Reg = φ'
            ax6.plot(theta_profile, energy, linewidth=2, label=label)
    
    ax6.set_xlim(0, 2*np.pi)
    ax6.set_ylim(-3, 3)
    ax6.set_xlabel('θ (radians)', fontsize=12)
    ax6.set_ylabel('Energy', fontsize=12)
    ax6.set_title('Energy Well Profiles for Different Regularization Values', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Mark key positions
    ax6.axvline(np.pi/2, color='red', alpha=0.3, linestyle=':')
    ax6.axvline(3*np.pi/2, color='red', alpha=0.3, linestyle=':')
    ax6.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax6.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    plt.suptitle('Phi Regularization and Inverse Fibonacci Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# Generate the analysis
fig = analyze_phi_regularization()
plt.show()

# Additional analysis: Numerical demonstration
print("NUMERICAL ANALYSIS OF PHI REGULARIZATION")
print("=" * 50)

# Show how regularization relates to Fibonacci
print("\n1. Inverse Fibonacci vs Phi Regularization:")
for n in range(1, 10):
    fib_n = fibonacci(n)
    inv_fib = 1/fib_n
    phi_reg = 1/(PHI**n + 1)
    ratio = inv_fib / phi_reg
    print(f"n={n}: 1/F_{n} = {inv_fib:.6f}, 1/(φ^{n}+1) = {phi_reg:.6f}, ratio = {ratio:.4f}")

# Show how the +1 preserves phi relationships
print("\n2. Phi Relationships with +1 Regularization:")
print(f"φ² = {PHI**2:.6f}")
print(f"φ + 1 = {PHI + 1:.6f}")
print(f"φ² - (φ + 1) = {PHI**2 - (PHI + 1):.6f}")
print(f"φ² + 1 = {PHI**2 + 1:.6f}")
print(f"(φ² + 1)/φ = {(PHI**2 + 1)/PHI:.6f} ≈ φ + 1/φ = {PHI + 1/PHI:.6f}")

# Show bounded dynamics
print("\n3. Bounded Dynamics with Regularization:")
theta_test = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
for theta in theta_test:
    singular = 1/np.cos(theta)**2 if np.cos(theta) != 0 else np.inf
    regularized = 1/(np.cos(theta)**2 + 1)
    print(f"θ = {theta:.3f}: Singular = {singular:.3f}, Regularized = {regularized:.3f}")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
import os

def analyze_fibonacci_phi_relationship():
    """Demonstrate that φ is the growth rate of Fibonacci sequence"""
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Generate Fibonacci numbers
    fib = [1, 1]
    for i in range(50):
        fib.append(fib[-1] + fib[-2])
    
    # Calculate ratios
    ratios = [fib[i+1] / fib[i] for i in range(len(fib)-1)]
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Fibonacci growth vs φⁿ growth
    ax1 = plt.subplot(2, 3, 1)
    n = np.arange(len(fib))
    phi_growth = phi**n / np.sqrt(5)
    
    ax1.semilogy(n, fib, 'bo-', label='Fibonacci numbers', markersize=4)
    ax1.semilogy(n, phi_growth, 'r--', label=f'φⁿ/√5', linewidth=2)
    ax1.set_xlabel('n')
    ax1.set_ylabel('Value (log scale)')
    ax1.set_title('Fibonacci vs φⁿ Growth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Ratio convergence to φ
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(ratios[:30], 'b.-', markersize=8)
    ax2.axhline(y=phi, color='gold', linewidth=3, label=f'φ = {phi:.6f}')
    ax2.fill_between(range(30), phi-0.1, phi+0.1, alpha=0.2, color='gold')
    ax2.set_xlabel('n')
    ax2.set_ylabel('Fₙ₊₁/Fₙ')
    ax2.set_title('Fibonacci Ratio Convergence to φ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error decay
    ax3 = plt.subplot(2, 3, 3)
    errors = [abs(ratio - phi) for ratio in ratios[:30]]
    ax3.semilogy(errors, 'g.-', markersize=8)
    ax3.set_xlabel('n')
    ax3.set_ylabel('|Fₙ₊₁/Fₙ - φ|')
    ax3.set_title('Convergence Error (Exponential Decay)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Fibonacci spiral vs Golden spiral
    ax4 = plt.subplot(2, 3, 4, projection='polar')
    
    # Fibonacci spiral (discrete)
    theta_fib = []
    r_fib = []
    for i in range(1, 15):
        angle = i * 2 * np.pi / phi**2  # Golden angle
        theta_fib.append(angle)
        r_fib.append(fib[i])
    
    ax4.plot(theta_fib, r_fib, 'bo-', markersize=8, label='Fibonacci points')
    
    # Golden spiral (continuous)
    theta_golden = np.linspace(0, 6*np.pi, 1000)
    r_golden = np.exp(theta_golden / (2*np.pi) * np.log(phi))
    r_golden = r_golden * fib[8] / r_golden[int(8 * len(theta_golden)/(6*np.pi))]
    ax4.plot(theta_golden, r_golden, 'gold', linewidth=2, label='Golden spiral')
    
    ax4.set_title('Fibonacci Points on Golden Spiral')
    ax4.legend()
    
    # 5. Matrix representation
    ax5 = plt.subplot(2, 3, 5)
    ax5.text(0.1, 0.9, 'Matrix Form:', fontsize=14, weight='bold', transform=ax5.transAxes)
    ax5.text(0.1, 0.75, 'Fibonacci Matrix Equation:',
             fontsize=12, weight='bold', transform=ax5.transAxes)
    ax5.text(0.1, 0.6, '[F(n+1)]   [1 1]^n   [1]',
             fontsize=11, family='monospace', transform=ax5.transAxes)
    ax5.text(0.1, 0.5, '[F(n)  ] = [1 0]   × [0]',
             fontsize=11, family='monospace', transform=ax5.transAxes)
    
    ax5.text(0.1, 0.35, 'Eigenvalues of Fibonacci matrix:', fontsize=12, weight='bold', transform=ax5.transAxes)
    ax5.text(0.1, 0.25, f'λ₁ = φ = {phi:.6f}', fontsize=12, transform=ax5.transAxes)
    ax5.text(0.1, 0.15, f'λ₂ = -1/φ = {-1/phi:.6f}', fontsize=12, transform=ax5.transAxes)
    
    ax5.text(0.1, 0.05, f'Therefore: Fₙ ~ φⁿ for large n', fontsize=12, 
             weight='bold', color='red', transform=ax5.transAxes)
    ax5.axis('off')
    
    # 6. Hurricane/Tornado connection
    ax6 = plt.subplot(2, 3, 6)
    
    # Generate spiral arms
    theta = np.linspace(0, 4*np.pi, 1000)
    
    # Hurricane spiral (logarithmic spiral with pitch angle related to φ)
    pitch_angle = np.arctan(1/phi)  # ≈ 32°
    a = 1
    b = 1/np.tan(pitch_angle)
    r_hurricane = a * np.exp(b * theta)
    
    # Plot in Cartesian
    x = r_hurricane * np.cos(theta)
    y = r_hurricane * np.sin(theta)
    
    # Normalize
    scale = 10 / np.max(np.sqrt(x**2 + y**2))
    x, y = x * scale, y * scale
    
    ax6.plot(x, y, 'b-', linewidth=2, label=f'Hurricane spiral (pitch={np.degrees(pitch_angle):.1f}°)')
    ax6.plot(-x, -y, 'b-', linewidth=2)
    
    # Add eye
    circle = plt.Circle((0, 0), 0.5, color='red', alpha=0.5)
    ax6.add_patch(circle)
    
    # Add Fibonacci growth zones
    for i in range(3, 8):
        radius = fib[i] / fib[7] * 10
        circle = plt.Circle((0, 0), radius, fill=False, 
                          linestyle='--', color='green', alpha=0.5)
        ax6.add_patch(circle)
        ax6.text(radius/np.sqrt(2), radius/np.sqrt(2), f'F_{i}', fontsize=8)
    
    ax6.set_xlim(-12, 12)
    ax6.set_ylim(-12, 12)
    ax6.set_aspect('equal')
    ax6.set_title('Hurricane Spiral with Fibonacci Zones')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.suptitle(f'The Golden Ratio φ = {phi:.6f} IS the Growth Rate of Fibonacci Sequence', 
                 fontsize=16, weight='bold')
    plt.tight_layout()
    
    # Save figure to outputs folder
    os.makedirs('fun_math/outputs', exist_ok=True)
    fig.savefig('fun_math/outputs/fibonacci_phi_analysis.png', dpi=300, bbox_inches='tight')
    
    # Additional analysis figure
    fig2, ((ax7, ax8), (ax9, ax10)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 7. Direct growth rate comparison
    ax7.set_title('Growth Rate Analysis')
    n_vals = np.arange(5, 30)
    fib_growth_rate = [fib[i]/fib[i-1] for i in n_vals]
    exponential_fit = [phi for _ in n_vals]
    
    ax7.plot(n_vals, fib_growth_rate, 'bo-', label='Fib(n)/Fib(n-1)')
    ax7.plot(n_vals, exponential_fit, 'r--', linewidth=2, label=f'φ = {phi:.4f}')
    ax7.set_xlabel('n')
    ax7.set_ylabel('Growth Rate')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Binet's Formula visualization
    ax8.set_title("Binet's Formula Components")
    n = np.arange(1, 20)
    
    # Components of Binet's formula
    phi_component = phi**n / np.sqrt(5)
    psi_component = ((-1/phi)**n) / np.sqrt(5)
    
    ax8.plot(n, phi_component, 'g-', linewidth=2, label='φⁿ/√5 (dominant)')
    ax8.plot(n, np.abs(psi_component), 'r--', linewidth=1, label='|ψⁿ|/√5 (vanishing)')
    ax8.plot(n, [fib[int(i)] for i in n], 'bo', markersize=8, label='Actual Fibonacci')
    ax8.set_xlabel('n')
    ax8.set_ylabel('Value')
    ax8.set_yscale('log')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Connection to continuous systems
    ax9.set_title('Discrete (Fibonacci) vs Continuous (φ-growth)')
    t = np.linspace(0, 10, 1000)
    continuous_growth = np.exp(t * np.log(phi))
    
    ax9.plot(t, continuous_growth, 'gold', linewidth=2, label=f'e^(t·ln(φ))')
    
    # Overlay discrete Fibonacci points
    fib_times = np.arange(len(fib[:15]))
    ax9.plot(fib_times, fib[:15], 'bo', markersize=8, label='Fibonacci numbers')
    
    ax9.set_xlabel('Time/Index')
    ax9.set_ylabel('Value')
    ax9.set_yscale('log')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Phase space portrait
    ax10.set_title('Fibonacci Phase Space (Fₙ vs Fₙ₊₁)')
    ax10.plot(fib[:-1][:20], fib[1:][:20], 'b.-', markersize=8)
    
    # Theoretical line with slope φ
    max_val = fib[20]
    theory_line = np.linspace(0, max_val, 100)
    ax10.plot(theory_line, phi * theory_line, 'r--', linewidth=2, label=f'slope = φ')
    
    ax10.set_xlabel('Fₙ')
    ax10.set_ylabel('Fₙ₊₁')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save second figure
    fig2.savefig('fun_math/outputs/fibonacci_detailed_analysis.png', dpi=300, bbox_inches='tight')
    
    # Print numerical verification
    print("Numerical Verification:")
    print(f"φ = {phi:.10f}")
    print(f"φ² = {phi**2:.10f}")
    print(f"1 + φ = {1 + phi:.10f}")
    print(f"Verification: φ² - (1 + φ) = {phi**2 - (1 + phi):.2e}")
    print("\nFibonacci ratio convergence:")
    for i in range(10, 15):
        ratio = fib[i+1] / fib[i]
        error = abs(ratio - phi)
        print(f"F_{i+1}/F_{i} = {ratio:.8f}, error = {error:.2e}")
    
    # Hurricane connection
    print(f"\nHurricane spiral pitch angle: {np.degrees(pitch_angle):.1f}°")
    print(f"This creates growth rate of e^(2π/tan(pitch)) = e^(2π·φ) per revolution")
    
    return fig, fig2

# Run the analysis
fig1, fig2 = analyze_fibonacci_phi_relationship()
plt.show()

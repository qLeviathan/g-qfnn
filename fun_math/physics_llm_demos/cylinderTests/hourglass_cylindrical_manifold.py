import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    """Custom 3D arrow for visualization with proper 3D projection"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)
        
    def draw(self, renderer):
        super().draw(renderer)

def generate_hourglass_embedding(vocab_size, num_cylinders=3, phi=1.618034):
    """
    Generate token embeddings on a multi-cylindrical manifold with hourglass-shaped radius
    
    Args:
        vocab_size: Number of tokens
        num_cylinders: Number of coupled cylinders (m)
        phi: Golden ratio for distribution
        
    Returns:
        embeddings: Token embeddings of shape [vocab_size, 2*num_cylinders]
    """
    embeddings = np.zeros((vocab_size, 2 * num_cylinders))
    
    for v in range(vocab_size):
        # Calculate base radius using Fibonacci modulo 1 (similar to phase)
        # This creates a conic/hourglass effect with varying radius
        base_radius_factor = (phi * v) % 1
        
        # Create hourglass shape: wider at edges, narrower in middle
        # r = 0.2 + 0.8 * |2x - 1| creates an hourglass shape from 0 to 1
        hourglass_factor = 0.2 + 0.8 * abs(2 * base_radius_factor - 1)
        
        for i in range(num_cylinders):
            # Generate phase angles using different powers of phi
            theta = 2 * np.pi * ((phi**(i+1) * v) % 1)
            
            # Each cylinder has a different radius scaling
            # This creates the branching effect like a multi-level hourglass
            cylinder_radius = (i+1) / num_cylinders  # Different base size for each cylinder
            
            # Apply Fibonacci modulo for radius of each cylinder
            # (phi^(i+2) * v) % 1 gives a different sequence for each cylinder
            fib_radius_mod = (phi**(i+2) * v) % 1
            
            # Combine hourglass shape with Fibonacci modulation
            # 0.5 + 0.5 * factor gives a range from 0.5 to 1.0 to prevent zero radius
            r_i = cylinder_radius * (0.5 + 0.5 * hourglass_factor * fib_radius_mod)
            
            # Set x, y coordinates for this cylinder
            embeddings[v, 2*i] = r_i * np.cos(theta)      # x-coordinate
            embeddings[v, 2*i+1] = r_i * np.sin(theta)    # y-coordinate
    
    return embeddings

def create_riemannian_metric(num_cylinders=3, coupling_strength=0.1):
    """
    Create a Riemannian metric tensor for the multi-cylindrical manifold
    
    Args:
        num_cylinders: Number of cylinders
        coupling_strength: Strength of coupling between cylinders
        
    Returns:
        g: Metric tensor of shape [2*num_cylinders, 2*num_cylinders]
    """
    dim = 2 * num_cylinders
    g = np.eye(dim)
    
    # Add coupling between cylinders
    for i in range(num_cylinders):
        for j in range(i+1, num_cylinders):
            # Couple x coordinates
            g[2*i, 2*j] = coupling_strength
            g[2*j, 2*i] = coupling_strength
            
            # Couple y coordinates
            g[2*i+1, 2*j+1] = coupling_strength
            g[2*j+1, 2*i+1] = coupling_strength
    
    return g

def compute_geodesic_distance(embeddings, g):
    """
    Compute pairwise geodesic distances using the metric tensor
    
    Args:
        embeddings: Token embeddings [vocab_size, 2*num_cylinders]
        g: Metric tensor [2*num_cylinders, 2*num_cylinders]
        
    Returns:
        distances: Pairwise distances [vocab_size, vocab_size]
    """
    vocab_size = embeddings.shape[0]
    distances = np.zeros((vocab_size, vocab_size))
    
    for i in range(vocab_size):
        for j in range(vocab_size):
            # Compute difference vector
            diff = embeddings[i] - embeddings[j]
            
            # Compute geodesic distance
            distances[i, j] = np.sqrt(diff @ g @ diff.T)
    
    return distances

def visualize_hourglass_manifold(embeddings, num_cylinders=3, num_tokens=20, save_path=None):
    """
    Visualize the multi-cylindrical hourglass manifold
    
    Args:
        embeddings: Token embeddings [vocab_size, 2*num_cylinders]
        num_cylinders: Number of cylinders
        num_tokens: Number of tokens to visualize
        save_path: Path to save the visualization
    """
    # Create a figure for each cylinder pair (picking the most informative pairs)
    fig = plt.figure(figsize=(15, 15))
    
    # If we have 3 or more cylinders, show 3 different views
    pairs = [(0, 1), (0, 2), (1, 2)] if num_cylinders >= 3 else [(0, min(1, num_cylinders-1))]
    num_pairs = len(pairs)
    
    # Colors for tokens
    colors = plt.cm.viridis(np.linspace(0, 1, num_tokens))
    
    # Create a 3D projection showing all cylinders together
    ax_all = fig.add_subplot(2, 2, 1, projection='3d')
    ax_all.set_title("Multi-Cylindrical Hourglass Manifold (3D)")
    
    # For each pair of cylinders, create a 2D plot
    for p_idx, (c1, c2) in enumerate(pairs):
        ax = fig.add_subplot(2, 2, p_idx+2)
        ax.set_title(f"Hourglass Cylinders ({c1+1}, {c2+1})")
        
        # Plot reference circles
        ref_radius = 0.8 / np.sqrt(num_cylinders)
        circle1 = plt.Circle((0, 0), ref_radius, fill=False, color='black', linestyle='-', linewidth=1)
        ax.add_artist(circle1)
        
        # Plot min/max circles to show hourglass bounds
        min_circle = plt.Circle((0, 0), 0.5 * ref_radius, fill=False, color='gray', linestyle='--', linewidth=0.5)
        max_circle = plt.Circle((0, 0), 1.0 * ref_radius, fill=False, color='gray', linestyle='--', linewidth=0.5)
        ax.add_artist(min_circle)
        ax.add_artist(max_circle)
        
        # Plot tokens
        for i in range(min(num_tokens, embeddings.shape[0])):
            # Extract coordinates for this cylinder pair
            x1, y1 = embeddings[i, 2*c1], embeddings[i, 2*c1+1]
            x2, y2 = embeddings[i, 2*c2], embeddings[i, 2*c2+1]
            
            # Compute radius for visualization
            r1 = np.sqrt(x1**2 + y1**2)
            
            # Plot in 2D projection with size proportional to radius
            ax.scatter(x1, y1, color=colors[i], s=100 * r1 / ref_radius, alpha=0.7)
            
            # Connect with a small arrow if not the same cylinder
            if c1 != c2:
                ax.arrow(x1, y1, (x2-x1)/4, (y2-y1)/4, 
                         head_width=0.02, head_length=0.03, 
                         fc=colors[i], ec=colors[i], alpha=0.5)
                
            # Add token index
            ax.text(x1, y1, str(i), fontsize=8, ha='center', va='center')
        
        # Plot cylinder origin
        ax.plot(0, 0, 'k+', markersize=10)
        
        # Add labels and set aspect ratio
        ax.set_xlabel(f"Cylinder {c1+1} - X")
        ax.set_ylabel(f"Cylinder {c1+1} - Y")
        ax.grid(linestyle='--', alpha=0.3)
        ax.set_aspect('equal')
        
        # Set limits
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
    
    # For the 3D projection
    # Pick 3 dimensions to visualize (we can only show 3D)
    dims = [0, 2, 4] if num_cylinders >= 3 else [0, 1, 2 if num_cylinders >= 2 else 0]
    
    # Plot tokens in 3D space
    for i in range(min(num_tokens, embeddings.shape[0])):
        # Get coordinates from different cylinder dimensions
        x = embeddings[i, dims[0]]
        y = embeddings[i, dims[1]]
        z = embeddings[i, dims[2]]
        
        # Calculate radius for this point (distance from origin)
        radius = np.sqrt(x**2 + y**2 + z**2)
        
        # Plot in 3D with size proportional to radius
        ax_all.scatter(x, y, z, color=colors[i], s=100 * radius * 2, alpha=0.7)
        
        # Add token index
        ax_all.text(x, y, z, str(i), fontsize=8)
    
    # Add coordinate system
    draw_coordinate_axes(ax_all, length=0.5)
    
    # Plot hourglass surfaces as wireframes
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, 1, 10)
    
    for c_idx in range(min(3, num_cylinders)):
        # Base radius for this cylinder
        base_r = (c_idx + 1) / num_cylinders
        
        # Create hourglass shape for visualization
        for h_idx, height in enumerate(np.linspace(-0.5, 0.5, 5)):
            # Hourglass radius - narrower in the middle
            h_factor = 0.2 + 0.8 * abs(2 * abs(height) - 1)
            radius = base_r * (0.5 + 0.5 * h_factor)
            
            # Create circle at this height
            circle_x = radius * np.cos(u)
            circle_y = radius * np.sin(u)
            circle_z = height * np.ones_like(u)
            
            # Rotate the circle to match the cylinder orientation
            if c_idx == 0:  # X-Y plane
                ax_all.plot(circle_x, circle_y, circle_z, 'gray', alpha=0.2)
            elif c_idx == 1:  # X-Z plane
                ax_all.plot(circle_x, circle_z, circle_y, 'gray', alpha=0.2)
            elif c_idx == 2:  # Y-Z plane
                ax_all.plot(circle_z, circle_x, circle_y, 'gray', alpha=0.2)
    
    # Set labels
    ax_all.set_xlabel(f"Dimension {dims[0]+1}")
    ax_all.set_ylabel(f"Dimension {dims[1]+1}")
    ax_all.set_zlabel(f"Dimension {dims[2]+1}")
    
    # Add distance matrix visualization
    ax_dist = fig.add_subplot(2, 2, 4)
    
    # Compute metric tensor
    g = create_riemannian_metric(num_cylinders=num_cylinders)
    
    # Compute geodesic distances
    distances = compute_geodesic_distance(embeddings[:num_tokens], g)
    
    # Visualize distance matrix
    im = ax_dist.imshow(distances, cmap='viridis')
    ax_dist.set_title("Geodesic Distances (Riemannian Metric)")
    ax_dist.set_xlabel("Token Index")
    ax_dist.set_ylabel("Token Index")
    plt.colorbar(im, ax=ax_dist)
    
    # Add grid
    for i in range(num_tokens):
        ax_dist.axhline(i-0.5, color='white', linewidth=0.5)
        ax_dist.axvline(i-0.5, color='white', linewidth=0.5)
    
    # Add axis annotations to show how radius varies with token index
    ax_radius = ax_dist.twinx()
    ax_radius.set_ylabel("Radius Factor", color='red')
    token_indices = np.arange(num_tokens)
    
    # Calculate and plot radius factors for first cylinder
    radius_factors = []
    for i in range(num_tokens):
        x, y = embeddings[i, 0], embeddings[i, 1]
        radius_factors.append(np.sqrt(x**2 + y**2))
    
    ax_radius.plot(token_indices, radius_factors, 'r-', alpha=0.7)
    ax_radius.tick_params(axis='y', labelcolor='red')
    
    # Set title and layout
    plt.suptitle(f"Hourglass Multi-Cylindrical Manifold (m={num_cylinders})", fontsize=16)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

# Simpler approach to draw axes without custom Arrow3D class
def draw_coordinate_axes(ax, length=0.5):
    """Draw coordinate axes in a 3D plot"""
    ax.plot([0, length], [0, 0], [0, 0], 'r-', linewidth=2, label='X-axis')
    ax.plot([0, 0], [0, length], [0, 0], 'g-', linewidth=2, label='Y-axis')
    ax.plot([0, 0], [0, 0], [0, length], 'b-', linewidth=2, label='Z-axis')

def main():
    """Main function to create and visualize hourglass cylindrical manifold"""
    print("=" * 80)
    print("HOURGLASS MULTI-CYLINDRICAL MANIFOLD VISUALIZATION")
    print("=" * 80)
    
    try:
        # Parameters
        vocab_size = 50
        num_cylinders = 3
        phi = 1.618034  # Golden ratio
        
        print(f"Generating embeddings for {vocab_size} tokens on a {num_cylinders}-cylinder hourglass manifold...")
        
        # Generate embeddings with hourglass shape
        embeddings = generate_hourglass_embedding(
            vocab_size=vocab_size,
            num_cylinders=num_cylinders,
            phi=phi
        )
        
        print(f"Embedding shape: {embeddings.shape}")
        
        # Visualize
        print("Creating visualization...")
        visualize_hourglass_manifold(
            embeddings=embeddings,
            num_cylinders=num_cylinders,
            num_tokens=16,
            save_path="hourglass_manifold.png"
        )
        
        print("Visualization saved to: hourglass_manifold.png")
        
        # Try different numbers of cylinders if the first one works
        print("\nCreating comparison with different numbers of cylinders...")
        
        for m in [2, 4]:
            print(f"Generating for m={m} cylinders...")
            
            # Generate embeddings with hourglass shape
            embeddings_m = generate_hourglass_embedding(
                vocab_size=vocab_size,
                num_cylinders=m,
                phi=phi
            )
            
            # Visualize
            visualize_hourglass_manifold(
                embeddings=embeddings_m,
                num_cylinders=m,
                num_tokens=16,
                save_path=f"hourglass_manifold_m{m}.png"
            )
            
            print(f"Saved to: hourglass_manifold_m{m}.png")
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        
        # Simplified version without 3D plotting in case of errors
        print("\nFalling back to 2D-only visualization...")
        
        # Try with only 2 cylinders for simplicity
        embeddings = generate_hourglass_embedding(
            vocab_size=50,
            num_cylinders=2,
            phi=phi
        )
        
        # Create simplified 2D plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot reference circles
        circle1 = plt.Circle((0, 0), 0.5, fill=False, color='black', linestyle='-', linewidth=1)
        ax.add_artist(circle1)
        
        # Plot tokens for first cylinder only
        for i in range(16):
            x, y = embeddings[i, 0], embeddings[i, 1]
            r = np.sqrt(x**2 + y**2)
            ax.scatter(x, y, color=plt.cm.viridis(i/16), s=100*r*2, alpha=0.7)
            ax.text(x, y, str(i), fontsize=8)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title("Hourglass Manifold (First Cylinder)")
        plt.savefig("hourglass_2d_only.png")
        print("2D visualization saved to: hourglass_2d_only.png")

if __name__ == "__main__":
    main()

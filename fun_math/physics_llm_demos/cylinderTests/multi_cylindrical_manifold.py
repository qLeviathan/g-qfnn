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

def generate_multicylindrical_embedding(vocab_size, num_cylinders=3, phi=1.618034):
    """
    Generate token embeddings in a multi-cylindrical manifold
    
    Args:
        vocab_size: Number of tokens
        num_cylinders: Number of coupled cylinders (m)
        phi: Golden ratio for phase distribution
        
    Returns:
        embeddings: Token embeddings of shape [vocab_size, 2*num_cylinders]
    """
    embeddings = np.zeros((vocab_size, 2 * num_cylinders))
    
    for v in range(vocab_size):
        for i in range(num_cylinders):
            # Generate phase angles using different powers of phi
            theta = 2 * np.pi * ((phi**(i+1) * v) % 1)
            
            # Generate radius with controlled variation
            alpha = 0.1  # Controls radius variation
            r = (1.0 / np.sqrt(num_cylinders)) * (1 + alpha * np.sin(2 * np.pi * i / num_cylinders))
            
            # Set x, y coordinates for this cylinder
            embeddings[v, 2*i] = r * np.cos(theta)      # x-coordinate
            embeddings[v, 2*i+1] = r * np.sin(theta)    # y-coordinate
    
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

def visualize_multicylindrical_manifold(embeddings, num_cylinders=3, num_tokens=20, save_path=None):
    """
    Visualize the multi-cylindrical manifold
    
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
    ax_all.set_title("Multi-Cylindrical Manifold (3D Projection)")
    
    # For each pair of cylinders, create a 2D plot
    for p_idx, (c1, c2) in enumerate(pairs):
        ax = fig.add_subplot(2, 2, p_idx+2)
        ax.set_title(f"Cylindrical Pair ({c1+1}, {c2+1})")
        
        # Plot unit circles for each cylinder
        circle1 = plt.Circle((0, 0), 1/np.sqrt(num_cylinders), fill=False, color='black', linestyle='-', linewidth=1)
        ax.add_artist(circle1)
        
        # Plot tokens
        for i in range(min(num_tokens, embeddings.shape[0])):
            # Extract coordinates for this cylinder pair
            x1, y1 = embeddings[i, 2*c1], embeddings[i, 2*c1+1]
            x2, y2 = embeddings[i, 2*c2], embeddings[i, 2*c2+1]
            
            # Plot in 2D projection
            ax.scatter(x1, y1, color=colors[i], s=100, alpha=0.7)
            
            # Connect with a small arrow if not the same cylinder
            if c1 != c2:
                ax.arrow(x1, y1, (x2-x1)/4, (y2-y1)/4, 
                         head_width=0.02, head_length=0.03, 
                         fc=colors[i], ec=colors[i], alpha=0.5)
                
            # Add token index
            ax.text(x1, y1, str(i), fontsize=8, ha='center', va='center')
        
        # Plot cylinder markers
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
        
        # Plot in 3D
        ax_all.scatter(x, y, z, color=colors[i], s=100, alpha=0.7)
        
        # Add token index
        ax_all.text(x, y, z, str(i), fontsize=8)
    
    # Add coordinate system
    origin = [0, 0, 0]
    x_axis = Arrow3D([0, 0.5], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
    y_axis = Arrow3D([0, 0], [0, 0.5], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="g")
    z_axis = Arrow3D([0, 0], [0, 0], [0, 0.5], mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
    
    ax_all.add_artist(x_axis)
    ax_all.add_artist(y_axis)
    ax_all.add_artist(z_axis)
    
    # Plot cylindrical surfaces (small portions for visualization)
    u = np.linspace(0, 2*np.pi, 30)
    h = np.linspace(-0.5, 0.5, 10)
    
    for c_idx in range(min(3, num_cylinders)):
        # Radius for this cylinder
        r = 1.0 / np.sqrt(num_cylinders)
        
        # Create a cylindrical surface
        if c_idx == 0:  # X-Y cylinder
            x = r * np.outer(np.cos(u), np.ones_like(h))
            y = r * np.outer(np.sin(u), np.ones_like(h))
            z = np.outer(np.ones_like(u), h)
            ax_all.plot_surface(x, y, z, alpha=0.1, color='gray')
        elif c_idx == 1:  # X-Z cylinder
            x = r * np.outer(np.cos(u), np.ones_like(h))
            y = np.outer(np.ones_like(u), h)
            z = r * np.outer(np.sin(u), np.ones_like(h))
            ax_all.plot_surface(x, y, z, alpha=0.1, color='gray')
        elif c_idx == 2:  # Y-Z cylinder
            x = np.outer(np.ones_like(u), h)
            y = r * np.outer(np.cos(u), np.ones_like(h))
            z = r * np.outer(np.sin(u), np.ones_like(h))
            ax_all.plot_surface(x, y, z, alpha=0.1, color='gray')
    
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
    
    # Set title and layout
    plt.suptitle(f"Multi-Cylindrical Manifold Visualization (m={num_cylinders})", fontsize=16)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def main():
    """Main function to create and visualize multi-cylindrical manifold"""
    print("=" * 80)
    print("MULTI-CYLINDRICAL FIELD MANIFOLD VISUALIZATION")
    print("=" * 80)
    
    # Parameters
    vocab_size = 100
    num_cylinders = 3
    phi = 1.618034  # Golden ratio
    
    print(f"Generating embeddings for {vocab_size} tokens on a {num_cylinders}-cylindrical manifold...")
    
    # Generate embeddings
    embeddings = generate_multicylindrical_embedding(
        vocab_size=vocab_size,
        num_cylinders=num_cylinders,
        phi=phi
    )
    
    print(f"Embedding shape: {embeddings.shape}")
    
    # Visualize
    print("Creating visualization...")
    visualize_multicylindrical_manifold(
        embeddings=embeddings,
        num_cylinders=num_cylinders,
        num_tokens=20,
        save_path="multi_cylindrical_manifold.png"
    )
    
    print("Visualization saved to: multi_cylindrical_manifold.png")
    
    # Also create a comparison between different numbers of cylinders
    print("\nCreating comparison visualizations for different numbers of cylinders...")
    
    for m in [2, 3, 4]:
        print(f"Generating for m={m} cylinders...")
        
        # Generate embeddings
        embeddings_m = generate_multicylindrical_embedding(
            vocab_size=vocab_size,
            num_cylinders=m,
            phi=phi
        )
        
        # Visualize
        visualize_multicylindrical_manifold(
            embeddings=embeddings_m,
            num_cylinders=m,
            num_tokens=20,
            save_path=f"multi_cylindrical_manifold_m{m}.png"
        )
        
        print(f"Saved to: multi_cylindrical_manifold_m{m}.png")

if __name__ == "__main__":
    main()

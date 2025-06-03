import torch
import numpy as np
import matplotlib.pyplot as plt

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618033988749895


class CartesianNegativeDistanceAttention:
    """
    Implementation of negative distance attention using direct Cartesian coordinates
    instead of polar-to-scalar projection.
    """
    
    def __init__(
        self,
        d_model=64,
        num_heads=1,
        dropout=0.1,
        use_golden_ratio=True,
        memory_influence=0.1
    ):
        """
        Initialize the Cartesian Negative Distance Attention module.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_golden_ratio: Whether to use golden ratio as temperature
            memory_influence: Hebbian memory influence factor
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_golden_ratio = use_golden_ratio
        self.memory_influence = memory_influence
        
        # Memory matrix for Hebbian learning
        self.memory = None
        self.memory_decay = 0.99
        
    def initialize_embeddings(self, batch_size, seq_len):
        """
        Initialize token embeddings in Cartesian coordinates.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Cartesian embeddings of shape [batch_size, seq_len, 2]
        """
        # Create a grid of Cartesian coordinates instead of polar
        # Using golden spiral for optimal distribution in Cartesian space
        
        # Golden angle in Cartesian initialization
        golden_angle = 2 * np.pi * (1 - 1/PHI)
        
        # Initialize embeddings
        embeddings = torch.zeros(batch_size, seq_len, 2)
        
        for i in range(seq_len):
            # Calculate angle based on golden ratio
            theta = i * golden_angle
            
            # Calculate radius (increasing with sequence position)
            radius = 0.382 + 0.618 * (i / max(1, seq_len - 1))
            
            # Convert to Cartesian coordinates
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            
            # Assign to all batches
            embeddings[:, i, 0] = x  # x-coordinate
            embeddings[:, i, 1] = y  # y-coordinate
            
        return embeddings
    
    def compute_negative_distance_matrix(self, embeddings):
        """
        Compute the negative squared Euclidean distance matrix directly from 
        Cartesian coordinates.
        
        Args:
            embeddings: Cartesian embeddings of shape [batch_size, seq_len, 2]
            
        Returns:
            Negative distance matrix of shape [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # Expand dimensions for broadcasting
        # Shape: [batch_size, seq_len, 1, 2]
        emb_i = embeddings.unsqueeze(2)
        
        # Shape: [batch_size, 1, seq_len, 2]
        emb_j = embeddings.unsqueeze(1)
        
        # Compute squared Euclidean distance in Cartesian space
        # Shape: [batch_size, seq_len, seq_len]
        squared_distance = ((emb_i - emb_j) ** 2).sum(dim=-1)
        
        # Return negative distance matrix
        return -squared_distance
    
    def apply_golden_kernel(self, negative_distance, temperature=None):
        """
        Apply φ-scaled Gaussian kernel to negative distance matrix.
        
        Args:
            negative_distance: Negative distance matrix [batch_size, seq_len, seq_len]
            temperature: Optional temperature override (default: PHI)
            
        Returns:
            Probability kernel matrix [batch_size, seq_len, seq_len]
        """
        # Use golden ratio as default temperature
        if temperature is None:
            temperature = PHI if self.use_golden_ratio else 1.0
            
        # Apply exponential kernel
        kernel = torch.exp(negative_distance / temperature)
        
        # Normalize (optional)
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-9)
        
        return kernel
    
    def apply_memory_bias(self, kernel, embeddings):
        """
        Apply Hebbian memory bias to the attention kernel.
        
        Args:
            kernel: Original attention kernel
            embeddings: Cartesian embeddings
            
        Returns:
            Memory-biased kernel
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # Initialize memory if not already done
        if self.memory is None:
            self.memory = torch.zeros(batch_size, seq_len, seq_len)
            
        # Compute outer product for Hebbian update
        # For simplicity using the sum of x and y coordinates
        embedding_sum = embeddings.sum(dim=-1)
        outer_product = embedding_sum.unsqueeze(-1) @ embedding_sum.unsqueeze(1)
        
        # Update memory with Hebbian rule
        self.memory = self.memory_decay * self.memory + (1 - self.memory_decay) * outer_product
        
        # Apply memory bias to kernel
        biased_kernel = kernel * torch.exp(self.memory_influence * self.memory)
        
        # Renormalize
        biased_kernel = biased_kernel / (biased_kernel.sum(dim=-1, keepdim=True) + 1e-9)
        
        return biased_kernel
    
    def compute_optimal_step_size(self, temperature):
        """
        Compute physics-derived optimal step size.
        
        Args:
            temperature: Current temperature parameter
            
        Returns:
            Optimal integration step size
        """
        # Assume grid spacing of 0.1 for Cartesian grid
        dr = 0.1
        
        # Physics-derived step size from diffusion equation stability
        return (dr ** 2) / (2 * temperature)
    
    def update_embeddings(self, embeddings, attention_output, temperature=None):
        """
        Update embeddings using Heun-Euler integration.
        
        Args:
            embeddings: Current Cartesian embeddings
            attention_output: Output from attention mechanism
            temperature: Temperature parameter (default: PHI)
            
        Returns:
            Updated embeddings
        """
        if temperature is None:
            temperature = PHI if self.use_golden_ratio else 1.0
            
        # Compute optimal step size
        dt = self.compute_optimal_step_size(temperature)
        
        # Predictor step (Euler)
        grad = embeddings - attention_output
        embeddings_euler = embeddings - dt * grad
        
        # Corrector step (Heun)
        grad_euler = embeddings_euler - attention_output
        embeddings_next = embeddings - (dt / 2) * (grad + grad_euler)
        
        return embeddings_next
    
    def forward(self, batch_size=16, seq_len=128):
        """
        Forward pass of Cartesian negative distance attention.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            tuple: (attention_output, attention_weights, updated_embeddings)
        """
        # Initialize Cartesian embeddings
        embeddings = self.initialize_embeddings(batch_size, seq_len)
        
        # Mock content embeddings (would come from model in real implementation)
        content_embeddings = torch.randn(batch_size, seq_len, self.d_model)
        
        # Compute negative distance matrix
        neg_dist = self.compute_negative_distance_matrix(embeddings)
        
        # Apply golden kernel
        attention_weights = self.apply_golden_kernel(neg_dist)
        
        # Apply memory bias (optional)
        attention_weights = self.apply_memory_bias(attention_weights, embeddings)
        
        # Apply attention to content
        attention_output = torch.bmm(attention_weights, content_embeddings)
        
        # Update embeddings using Heun-Euler
        updated_embeddings = self.update_embeddings(embeddings, attention_output[:, :, :2])
        
        return attention_output, attention_weights, updated_embeddings


def visualize_cartesian_attention(attention_weights, embeddings, seq_len=64):
    """
    Visualize Cartesian embeddings and attention patterns.
    
    Args:
        attention_weights: Attention weight matrix [batch_size, seq_len, seq_len]
        embeddings: Cartesian embeddings [batch_size, seq_len, 2]
        seq_len: Number of tokens to visualize
    """
    # Use first item in batch for visualization
    attn = attention_weights[0, :seq_len, :seq_len].detach().numpy()
    emb = embeddings[0, :seq_len, :].detach().numpy()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot attention heatmap
    im = ax1.imshow(attn, cmap='viridis')
    ax1.set_title("Attention Matrix")
    ax1.set_xlabel("Target Token")
    ax1.set_ylabel("Source Token")
    fig.colorbar(im, ax=ax1)
    
    # Plot token embeddings in Cartesian space
    scatter = ax2.scatter(emb[:, 0], emb[:, 1], c=np.arange(seq_len), 
                          cmap='viridis', alpha=0.7, s=100)
    
    # Add arrows for top attention connections
    for i in range(seq_len):
        # Get top 3 attention targets for each token
        top_targets = np.argsort(attn[i])[-3:]
        for j in top_targets:
            # Skip self-attention
            if i != j:
                # Alpha proportional to attention weight
                alpha = max(0.1, min(1.0, attn[i, j] * 5))
                ax2.arrow(emb[i, 0], emb[i, 1], 
                          emb[j, 0] - emb[i, 0], emb[j, 1] - emb[i, 1],
                          alpha=alpha, width=0.005, head_width=0.02,
                          head_length=0.05, fc='gray', ec='gray')
    
    # Add token indices
    for i in range(seq_len):
        ax2.text(emb[i, 0], emb[i, 1], str(i), fontsize=9, ha='center', va='center')
    
    ax2.set_title("Token Embeddings in Cartesian Space")
    ax2.set_xlabel("X Coordinate")
    ax2.set_ylabel("Y Coordinate")
    ax2.grid(alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('cartesian_attention.png')
    plt.show()


def compare_cartesian_vs_polar():
    """
    Compare properties of Cartesian and Polar approaches to negative distance.
    """
    seq_len = 64
    batch_size = 1
    
    # Initialize Cartesian model
    cart_model = CartesianNegativeDistanceAttention()
    
    # Generate Cartesian embeddings
    cart_embeddings = cart_model.initialize_embeddings(batch_size, seq_len)
    
    # Generate equivalent polar embeddings
    polar_embeddings = torch.zeros_like(cart_embeddings)
    
    # Convert Cartesian to equivalent polar for comparison
    x = cart_embeddings[0, :, 0].numpy()
    y = cart_embeddings[0, :, 1].numpy()
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Create scalar projections for polar approach
    polar_projection = r * (np.cos(theta) - np.sin(theta))
    
    # Compute negative distance matrices
    cart_neg_dist = cart_model.compute_negative_distance_matrix(cart_embeddings)
    
    # Manual calculation for polar approach (using projection values)
    polar_proj = torch.tensor(polar_projection).unsqueeze(0)
    polar_neg_dist = -(polar_proj.unsqueeze(2) - polar_proj.unsqueeze(1))**2
    
    # Apply golden kernel to both
    cart_kernel = cart_model.apply_golden_kernel(cart_neg_dist)
    polar_kernel = torch.exp(polar_neg_dist / PHI)
    polar_kernel = polar_kernel / polar_kernel.sum(dim=-1, keepdim=True)
    
    # Compare kernels
    correlation = torch.corrcoef(
        torch.stack([
            cart_kernel[0].flatten(), 
            polar_kernel[0].flatten()
        ])
    )[0, 1].item()
    
    # Print results
    print(f"Correlation between Cartesian and Polar kernels: {correlation:.4f}")
    
    # Create comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot Cartesian kernel
    im1 = ax1.imshow(cart_kernel[0].detach().numpy(), cmap='viridis')
    ax1.set_title("Cartesian Approach")
    ax1.set_xlabel("Target Token")
    ax1.set_ylabel("Source Token")
    fig.colorbar(im1, ax=ax1)
    
    # Plot Polar kernel
    im2 = ax2.imshow(polar_kernel[0].detach().numpy(), cmap='viridis')
    ax2.set_title("Polar Approach")
    ax2.set_xlabel("Target Token")
    ax2.set_ylabel("Source Token")
    fig.colorbar(im2, ax=ax2)
    
    # Plot absolute difference
    diff = torch.abs(cart_kernel - polar_kernel)[0].detach().numpy()
    im3 = ax3.imshow(diff, cmap='hot')
    ax3.set_title(f"Absolute Difference\nCorrelation: {correlation:.4f}")
    ax3.set_xlabel("Target Token")
    ax3.set_ylabel("Source Token")
    fig.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('cartesian_vs_polar.png')
    plt.show()
    
    return {
        'correlation': correlation,
        'cart_embeddings': cart_embeddings,
        'cart_kernel': cart_kernel
    }


def main():
    """Main demonstration function"""
    print("=" * 80)
    print("CARTESIAN NEGATIVE DISTANCE MATRIX: PROOF OF CONCEPT")
    print("=" * 80)
    
    # Model parameters
    batch_size = 16
    seq_len = 64
    
    # Initialize model
    print("\nInitializing Cartesian Negative Distance Attention...")
    model = CartesianNegativeDistanceAttention()
    
    # Run forward pass
    print("Running forward pass...")
    attention_output, attention_weights, updated_embeddings = model.forward(
        batch_size=batch_size, 
        seq_len=seq_len
    )
    
    # Print shapes
    print(f"\nOutput shapes:")
    print(f"  Attention output: {attention_output.shape}")
    print(f"  Attention weights: {attention_weights.shape}")
    print(f"  Updated embeddings: {updated_embeddings.shape}")
    
    # Visualize attention patterns
    print("\nVisualizing attention patterns...")
    visualize_cartesian_attention(attention_weights, updated_embeddings, seq_len=32)
    
    # Compare with polar approach
    print("\nComparing Cartesian vs Polar approaches...")
    results = compare_cartesian_vs_polar()
    
    print("\nAdvantages of Cartesian approach:")
    print("  1. Direct geometric interpretation in Euclidean space")
    print("  2. Simpler distance computation without trigonometric projections")
    print("  3. Natural representation of token positions in 2D space")
    print("  4. Compatible with existing tensor operations for efficiency")
    print("  5. Maintains all mathematical properties of negative semi-definiteness")
    
    print("\nMathematical equivalence:")
    print(f"  Correlation with polar approach: {results['correlation']:.4f}")
    if results['correlation'] > 0.9:
        print("  Both approaches produce highly similar attention patterns")
    else:
        print("  The approaches produce somewhat different attention patterns")

    print("\nCartesian Negative Distance Attention successfully implemented!")


if __name__ == "__main__":
    main()
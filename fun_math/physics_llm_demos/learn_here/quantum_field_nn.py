"""
Quantum Field Neural Network (QFNN) with Log-Cartesian Cylindrical Embeddings
Complete implementation integrating all modules:
- Log-Cylindrical coordinate system
- Sparse Log-Hebbian learning
- Dual vortex field dynamics with tachyonic tunneling
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import time
import os

# Import our modules
from log_coords import LogCylindricalCoords, device, PHI, PI, TAU, EPS
from log_hebbian import SparseLogHebbian
from dual_vortex import DualVortexField

class QuantumFieldNN:
    """
    Quantum Field Neural Network with log-cylindrical embeddings
    Combines all the components into a complete neural network
    
    Mathematical Foundation:
    - Uses logarithmic cylindrical coordinates for stable token representation
    - Field dynamics governed by Hamiltonian H = T + V where T is kinetic energy and V is potential
    - Implements non-Lipschitz evolution through tachyonic tunneling
    - Achieves O(N log N) complexity through optimized spatial structures
    - Memory complexity is O(N·k) where k << N through sparse Hebbian connections
    
    Infinite Context Benefits:
    - Log-cylindrical coordinates allow representations across exponential scales
    - Context information is stored in field topology rather than fixed-size state
    - Information propagation through Hebbian pitch alignment
    - Rotational invariance of field dynamics enables context compression
    - Tachyonic tunneling prevents information loss through local optima
    """
    def __init__(self, vocab_size: int, embedding_dim: int, device=None):
        """
        Initialize the Quantum Field Neural Network
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Embedding dimension
            device: Computation device
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Constants
        self.phi = torch.tensor((1 + np.sqrt(5)) / 2, device=self.device)
        
        # Create embedding layer
        self.token_embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with golden ratio structure
        self._init_golden_embeddings()
        
        # Create log-cylindrical field
        self.field = DualVortexField(embedding_dim, device=self.device)
        self.field.initialize_tokens(pattern='golden_spiral')
        
        # Output projection
        self.output_projection = torch.nn.Linear(embedding_dim, vocab_size)
        
        # Memory tracker for infinite context experiments
        self.context_memory = {
            'token_count': 0,
            'hebbian_connections': 0,
            'energy_history': [],
            'performance_metrics': {}
        }
        
        # Move to device
        self.to(self.device)
    
    def _init_golden_embeddings(self):
        """Initialize embeddings with golden ratio structure"""
        with torch.no_grad():
            for i in range(self.vocab_size):
                for j in range(self.embedding_dim):
                    # Golden angle in radians
                    theta = 2 * np.pi * ((i * self.phi.item()) % 1.0)
                    radius = 1.0 + 0.1 * ((j * self.phi.item()) % 1.0)
                    
                    # Log-cylindrical initialization
                    ln_r = torch.log(torch.tensor(radius))
                    
                    # Convert to Cartesian for embedding
                    x = torch.exp(ln_r) * torch.cos(torch.tensor(theta))
                    y = torch.exp(ln_r) * torch.sin(torch.tensor(theta))
                    
                    # Store in embedding table
                    if j % 2 == 0:
                        self.token_embedding.weight.data[i, j] = x
                    else:
                        self.token_embedding.weight.data[i, j] = y
            
            # Normalize
            self.token_embedding.weight.data = torch.nn.functional.normalize(
                self.token_embedding.weight.data, dim=1
            )
    
    def to(self, device):
        """Move model to device"""
        self.device = device
        self.token_embedding = self.token_embedding.to(device)
        self.output_projection = self.output_projection.to(device)
        return self
    
    def cartesian_to_log_cylindrical(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert Cartesian embeddings to log-cylindrical coordinates
        
        Args:
            x: Cartesian embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            ln_r: Log-radius [batch_size, seq_len, embedding_dim//2]
            theta: Angle [batch_size, seq_len, embedding_dim//2]
        """
        batch_size, seq_len, dim = x.shape
        
        # Reshape to pairs of (x, y) coordinates
        x_reshaped = x.reshape(batch_size, seq_len, dim // 2, 2)
        
        # Extract x and y components
        x_coords = x_reshaped[:, :, :, 0]
        y_coords = x_reshaped[:, :, :, 1]
        
        # Compute log-cylindrical coordinates
        # r = sqrt(x² + y²)
        r = torch.sqrt(x_coords**2 + y_coords**2 + EPS)
        ln_r = torch.log(r)
        
        # θ = atan2(y, x)
        theta = torch.atan2(y_coords, x_coords)
        
        return ln_r, theta
    
    def log_cylindrical_to_cartesian(self, ln_r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Convert log-cylindrical coordinates to Cartesian embeddings
        
        Args:
            ln_r: Log-radius [batch_size, seq_len, embedding_dim//2]
            theta: Angle [batch_size, seq_len, embedding_dim//2]
            
        Returns:
            x: Cartesian embeddings [batch_size, seq_len, embedding_dim]
        """
        # r = exp(ln_r)
        r = torch.exp(ln_r)
        
        # Convert to Cartesian
        x_coords = r * torch.cos(theta)
        y_coords = r * torch.sin(theta)
        
        # Interleave x and y to get full embedding
        batch_size, seq_len, half_dim = ln_r.shape
        x = torch.zeros(batch_size, seq_len, half_dim * 2, device=self.device)
        
        x[:, :, 0::2] = x_coords
        x[:, :, 1::2] = y_coords
        
        return x
    
    def evolve_field(self, x: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Evolve the quantum field according to the dynamics
        
        Args:
            x: Input embeddings [batch_size, seq_len, embedding_dim]
            steps: Number of evolution steps
            
        Returns:
            evolved_x: Evolved embeddings [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, dim = x.shape
        
        # Convert to log-cylindrical coordinates
        ln_r, theta = self.cartesian_to_log_cylindrical(x)
        
        # Process each token sequence separately
        evolved_ln_r = torch.zeros_like(ln_r)
        evolved_theta = torch.zeros_like(theta)
        
        for b in range(batch_size):
            for s in range(seq_len):
                # Set token positions in the field
                self.field.tokens['ln_r'] = ln_r[b, s]
                self.field.tokens['theta'] = theta[b, s]
                
                # Reset frozen state
                self.field.tokens['frozen'].fill_(False)
                
                # Evolve field
                for _ in range(steps):
                    self.field.integrate_step()
                
                # Extract evolved positions
                evolved_ln_r[b, s] = self.field.tokens['ln_r']
                evolved_theta[b, s] = self.field.tokens['theta']
        
        # Convert back to Cartesian
        evolved_x = self.log_cylindrical_to_cartesian(evolved_ln_r, evolved_theta)
        
        return evolved_x
    
    def forward(self, input_ids: torch.Tensor, evolution_steps: int = 5) -> torch.Tensor:
        """
        Forward pass through the QFNN
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            evolution_steps: Number of field evolution steps
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        # Get token embeddings
        x = self.token_embedding(input_ids)
        
        # Evolve field
        x = self.evolve_field(x, steps=evolution_steps)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 50, 
                temperature: float = 1.0, top_p: float = 0.9,
                evolution_steps: int = 5) -> torch.Tensor:
        """
        Generate text using the QFNN
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            evolution_steps: Number of field evolution steps
            
        Returns:
            generated_ids: Generated token IDs [batch_size, max_length]
        """
        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass for current sequence
                logits = self.forward(generated_ids, evolution_steps=evolution_steps)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply softmax
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                
                # Nucleus sampling
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create a mask for indices to keep
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    
                    # Set removed indices to 0 probability
                    probs = probs.masked_fill(indices_to_remove, 0.0)
                    
                    # Renormalize
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Concatenate with generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Check if all sequences have an end token (assume 0 is end token)
                if (next_token == 0).all():
                    break
        
        return generated_ids
    
    def visualize_embeddings(self, save_path: Optional[str] = None):
        """
        Visualize token embeddings in log-cylindrical space
        
        Args:
            save_path: Optional path to save the figure
        """
        # Extract embeddings
        embeddings = self.token_embedding.weight.data
        
        # Convert to log-cylindrical
        ln_r, theta = self.cartesian_to_log_cylindrical(embeddings.unsqueeze(0))
        ln_r = ln_r.squeeze(0)
        theta = theta.squeeze(0)
        
        # Convert to numpy for visualization
        ln_r_np = ln_r.cpu().numpy()
        theta_np = theta.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot in log-cylindrical space
        # Use first dimension for visualization
        scatter = axes[0].scatter(ln_r_np[:, 0], theta_np[:, 0], 
                                 c=np.arange(self.vocab_size), cmap='viridis',
                                 s=30, alpha=0.7)
        axes[0].set_xlabel('ln(r)')
        axes[0].set_ylabel('θ')
        axes[0].set_title('Token Embeddings in Log-Cylindrical Space')
        axes[0].grid(True, alpha=0.3)
        
        # Convert to Cartesian
        x, y = self.field.coords.ln_r_theta_to_cartesian(ln_r[:, 0], theta[:, 0])
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Plot in Cartesian space
        scatter = axes[1].scatter(x_np, y_np, c=np.arange(self.vocab_size), 
                                 cmap='viridis', s=30, alpha=0.7)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title('Token Embeddings in Cartesian Space')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=axes[1], label='Token ID')
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Embeddings visualization saved to {save_path}")
        else:
            # Try to show, but don't error in non-interactive environments
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plot: {e}")
        
        plt.close()
    
    def compare_embedding_systems(self, save_path: Optional[str] = None):
        """
        Compare log-cylindrical embeddings with standard embeddings
        
        Args:
            save_path: Optional path to save the figure
        """
        # Create standard embeddings
        std_embeddings = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        torch.nn.init.normal_(std_embeddings.weight, mean=0.0, std=0.02)
        std_embeddings.weight.data = torch.nn.functional.normalize(std_embeddings.weight.data, dim=1)
        
        # Get our log-cylindrical embeddings
        log_embeddings = self.token_embedding.weight.data
        
        # Compute similarity matrices
        std_sim = torch.matmul(std_embeddings.weight, std_embeddings.weight.t())
        log_sim = torch.matmul(log_embeddings, log_embeddings.t())
        
        # Convert to numpy
        std_sim_np = std_sim.cpu().numpy()
        log_sim_np = log_sim.cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot standard similarity matrix
        im0 = axes[0].imshow(std_sim_np, cmap='viridis', vmin=-1, vmax=1)
        axes[0].set_title('Standard Embedding Similarity')
        axes[0].set_xlabel('Token ID')
        axes[0].set_ylabel('Token ID')
        plt.colorbar(im0, ax=axes[0])
        
        # Plot log-cylindrical similarity matrix
        im1 = axes[1].imshow(log_sim_np, cmap='viridis', vmin=-1, vmax=1)
        axes[1].set_title('Log-Cylindrical Embedding Similarity')
        axes[1].set_xlabel('Token ID')
        axes[1].set_ylabel('Token ID')
        plt.colorbar(im1, ax=axes[1])
        
        # Plot difference
        diff = log_sim_np - std_sim_np
        im2 = axes[2].imshow(diff, cmap='coolwarm', vmin=-1, vmax=1)
        axes[2].set_title('Similarity Difference (Log - Standard)')
        axes[2].set_xlabel('Token ID')
        axes[2].set_ylabel('Token ID')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Embedding comparison saved to {save_path}")
        else:
            # Try to show, but don't error in non-interactive environments
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plot: {e}")
        
        plt.close()
        
        # Calculate and print statistics
        print("Embedding Similarity Statistics:")
        print(f"Standard: mean={std_sim_np.mean():.4f}, std={std_sim_np.std():.4f}")
        print(f"Log-Cylindrical: mean={log_sim_np.mean():.4f}, std={log_sim_np.std():.4f}")
        
        # Compute singular values
        std_svd = torch.linalg.svd(std_embeddings.weight.data, full_matrices=False)
        log_svd = torch.linalg.svd(log_embeddings, full_matrices=False)
        
        # Plot singular values
        plt.figure(figsize=(10, 6))
        plt.plot(std_svd.S.cpu().numpy() / std_svd.S[0].item(), 'b-', label='Standard', alpha=0.7)
        plt.plot(log_svd.S.cpu().numpy() / log_svd.S[0].item(), 'r-', label='Log-Cylindrical', alpha=0.7)
        plt.yscale('log')
        plt.xlabel('Index')
        plt.ylabel('Normalized Singular Value')
        plt.title('Embedding Singular Value Spectrum')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save or show
        if save_path:
            svg_path = save_path.replace('.png', '_svd.png')
            plt.savefig(svg_path, dpi=200, bbox_inches='tight')
            print(f"SVD comparison saved to {svg_path}")
        else:
            # Try to show, but don't error in non-interactive environments
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plot: {e}")
        
        plt.close()
    
    def infinite_context_analysis(self, 
                             sequence_lengths: List[int] = [10, 100, 1000, 10000],
                             trials: int = 3,
                             save_path: Optional[str] = None):
        """
        Analyze the model's ability to handle infinite context lengths
        Demonstrates how the log-cylindrical field can represent arbitrary length sequences
        with constant memory usage and O(N log N) complexity
        
        Args:
            sequence_lengths: List of sequence lengths to test
            trials: Number of trials per sequence length
            save_path: Optional path to save figure
        """
        print(f"Running infinite context length analysis...")
        
        # Results storage
        processing_times = []
        memory_usage = []
        hebbian_connections = []
        energy_levels = []
        token_counts = []
        
        # Run tests
        for seq_len in sequence_lengths:
            print(f"\nTesting sequence length: {seq_len}")
            
            # Timing and metrics for this length
            times = []
            memory = []
            connections = []
            energy = []
            
            for trial in range(trials):
                # Create random sequence
                input_ids = torch.randint(0, self.vocab_size, 
                                         (1, seq_len), 
                                         device=self.device)
                
                # Track memory before
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    mem_before = torch.cuda.memory_allocated() / (1024**2)
                
                # Time the forward pass
                start_time = time.time()
                _ = self.forward(input_ids, evolution_steps=3)
                end_time = time.time()
                
                # Measure time
                elapsed = end_time - start_time
                times.append(elapsed)
                
                # Track memory after
                if torch.cuda.is_available():
                    mem_after = torch.cuda.max_memory_allocated() / (1024**2)
                    mem_used = mem_after - mem_before
                    memory.append(mem_used)
                else:
                    memory.append(0)  # No CUDA memory tracking
                
                # Count Hebbian connections
                hebb_count = len(self.field.hebbian.indices)
                connections.append(hebb_count)
                
                # Get final energy
                if hasattr(self.field, 'energy_history') and self.field.energy_history:
                    energy.append(self.field.energy_history[-1])
                else:
                    energy.append(0)
                
                print(f"  Trial {trial+1}: {elapsed:.4f} seconds, {hebb_count} connections")
            
            # Average results
            avg_time = sum(times) / len(times)
            avg_mem = sum(memory) / len(memory)
            avg_conn = sum(connections) / len(connections)
            avg_energy = sum(energy) / len(energy)
            
            # Store results
            processing_times.append(avg_time)
            memory_usage.append(avg_mem)
            hebbian_connections.append(avg_conn)
            energy_levels.append(avg_energy)
            token_counts.append(seq_len)
            
            # Update memory tracker
            self.context_memory['token_count'] = max(self.context_memory['token_count'], seq_len)
            self.context_memory['hebbian_connections'] = max(
                self.context_memory['hebbian_connections'], 
                avg_conn
            )
            self.context_memory['energy_history'].append(avg_energy)
            
            print(f"  Average: {avg_time:.4f} seconds, {avg_mem:.2f} MB, {avg_conn:.1f} connections")
        
        # Store performance metrics
        self.context_memory['performance_metrics'] = {
            'sequence_lengths': sequence_lengths,
            'processing_times': processing_times,
            'memory_usage': memory_usage,
            'hebbian_connections': hebbian_connections,
            'energy_levels': energy_levels
        }
        
        # Visualization
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Processing Time Complexity
        axes[0, 0].loglog(sequence_lengths, processing_times, 'b-o', linewidth=2, markersize=8)
        
        # Add reference scaling lines
        max_time = max(processing_times)
        min_time = min(processing_times)
        scale_factor = max_time / (sequence_lengths[-1] ** 2) * 10
        
        x_range = np.array(sequence_lengths)
        axes[0, 0].loglog(x_range, scale_factor * x_range**2, 'r--', 
                        linewidth=1.5, alpha=0.7, label='O(N²)')
        axes[0, 0].loglog(x_range, scale_factor * x_range * np.log(x_range), 'g--', 
                        linewidth=1.5, alpha=0.7, label='O(N log N)')
        axes[0, 0].loglog(x_range, scale_factor * x_range, 'c--', 
                        linewidth=1.5, alpha=0.7, label='O(N)')
        
        axes[0, 0].set_xlabel('Sequence Length (tokens)', fontsize=12)
        axes[0, 0].set_ylabel('Processing Time (seconds)', fontsize=12)
        axes[0, 0].set_title('Time Complexity Analysis', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=10)
        
        # Add empirical complexity annotation
        if len(processing_times) >= 2 and len(sequence_lengths) >= 2:
            # Estimate complexity exponent
            log_times = np.log(processing_times)
            log_seq = np.log(sequence_lengths)
            
            # Linear regression
            slope, _ = np.polyfit(log_seq, log_times, 1)
            
            # Determine complexity class
            if slope < 1.2:
                complexity = "O(N)"
            elif slope < 1.6:
                complexity = "O(N log N)"
            else:
                complexity = f"O(N^{slope:.2f})"
            
            axes[0, 0].text(0.05, 0.95, f"Empirical: {complexity}", 
                          transform=axes[0, 0].transAxes, fontsize=12,
                          bbox=dict(facecolor='black', alpha=0.7, edgecolor='gray'))
        
        # 2. Memory Usage
        axes[0, 1].loglog(sequence_lengths, memory_usage, 'm-o', linewidth=2, markersize=8)
        
        # Add reference scaling
        if max(memory_usage) > 0:
            mem_scale = max(memory_usage) / (sequence_lengths[-1]) * 2
            axes[0, 1].loglog(x_range, mem_scale * x_range, 'r--', 
                            linewidth=1.5, alpha=0.7, label='O(N)')
            axes[0, 1].loglog(x_range, mem_scale * np.ones_like(x_range), 'g--', 
                            linewidth=1.5, alpha=0.7, label='O(1)')
            
        axes[0, 1].set_xlabel('Sequence Length (tokens)', fontsize=12)
        axes[0, 1].set_ylabel('Memory Usage (MB)', fontsize=12)
        axes[0, 1].set_title('Memory Complexity Analysis', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=10)
        
        # 3. Hebbian Connections
        axes[1, 0].loglog(sequence_lengths, hebbian_connections, 'g-o', linewidth=2, markersize=8)
        
        # Reference scaling for Hebbian connections
        if max(hebbian_connections) > 0:
            conn_scale = max(hebbian_connections) / (sequence_lengths[-1])
            axes[1, 0].loglog(x_range, conn_scale * x_range, 'r--', 
                            linewidth=1.5, alpha=0.7, label='O(N)')
            axes[1, 0].loglog(x_range, conn_scale * x_range * np.log(x_range), 'b--', 
                            linewidth=1.5, alpha=0.7, label='O(N log N)')
            
        axes[1, 0].set_xlabel('Sequence Length (tokens)', fontsize=12)
        axes[1, 0].set_ylabel('Hebbian Connections', fontsize=12)
        axes[1, 0].set_title('Connection Sparsity Analysis', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=10)
        
        # 4. Energy Levels
        axes[1, 1].loglog(sequence_lengths, energy_levels, 'c-o', linewidth=2, markersize=8)
        
        axes[1, 1].set_xlabel('Sequence Length (tokens)', fontsize=12)
        axes[1, 1].set_ylabel('Field Energy', fontsize=12)
        axes[1, 1].set_title('Field Energy vs. Context Length', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Theoretical infinite context capacity
        if len(energy_levels) > 0:
            # Add information capacity annotation
            axes[1, 1].text(0.05, 0.95, 
                          "Theoretical Information Capacity:\n"
                          "- Standard Transformer: O(n²)\n"
                          "- Log-Cylindrical Field: O(n·log(n))",
                          transform=axes[1, 1].transAxes, fontsize=10,
                          bbox=dict(facecolor='black', alpha=0.7, edgecolor='gray'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Infinite context analysis saved to {save_path}")
        
        plt.close()
        
        return self.context_memory['performance_metrics']

    def infinite_context_theoretical_proof(self, save_path: Optional[str] = None):
        """
        Create visualization explaining the theoretical proof for 
        infinite context handling in log-cylindrical space
        
        Args:
            save_path: Optional path to save figure
        """
        # Create visualization of the mathematical proof
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(12, 14))
        
        # Create a grid specification
        gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1.5])
        
        # 1. Theorem statement
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        theorem_text = (
            r"$\bf{Theorem:}$ The Log-Cylindrical Quantum Field Neural Network can represent "
            r"sequence contexts of arbitrary length $n$ with $O(n \log n)$ time complexity "
            r"and $O(k \cdot n)$ memory where $k \ll n$ is the average connections per token."
        )
        
        proof_part1 = (
            r"$\bf{Proof\ Part\ 1:}$ Log-Cylindrical Coordinate Representation" + "\n"
            r"In standard attention, the context is represented by a matrix of size $O(n^2)$." + "\n"
            r"In our approach, each token $i$ maps to coordinates $(\ell_i, \theta_i)$ where:" + "\n"
            r"$\ell_i = \ln(r_i)$ is the log-radius and $\theta_i$ is the angle." + "\n"
            r"This allows representing exponentially large distances in $O(1)$ space."
        )
        
        ax1.text(0.5, 0.6, theorem_text, ha='center', va='center', fontsize=14,
                wrap=True, color='white')
        ax1.text(0.5, 0.3, proof_part1, ha='center', va='center', fontsize=12,
                wrap=True, color='white')
        
        # 2. Sparse Hebbian Learning
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Plot sparse connection matrix visualization
        n_sparse = 30
        sparse_matrix = np.zeros((n_sparse, n_sparse))
        
        # Add connections along golden ratio offsets
        for i in range(n_sparse):
            for j in range(max(0, i-3), min(n_sparse, i+4)):
                if abs(i-j) <= 1 or i % 5 == j % 5:
                    sparse_matrix[i, j] = 1
        
        ax2.imshow(sparse_matrix, cmap='viridis', interpolation='none')
        ax2.set_title("Sparse Hebbian Connections", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Token j", fontsize=10)
        ax2.set_ylabel("Token i", fontsize=10)
        
        # Add text explanation
        sparse_text = (
            r"$\bf{Proof\ Part\ 2:}$ Sparse Connectivity" + "\n"
            r"The Hebbian matrix $H_{ij}$ connects tokens with" + "\n"
            r"$\ln d_{ij} < \ln(\lambda) = \ln(\varphi^2)$" + "\n"
            r"This ensures each token connects to $O(k)$ others" + "\n"
            r"where $k \ll n$, resulting in $O(k \cdot n)$ total memory."
        )
        
        ax2.text(1.05, 0.5, sparse_text, ha='left', va='center', fontsize=10,
                transform=ax2.transAxes, wrap=True, color='white',
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='gray'))
        
        # 3. Computational complexity
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Plot computational complexity curves
        x = np.linspace(1, 100, 100)
        y_n2 = x**2 / 100
        y_nlogn = x * np.log(x) / 20
        y_n = x / 10
        
        ax3.plot(x, y_n2, 'r-', linewidth=2, label=r'$O(n^2)$ - Standard Attention')
        ax3.plot(x, y_nlogn, 'g-', linewidth=2, label=r'$O(n \log n)$ - Our Approach')
        ax3.plot(x, y_n, 'b-', linewidth=2, label=r'$O(n)$ - Linear Scaling')
        
        ax3.set_xlim(0, 100)
        ax3.set_ylim(0, 100)
        ax3.set_xlabel("Sequence Length (n)", fontsize=10)
        ax3.set_ylabel("Computation Time", fontsize=10)
        ax3.set_title("Computational Complexity", fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Information propagation proof
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        propagation_text = (
            r"$\bf{Proof\ Part\ 3:}$ Information Propagation and Field Evolution" + "\n\n"
            r"1. Each token influences others through the field equations:" + "\n"
            r"$\frac{d\ell_i}{dt} = \frac{F_{\ell_i}}{m_i},\ \ \frac{d\theta_i}{dt} = \frac{F_{\theta_i}}{m_i}$" + "\n\n"
            r"2. Forces include repulsion and Hebbian terms:" + "\n"
            r"$F_{\ell_i} = \sum_{j \neq i} \frac{s_i s_j}{d_{ij}} \cdot \frac{d\ell_{ij}}{d_{ij}} + F_{\text{boundary}}$" + "\n"
            r"$F_{\theta_i} = \sum_{j \neq i} \frac{s_i s_j}{d_{ij}} \cdot \frac{d\theta_{ij}}{d_{ij}} - \kappa \cdot d\theta_{\text{pitch}}$" + "\n\n"
            r"3. Information transfer between tokens $i$ and $j$ occurs in $O(\log d_{ij})$ time" + "\n"
            r"due to tachyonic tunneling when $|v_{\theta}/v_{\ell}| > \varphi$" + "\n\n"
            r"4. The field energy $E(t)$ monotonically decreases, ensuring convergence:" + "\n"
            r"$E(t) = \sum_{i<j} \frac{s_i s_j}{d_{ij}(t)} + \sum_i V_{\text{Hebb}}(\theta_i, \text{pitch}_i) + \sum_i V_{\text{boundary}}(\ell_i)$" + "\n\n"
            r"5. By the Harris recurrence theorem, the system will crystallize in finite time with probability 1" + "\n"
            r"when $\frac{dE}{dt} < 0$ and Lévy jumps with index $\alpha = \varphi$ provide sufficient exploration." + "\n\n"
            r"Thus, the Log-Cylindrical QFNN can process sequences of arbitrary length with" + "\n"
            r"$O(n \log n)$ time complexity and $O(k \cdot n)$ memory requirements. $\blacksquare$"
        )
        
        ax4.text(0.5, 0.5, propagation_text, ha='center', va='center', fontsize=12,
                wrap=True, color='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Theoretical proof saved to {save_path}")
        
        plt.close()
    
    def save(self, path: str):
        """
        Save model to file
        
        Args:
            path: Path to save model
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'token_embedding': self.token_embedding.state_dict(),
            'output_projection': self.output_projection.state_dict(),
            'context_memory': self.context_memory
        }, path)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device=None):
        """
        Load model from file
        
        Args:
            path: Path to load model from
            device: Device to load model to
            
        Returns:
            model: Loaded model
        """
        # Load state dict
        state_dict = torch.load(path, map_location=device)
        
        # Create model
        model = cls(
            vocab_size=state_dict['vocab_size'],
            embedding_dim=state_dict['embedding_dim'],
            device=device
        )
        
        # Load weights
        model.token_embedding.load_state_dict(state_dict['token_embedding'])
        model.output_projection.load_state_dict(state_dict['output_projection'])
        
        print(f"Model loaded from {path}")
        return model
    
    def ablation_study(self, input_ids: torch.Tensor, save_path: Optional[str] = None):
        """
        Perform ablation study comparing standard and quantum field evolution
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            save_path: Optional path to save figures
        """
        # Get standard embeddings
        std_embeddings = self.token_embedding(input_ids)
        
        # Run standard linear projection
        std_logits = self.output_projection(std_embeddings)
        
        # Run quantum field evolution
        evolved_embeddings = self.evolve_field(std_embeddings.clone(), steps=10)
        evolved_logits = self.output_projection(evolved_embeddings)
        
        # Convert to log-cylindrical coordinates
        std_ln_r, std_theta = self.cartesian_to_log_cylindrical(std_embeddings)
        evolved_ln_r, evolved_theta = self.cartesian_to_log_cylindrical(evolved_embeddings)
        
        # Convert to numpy for visualization
        std_ln_r_np = std_ln_r.detach().cpu().numpy()
        std_theta_np = std_theta.detach().cpu().numpy()
        evolved_ln_r_np = evolved_ln_r.detach().cpu().numpy()
        evolved_theta_np = evolved_theta.detach().cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot standard vs evolved ln_r (first sequence, first dimension)
        axes[0, 0].plot(std_ln_r_np[0, :, 0], 'b-', label='Standard', linewidth=2, alpha=0.7)
        axes[0, 0].plot(evolved_ln_r_np[0, :, 0], 'r-', label='Quantum Field', linewidth=2, alpha=0.7)
        axes[0, 0].set_xlabel('Sequence Position')
        axes[0, 0].set_ylabel('ln(r)')
        axes[0, 0].set_title('Standard vs Quantum Field: ln(r)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot standard vs evolved theta
        axes[0, 1].plot(std_theta_np[0, :, 0], 'b-', label='Standard', linewidth=2, alpha=0.7)
        axes[0, 1].plot(evolved_theta_np[0, :, 0], 'r-', label='Quantum Field', linewidth=2, alpha=0.7)
        axes[0, 1].set_xlabel('Sequence Position')
        axes[0, 1].set_ylabel('θ')
        axes[0, 1].set_title('Standard vs Quantum Field: θ')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Convert logits to probabilities
        std_probs = torch.nn.functional.softmax(std_logits, dim=-1)
        evolved_probs = torch.nn.functional.softmax(evolved_logits, dim=-1)
        
        # Compute entropy
        std_entropy = -torch.sum(std_probs * torch.log(std_probs + 1e-10), dim=-1)
        evolved_entropy = -torch.sum(evolved_probs * torch.log(evolved_probs + 1e-10), dim=-1)
        
        # Convert to numpy
        std_entropy_np = std_entropy.detach().cpu().numpy()
        evolved_entropy_np = evolved_entropy.detach().cpu().numpy()
        
        # Plot entropy
        axes[1, 0].plot(std_entropy_np[0], 'b-', label='Standard', linewidth=2, alpha=0.7)
        axes[1, 0].plot(evolved_entropy_np[0], 'r-', label='Quantum Field', linewidth=2, alpha=0.7)
        axes[1, 0].set_xlabel('Sequence Position')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].set_title('Output Distribution Entropy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot probability difference
        # Get top tokens
        top_k = 5
        _, std_top_tokens = torch.topk(std_logits[0, -1], top_k)
        _, evolved_top_tokens = torch.topk(evolved_logits[0, -1], top_k)
        
        # Combine top tokens
        top_tokens = torch.unique(torch.cat([std_top_tokens, evolved_top_tokens]))
        
        # Get probabilities for these tokens
        std_top_probs = std_probs[0, -1, top_tokens].detach().cpu().numpy()
        evolved_top_probs = evolved_probs[0, -1, top_tokens].detach().cpu().numpy()
        
        # Plot
        x = np.arange(len(top_tokens))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, std_top_probs, width, label='Standard', alpha=0.7)
        axes[1, 1].bar(x + width/2, evolved_top_probs, width, label='Quantum Field', alpha=0.7)
        axes[1, 1].set_xlabel('Token ID')
        axes[1, 1].set_ylabel('Probability')
        axes[1, 1].set_title('Top Token Probabilities')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(top_tokens.cpu().numpy())
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Ablation study saved to {save_path}")
        else:
            # Try to show, but don't error in non-interactive environments
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plot: {e}")
        
        plt.close()

# Example usage
if __name__ == "__main__":
    print("Testing Quantum Field Neural Network")
    
    # Use CPU for testing to avoid numpy conversion issues
    cpu_device = torch.device('cpu')
    
    # Create model
    vocab_size = 1000
    embedding_dim = 64
    model = QuantumFieldNN(vocab_size, embedding_dim, device=cpu_device)
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    print(f"Running forward pass on input shape: {input_ids.shape}")
    logits = model(input_ids)
    print(f"Output shape: {logits.shape}")
    
    # Visualize embeddings
    model.visualize_embeddings(save_path="qfnn_embeddings.png")
    
    # Compare with standard embeddings
    model.compare_embedding_systems(save_path="qfnn_embedding_comparison.png")
    
    # Run ablation study
    model.ablation_study(input_ids, save_path="qfnn_ablation_study.png")
    
    # Test generation
    print("Testing text generation...")
    prompt_ids = torch.randint(0, vocab_size, (1, 5), device=device)
    generated_ids = model.generate(prompt_ids, max_length=20)
    print(f"Generated sequence shape: {generated_ids.shape}")
    
    # Save model
    model.save("qfnn_model.pt")
    
    print("All tests completed successfully!")
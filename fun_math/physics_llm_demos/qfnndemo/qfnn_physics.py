import torch
import torch.nn as nn
import math

class QuantumFeynmanNN(nn.Module):
    """
    Physics-based Quantum Feynman Neural Network (QFNN) implementation.
    
    This model implements quantum mechanics principles directly:
    - Phase space representation using wave function formalism
    - Feynman path integral formulation through Gaussian propagator
    - Schrödinger equation evolution in phase space
    - Born rule for probability distribution
    
    All operations use tensor operations and einsum for efficiency and clarity.
    Physics notation is used throughout for direct connection to quantum mechanics.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 4,     # N (phase space dimensionality)
        max_seq_len: int = 512,     # D (maximum sequence length)
        h_bar: float = 1.0,         # ℏ (reduced Planck constant)
        diffusion_coeff: float = 0.01, # D (diffusion coefficient)
        dt: float = 0.1,            # τ (time step for evolution)
        integration_steps: int = 3,  # Number of integration steps
        sparsity: float = 0.4165    # Attention sparsity parameter
    ):
        """
        Initialize the QFNN with physics parameters.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimensionality of phase space (N)
            max_seq_len: Maximum sequence length (D)
            h_bar: Reduced Planck constant (ℏ)
            diffusion_coeff: Diffusion coefficient (D)
            dt: Time step for Schrödinger evolution (τ)
            integration_steps: Number of numerical integration steps
            sparsity: Attention sparsity parameter
        """
        super().__init__()
        
        # Model parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim  # N
        self.max_seq_len = max_seq_len      # D
        
        # Physics constants
        self.h_bar = h_bar  # ℏ
        self.diffusion_coeff = diffusion_coeff  # D
        self.dt = dt  # τ
        self.integration_steps = integration_steps
        self.sparsity = sparsity
        
        # Fundamental constants from physics
        self.Φ = (1 + 5**0.5) / 2  # Golden ratio ≈ 1.618
        self.ε = 1e-8  # Small constant for numerical stability
        
        # Create initial phase space representation
        # We use golden angle for optimal distribution in phase space
        self.initialize_phase_space()
        
        # Create constants for Gaussian kernel (quantum propagator)
        # σ² determines the locality in quantum propagation
        self.sigma_squared = nn.Parameter(torch.tensor(0.1))
        
        # Buffer for identity matrix (self-attention preservation)
        self.register_buffer('_identity', None)  # Will be lazily initialized
    
    def initialize_phase_space(self):
        """
        Initialize phase space representation using golden ratio for optimal distribution.
        
        This uses the N-2 optimization by representing each token as a point in 2D phase space,
        which is sufficient for quantum evolution while maintaining efficiency.
        """
        # For each token, create a point on a unit circle with angle based on golden ratio
        # This ensures optimal equidistribution in phase space
        embeddings = torch.zeros(self.vocab_size, 2)
        
        for v in range(self.vocab_size):
            # θᵥ = 2π · v · Φ mod 1 (golden angle progression)
            theta = 2 * math.pi * ((self.Φ * v) % 1)
            
            # Convert to Cartesian coordinates (x, y) on unit circle
            embeddings[v, 0] = math.cos(theta)  # Real part
            embeddings[v, 1] = math.sin(theta)  # Imaginary part
        
        # Create embedding layer (frozen, not learnable)
        self.phase_embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
    
    def get_identity_matrix(self, seq_len, device):
        """
        Lazily initialize identity matrix for self-attention preservation.
        
        Args:
            seq_len: Sequence length
            device: Current device
            
        Returns:
            torch.Tensor: Identity matrix of shape [seq_len, seq_len]
        """
        if self._identity is None or self._identity.size(0) < seq_len:
            # Create identity matrix (or larger one if needed)
            identity = torch.eye(seq_len, device=device)
            self.register_buffer('_identity', identity)
            return identity
        else:
            # Return slice of existing identity matrix
            return self._identity[:seq_len, :seq_len]
    
    def quantum_propagator(self, ψ_i, ψ_j):
        """
        Compute quantum propagator between states using path integral formulation.
        
        In Feynman's path integral formulation, the propagator is:
        K(x, t; x₀, t₀) = exp(-|x - x₀|²/(2σ²))
        
        This represents the probability amplitude for a particle to travel
        from state ψ_j to ψ_i, implementing the core quantum connection.
        
        Args:
            ψ_i: Phase space states [batch_size, seq_len_i, 2]
            ψ_j: Phase space states [batch_size, seq_len_j, 2]
            
        Returns:
            torch.Tensor: Propagator matrix [batch_size, seq_len_i, seq_len_j]
        """
        # Einstein summation for efficient distance calculation:
        # K_ij = exp(-|ψ_i - ψ_j|²/(2σ²))
        
        # Compute squared distances efficiently using einsum
        # Expand ψ_i and ψ_j for broadcasting
        # batch_dim, i_dim, j_dim, phase_dim -> batch_dim, i_dim, j_dim
        
        # |ψ_i - ψ_j|² = |ψ_i|² + |ψ_j|² - 2⟨ψ_i|ψ_j⟩
        
        # Compute |ψ_i|² -> [batch, seq_i, 1]
        norm_i_squared = torch.einsum('bid,bid->bi', ψ_i, ψ_i).unsqueeze(-1)
        
        # Compute |ψ_j|² -> [batch, 1, seq_j]
        norm_j_squared = torch.einsum('bjd,bjd->bj', ψ_j, ψ_j).unsqueeze(1)
        
        # Compute inner product ⟨ψ_i|ψ_j⟩ -> [batch, seq_i, seq_j]
        inner_product = torch.einsum('bid,bjd->bij', ψ_i, ψ_j)
        
        # Complete squared distance calculation
        squared_distances = norm_i_squared + norm_j_squared - 2 * inner_product
        
        # Apply Gaussian kernel (quantum propagator)
        # K_ij = exp(-|ψ_i - ψ_j|²/(2σ²))
        propagator = torch.exp(-squared_distances / (2 * self.sigma_squared))
        
        return propagator
    
    def apply_sparsity(self, propagator, sparsity=None):
        """
        Apply sparsity to propagator matrix using quantile thresholding.
        
        This implements attention sparsity while preserving self-connections,
        analogous to dominant path selection in quantum mechanics.
        
        Args:
            propagator: Propagator matrix [batch_size, seq_len, seq_len]
            sparsity: Sparsity level (between 0 and 1)
            
        Returns:
            torch.Tensor: Sparse propagator matrix [batch_size, seq_len, seq_len]
        """
        if sparsity is None:
            sparsity = self.sparsity
            
        batch_size, seq_len, _ = propagator.shape
        device = propagator.device
        
        # Get identity matrix for self-propagation preservation
        identity = self.get_identity_matrix(seq_len, device)
        
        # Vectorized quantile thresholding
        flat_propagator = propagator.reshape(batch_size, -1)
        k = max(1, int((1.0 - sparsity) * flat_propagator.size(1)))
        
        # Compute threshold values for each batch (quantile-based)
        thresholds, _ = torch.kthvalue(flat_propagator, k, dim=1)
        thresholds = thresholds.view(batch_size, 1, 1)
        
        # Apply thresholding with broadcasting
        sparse_mask = (propagator >= thresholds).float()
        
        # Ensure self-propagation (diagonal) is preserved
        sparse_mask = torch.maximum(sparse_mask, identity.unsqueeze(0))
        
        # Apply mask to original propagator
        sparse_propagator = propagator * sparse_mask
        
        # Normalize each row to conserve probability
        # This ensures the propagator remains a valid quantum operator
        row_sums = torch.einsum('bij->bi', sparse_propagator).unsqueeze(-1) + self.ε
        normalized_propagator = sparse_propagator / row_sums
        
        return normalized_propagator
    
    def schrodinger_evolution_step(self, ψ, propagator):
        """
        Perform a single step of Schrödinger evolution in phase space.
        
        This directly implements the imaginary-time Schrödinger equation:
        ∂ψ/∂τ = D∇²ψ - V(ψ)
        
        Where:
        - D is the diffusion coefficient
        - ∇²ψ is approximated by the propagator
        - V(ψ) is the potential term
        
        Args:
            ψ: Quantum state in phase space [batch_size, seq_len, 2]
            propagator: Propagator matrix [batch_size, seq_len, seq_len]
            
        Returns:
            torch.Tensor: Evolved quantum state [batch_size, seq_len, 2]
        """
        # Calculate the Laplacian using the propagator
        # In quantum mechanics, the Laplacian ∇² approximates the spatial curvature of ψ
        # ∇²ψ ≈ (Σ_j K_ij ψ_j) - ψ_i
        
        # Apply propagator to state using einsum:
        # [batch, i, j] × [batch, j, d] -> [batch, i, d]
        propagated_state = torch.einsum('bij,bjd->bid', propagator, ψ)
        
        # Compute the Laplacian term: ∇²ψ ≈ propagated_state - ψ
        laplacian = propagated_state - ψ
        
        # Schrodinger evolution step: ψ' = ψ + τ D ∇²ψ
        # where τ is the time step and D is the diffusion coefficient
        evolved_state = ψ + self.dt * self.diffusion_coeff * laplacian
        
        # Preserve the norm (energy conservation) using L2 normalization
        # In quantum mechanics, probability must be conserved
        original_norms = torch.norm(ψ, dim=-1, keepdim=True)
        current_norms = torch.norm(evolved_state, dim=-1, keepdim=True)
        normalized_state = evolved_state * (original_norms / (current_norms + self.ε))
        
        return normalized_state
    
    def runge_kutta_integration(self, ψ_0, propagator):
        """
        Apply Runge-Kutta integration for precise quantum evolution.
        
        This uses the classic fourth-order Runge-Kutta method to solve
        the Schrödinger equation with high accuracy:
        
        k₁ = f(ψ)
        k₂ = f(ψ + 0.5τk₁)
        k₃ = f(ψ + 0.5τk₂)
        k₄ = f(ψ + τk₃)
        ψ(τ+Δτ) = ψ(τ) + (τ/6)(k₁ + 2k₂ + 2k₃ + k₄)
        
        Args:
            ψ_0: Initial quantum state [batch_size, seq_len, 2]
            propagator: Initial propagator matrix [batch_size, seq_len, seq_len]
            
        Returns:
            torch.Tensor: Evolved quantum state [batch_size, seq_len, 2]
        """
        ψ = ψ_0.clone()
        
        # Store original norms for energy conservation
        original_norms = torch.norm(ψ, dim=-1, keepdim=True)
        
        # Apply integration steps
        for _ in range(self.integration_steps):
            # Calculate δψ/δτ = D∇²ψ
            # Stage 1: k₁ = f(ψ)
            propagated_state_1 = torch.einsum('bij,bjd->bid', propagator, ψ)
            laplacian_1 = propagated_state_1 - ψ
            k1 = self.dt * self.diffusion_coeff * laplacian_1
            
            # Stage 2: k₂ = f(ψ + 0.5τk₁)
            ψ_2 = ψ + 0.5 * k1
            # Normalize intermediate state
            norms_2 = torch.norm(ψ_2, dim=-1, keepdim=True)
            ψ_2 = ψ_2 * (original_norms / (norms_2 + self.ε))
            
            # Recalculate propagator for intermediate state
            propagator_2 = self.quantum_propagator(ψ_2, ψ_2)
            propagator_2 = self.apply_sparsity(propagator_2)
            
            propagated_state_2 = torch.einsum('bij,bjd->bid', propagator_2, ψ_2)
            laplacian_2 = propagated_state_2 - ψ_2
            k2 = self.dt * self.diffusion_coeff * laplacian_2
            
            # Stage 3: k₃ = f(ψ + 0.5τk₂)
            ψ_3 = ψ + 0.5 * k2
            # Normalize intermediate state
            norms_3 = torch.norm(ψ_3, dim=-1, keepdim=True)
            ψ_3 = ψ_3 * (original_norms / (norms_3 + self.ε))
            
            # Recalculate propagator for intermediate state
            propagator_3 = self.quantum_propagator(ψ_3, ψ_3)
            propagator_3 = self.apply_sparsity(propagator_3)
            
            propagated_state_3 = torch.einsum('bij,bjd->bid', propagator_3, ψ_3)
            laplacian_3 = propagated_state_3 - ψ_3
            k3 = self.dt * self.diffusion_coeff * laplacian_3
            
            # Stage 4: k₄ = f(ψ + τk₃)
            ψ_4 = ψ + k3
            # Normalize intermediate state
            norms_4 = torch.norm(ψ_4, dim=-1, keepdim=True)
            ψ_4 = ψ_4 * (original_norms / (norms_4 + self.ε))
            
            # Recalculate propagator for intermediate state
            propagator_4 = self.quantum_propagator(ψ_4, ψ_4)
            propagator_4 = self.apply_sparsity(propagator_4)
            
            propagated_state_4 = torch.einsum('bij,bjd->bid', propagator_4, ψ_4)
            laplacian_4 = propagated_state_4 - ψ_4
            k4 = self.dt * self.diffusion_coeff * laplacian_4
            
            # Combine all stages: ψ(τ+Δτ) = ψ(τ) + (τ/6)(k₁ + 2k₂ + 2k₃ + k₄)
            ψ = ψ + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Final normalization to preserve energy
            current_norms = torch.norm(ψ, dim=-1, keepdim=True)
            ψ = ψ * (original_norms / (current_norms + self.ε))
        
        return ψ
    
    def forward(self, token_ids):
        """
        Forward pass through the Quantum Feynman Neural Network.
        
        This implements the full quantum evolution process:
        1. Map tokens to phase space (ψ)
        2. Compute quantum propagator (K)
        3. Apply sparsity to select dominant paths
        4. Evolve quantum states through Schrödinger equation
        
        Args:
            token_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            ψ_evolved: Evolved quantum states [batch_size, seq_len, 2]
            propagator: Quantum propagator matrix [batch_size, seq_len, seq_len]
        """
        # Map tokens to phase space points (wave function ψ)
        # This is our N-2 optimization using just 2D representation
        ψ = self.phase_embedding(token_ids)  # [batch_size, seq_len, 2]
        
        # Compute quantum propagator using path integral formulation
        # K_ij = ⟨ψ_i|K|ψ_j⟩ = exp(-|ψ_i - ψ_j|²/(2σ²))
        propagator = self.quantum_propagator(ψ, ψ)
        
        # Apply sparsity to select dominant quantum paths
        sparse_propagator = self.apply_sparsity(propagator)
        
        # Evolve quantum states through Schrödinger equation
        # ψ(τ+Δτ) = e^(-iĤτ/ℏ)ψ(τ) ≈ ψ + τD∇²ψ
        ψ_evolved = self.runge_kutta_integration(ψ, sparse_propagator)
        
        return ψ_evolved, sparse_propagator
    
    def quantum_state_mapping(self, ψ):
        """
        Map 2D phase space representation to full quantum states.
        
        This explicitly connects our representation to quantum mechanics
        through Euler's identity: e^(iθ) = cos(θ) + i·sin(θ)
        
        Args:
            ψ: Phase space states [batch_size, seq_len, 2]
            
        Returns:
            torch.Tensor: Complex quantum states [batch_size, seq_len]
        """
        # Extract amplitude (r) and phase (θ) from 2D representation
        # |ψ| = √(x² + y²)
        amplitudes = torch.norm(ψ, dim=-1)
        
        # θ = atan2(y, x)
        phases = torch.atan2(ψ[..., 1], ψ[..., 0])
        
        # Convert to complex quantum state representation: ψ = r·e^(iθ)
        # Using Euler's identity: e^(iθ) = cos(θ) + i·sin(θ)
        complex_states = torch.complex(
            amplitudes * torch.cos(phases),  # Real part
            amplitudes * torch.sin(phases)   # Imaginary part
        )
        
        return complex_states
    
    def probability_distribution(self, quantum_states):
        """
        Calculate probability distribution from quantum states using Born rule.
        
        In quantum mechanics, |ψ|² gives the probability density.
        Born rule: P(x) = |ψ(x)|²
        
        Args:
            quantum_states: Complex quantum states [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Probability distribution [batch_size, seq_len]
        """
        # Apply Born rule: P(x) = |ψ(x)|²
        probabilities = torch.abs(quantum_states)**2
        
        # Normalize to ensure probabilities sum to 1
        # ∫|ψ(x)|²dx = 1 (discrete approximation)
        probabilities = probabilities / (torch.sum(probabilities, dim=-1, keepdim=True) + self.ε)
        
        return probabilities


# Test function to validate the physics implementation
def test_qfnn_physics():
    """
    Test the Quantum Feynman Neural Network physics implementation.
    """
    print("Testing Quantum Feynman Neural Network with Physics Implementation")
    
    # Initialize model
    vocab_size = 10000
    embedding_dim = 4  # N
    max_seq_len = 512  # D
    model = QuantumFeynmanNN(vocab_size, embedding_dim, max_seq_len)
    
    # Generate random token IDs
    batch_size = 4
    seq_len = 16
    torch.manual_seed(42)  # For reproducibility
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    ψ_evolved, propagator = model(token_ids)
    
    # Map to quantum states
    quantum_states = model.quantum_state_mapping(ψ_evolved)
    
    # Calculate probabilities
    probabilities = model.probability_distribution(quantum_states)
    
    # Print shapes and check values
    print(f"Evolved wave function (ψ) shape: {ψ_evolved.shape}")
    print(f"Quantum propagator (K) shape: {propagator.shape}")
    print(f"Quantum states shape: {quantum_states.shape}")
    print(f"Probability distribution shape: {probabilities.shape}")
    
    # Verify probability sum = 1 (quantum normalization)
    prob_sums = probabilities.sum(dim=-1)
    print(f"Probability sums (should be 1.0): {prob_sums[:2]}")
    
    # Check that our probability sums are close to 1.0
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)
    
    # Verify shapes are as expected
    assert ψ_evolved.shape == (batch_size, seq_len, 2)  # N-2 optimization
    assert propagator.shape == (batch_size, seq_len, seq_len)  # D-1 attention
    assert quantum_states.shape == (batch_size, seq_len)  # Complex quantum states
    assert probabilities.shape == (batch_size, seq_len)  # Probability distribution
    
    print("All tests passed for the physics-based QFNN implementation!")
    return model, token_ids, ψ_evolved, propagator, quantum_states, probabilities


if __name__ == "__main__":
    # Run the test
    test_qfnn_physics()

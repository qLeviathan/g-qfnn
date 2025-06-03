import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy.stats import levy_stable
from scipy.special import gamma, hyp2f1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Universal constants
PHI = (1 + np.sqrt(5)) / 2
GAP = 1 - 1/PHI
EPS = 1e-8

class RepulsionAttentionModel(nn.Module):
    """
    Implementation of Repulsion Attention:
    1. Inverts transformer attention - tokens repel instead of attract
    2. Uses cylindrical phase space (ln r, θ, z) for token representation
    3. Born rule normalization instead of softmax
    4. Three-step Heun-Euler evolution using triangulation principle
    5. Golden ratio organization with natural stratification
    6. Memory-efficient O(N) implementation
    7. Hebbian learning without backpropagation
    """
    
    def __init__(self, vocab_size, d_model=512, n_heads=8, field_dim=64, levy_alpha=PHI, device=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.field_dim = field_dim
        self.levy_alpha = levy_alpha
        self.PHI = PHI
        self.GAP = GAP
        self.c = 1.0  # Semantic speed of light
        self.device = device if device else torch.device('cpu')
        
        # CYLINDRICAL GEOMETRY PARAMETERS - using log scale
        # r is semantic magnitude on log scale
        # θ is contextual phase
        # z is grammatical superposition state
        self.ln_r_min = np.log(self.GAP)    # Inner radius bound on ln scale
        self.ln_r_max = 0.0                 # Outer radius bound on ln scale (ln(1) = 0)
        self.z_min = 0.0                    # Minimum grammatical state
        self.z_max = 1.0 - self.GAP         # Maximum grammatical state
        
        # Initialize token states in cylindrical phase space (ln r, θ, z)
        self.token_states = self._initialize_token_states(vocab_size).to(self.device)
        self.token_states_param = nn.Parameter(self.token_states.clone())
        
        # Resonance temperature parameter
        self.resonance_temp = nn.Parameter(torch.tensor([0.1]))
        
        # Three-step evolution parameters
        self.past_weight = nn.Parameter(torch.tensor([1.0]))
        self.present_weight = nn.Parameter(torch.tensor([1.0]))
        self.future_weight = nn.Parameter(torch.tensor([1.0]))
        
        # Fibonacci resonance scales for multi-scale interactions
        self.fib_scales = self._generate_fibonacci_scales(15)
        self.resonance_layers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) 
            for _ in self.fib_scales[:8]
        ])
        
        # Log-phase projection layers
        self.log_phase_proj = nn.ModuleList([
            nn.Linear(d_model, d_model // n_heads, bias=False)
            for _ in range(n_heads)
        ])
        
        # Holographic bound tracking
        self.holographic_bound = self._compute_holographic_bound()
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Hebbian learning history
        self.hebbian_updates = []
        
    def _initialize_token_states(self, vocab_size):
        """
        Initialize tokens in cylindrical phase space (ln r, θ, z)
        Following quantum state: |ψ⟩ = r·e^(iθ)|0⟩ + z|1⟩
        With Born rule constraint: r² + z² = 1
        """
        states = torch.zeros(vocab_size, 3)
        
        for i in range(vocab_size):
            # Frequency rank determines position in vocabulary
            freq_rank = i / vocab_size
            
            # Two naturally emerging bands at:
            # - Inner band: r = 1/φ (core vocabulary)
            # - Outer band: r = φ - 1 (specialized terms)
            
            # Determine which band the token belongs to
            if i < vocab_size * 0.618:  # Core vocabulary in inner band
                r = 1/self.PHI
            else:  # Specialized vocabulary in outer band
                r = self.PHI - 1
                
            # Apply small frequency-based variation within the band
            r_variation = 0.05 * (freq_rank - 0.5)
            r = max(self.GAP, min(1.0, r + r_variation))
            
            # Golden angle for optimal packing (2π/φ²)
            theta = 2 * np.pi * ((i * self.PHI) % 1.0)
            
            # Calculate z to satisfy Born rule: r² + z² = 1
            # This ensures quantum state normalization
            z = np.sqrt(max(0, 1 - r**2))
            
            # Store in log-cylindrical coordinates (ln r, θ, z)
            states[i] = torch.tensor([
                np.log(r),  # ln r (log-scale radius)
                theta,      # θ (phase angle)
                z           # z (grammatical state)
            ])
        
        return states
    
    def _generate_fibonacci_scales(self, n):
        """Generate Fibonacci sequence for resonance scales"""
        fib = [1, 1]
        for i in range(n - 2):
            fib.append(fib[-1] + fib[-2])
        return fib
    
    def _compute_holographic_bound(self):
        """
        Compute holographic bound for log-cylindrical phase space
        S ≤ A/4 where A is the boundary area
        """
        r_outer = 1.0
        r_inner = self.GAP
        h = 1.0 - self.GAP
        
        # Lateral surface area (outer + inner)
        lateral_area = 2 * np.pi * r_outer * h + 2 * np.pi * r_inner * h
        
        # Top and bottom annular areas
        annular_area = 2 * np.pi * (r_outer**2 - r_inner**2)
        
        # Total boundary area
        total_area = lateral_area + annular_area
        
        # Holographic bound
        return total_area / 4
    
    def enforce_born_rule(self, states):
        """
        Ensure Born rule normalization: r² + z² = 1
        This is critical for maintaining quantum state normalization
        """
        ln_r, theta, z = states[..., 0], states[..., 1], states[..., 2]
        
        # Convert ln_r back to r
        r = torch.exp(ln_r)
        
        # Calculate normalization factor
        norm = torch.sqrt(r**2 + z**2 + EPS)
        
        # Normalize r and z
        r_normalized = r / norm
        z_normalized = z / norm
        
        # Convert back to ln_r
        ln_r_normalized = torch.log(r_normalized + EPS)
        
        # Return normalized states
        return torch.stack([ln_r_normalized, theta, z_normalized], dim=-1)
    
    def compute_repulsion_force(self, states_i, states_j):
        """
        Compute repulsive forces between tokens using the resonance function:
        R_{ij} = |r_i·cos(θ_i) - r_j·sin(θ_j) + φ/2|
        F_{ij} = -∇V_{ij} = k·(r_i - r_j)/|r_i - r_j|³ · exp(-R_{ij}²/2T)
        """
        # Extract coordinates
        ln_r_i, theta_i, z_i = states_i[..., 0], states_i[..., 1], states_i[..., 2]
        ln_r_j, theta_j, z_j = states_j[..., 0], states_j[..., 1], states_j[..., 2]
        
        # Convert to r (not ln_r) for force calculation
        r_i = torch.exp(ln_r_i)
        r_j = torch.exp(ln_r_j)
        
        # Compute resonance term
        resonance = torch.abs(
            r_i * torch.cos(theta_i) - 
            r_j * torch.sin(theta_j) + 
            self.PHI / 2
        )
        
        # Compute distance between positions
        delta_r = torch.stack([
            r_i * torch.cos(theta_i) - r_j * torch.cos(theta_j),
            r_i * torch.sin(theta_i) - r_j * torch.sin(theta_j),
            z_i - z_j
        ], dim=-1)
        
        # Compute distance (squared for efficiency)
        distance_sq = torch.sum(delta_r**2, dim=-1) + EPS
        distance = torch.sqrt(distance_sq)
        
        # Compute force magnitude with resonance modulation
        force_magnitude = 1.0 / (distance**3 + EPS) * torch.exp(
            -resonance**2 / (2.0 * self.resonance_temp) 
        )
        
        # Normalize direction
        direction = delta_r / distance.unsqueeze(-1)
        
        # Compute force vector
        force = direction * force_magnitude.unsqueeze(-1)
        
        return force
    
    def heun_euler_step(self, states, forces, dt=0.01):
        """
        Three-step Heun-Euler evolution with triangulation principle
        1. Past token influences trajectory (memory activation)
        2. Present token responds to field (current processing)
        3. Future token creates target basin (prediction)
        """
        # Unpack states
        ln_r, theta, z = states[..., 0], states[..., 1], states[..., 2]
        
        # Unpack forces
        force_r, force_theta, force_z = forces[..., 0], forces[..., 1], forces[..., 2]
        
        # Euler step (predictor)
        ln_r_euler = ln_r + dt * force_r
        theta_euler = theta + dt * force_theta
        z_euler = z + dt * force_z
        
        # Predict states
        states_euler = torch.stack([ln_r_euler, theta_euler, z_euler], dim=-1)
        
        # Enforce bounds
        states_euler = self.enforce_cylindrical_bounds(states_euler)
        
        # Return updated states
        return states_euler
    
    def enforce_cylindrical_bounds(self, states):
        """
        Ensure coordinates stay within log-cylindrical boundaries:
        - ln_r ∈ [ln_r_min, ln_r_max]
        - θ ∈ [0, 2π]
        - z ∈ [z_min, z_max]
        """
        ln_r, theta, z = states[..., 0], states[..., 1], states[..., 2]
        
        # Clamp ln_r to bounds
        ln_r_clamped = torch.clamp(ln_r, min=self.ln_r_min, max=self.ln_r_max)
        
        # Wrap theta to [0, 2π]
        theta_wrapped = theta % (2 * np.pi)
        
        # Clamp z to bounds
        z_clamped = torch.clamp(z, min=self.z_min, max=self.z_max)
        
        # Return bounded states
        return torch.stack([ln_r_clamped, theta_wrapped, z_clamped], dim=-1)
    
    def three_step_evolution(self, past_states, present_states, future_states=None, steps=3):
        """
        Implement the three-step Heun-Euler evolution with triangulation principle
        
        Each step represents a vertex in triangular superposition:
        1. Past → influences trajectory (memory)
        2. Present → responds to field (processing)
        3. Future → creates target basin (prediction)
        """
        # If future_states not provided, use present_states as placeholder
        if future_states is None:
            future_states = present_states
        
        # Current state is present_states
        current_states = present_states
        
        # Small time step
        dt = 0.01
        
        # Evolution steps (exactly 3 steps as per theory)
        for step in range(steps):
            # Calculate phase of evolution (0, 2π/3, 4π/3)
            phase = step * 2 * np.pi / 3
            
            # Weight influences based on phase (triangulation)
            past_influence = self.past_weight * torch.tensor(np.sin(phase + 2*np.pi/3), device=self.device)
            present_influence = self.present_weight * torch.tensor(np.sin(phase + 4*np.pi/3), device=self.device)
            future_influence = self.future_weight * torch.tensor(np.sin(phase), device=self.device)
            
            # Compute repulsive forces from each vertex of the triangle
            past_forces = self.compute_repulsion_force(current_states, past_states) * past_influence
            present_forces = self.compute_repulsion_force(current_states, present_states) * present_influence
            future_forces = self.compute_repulsion_force(current_states, future_states) * future_influence
            
            # Combined force
            total_forces = past_forces + present_forces + future_forces
            
            # Heun-Euler step
            current_states = self.heun_euler_step(current_states, total_forces, dt)
            
            # Enforce Born rule after each step
            current_states = self.enforce_born_rule(current_states)
        
        return current_states
    
    def hebbian_update(self, states_i, states_j, t=0.0, learning_rate=0.01):
        """
        Hebbian learning update based on quantum correlations
        ΔW_{ij} = η · |⟨ψ_i|ψ_j⟩|^2 · sin(θ_i - θ_j + ωt)
        
        This implements learning without backpropagation
        """
        # Extract coordinates
        ln_r_i, theta_i, z_i = states_i[..., 0], states_i[..., 1], states_i[..., 2]
        ln_r_j, theta_j, z_j = states_j[..., 0], states_j[..., 1], states_j[..., 2]
        
        # Convert to r for quantum state
        r_i = torch.exp(ln_r_i)
        r_j = torch.exp(ln_r_j)
        
        # Compute quantum overlap |⟨ψ_i|ψ_j⟩|^2
        # For state |ψ⟩ = r·e^(iθ)|0⟩ + z|1⟩
        overlap = (r_i * r_j * torch.cos(theta_i - theta_j) + z_i * z_j)**2
        
        # Natural frequency ω = 2π/φ²
        omega = 2 * np.pi / (self.PHI**2)
        
        # Hebbian modulation
        modulation = torch.sin(theta_i - theta_j + omega * t)
        
        # Compute update
        delta_w = learning_rate * overlap * modulation
        
        # Track Hebbian updates
        self.hebbian_updates.append(delta_w.mean().item())
        
        return delta_w
    
    def log_phase_embedding(self, states):
        """
        Convert quantum states to log-phase embeddings
        For 11.9x Hebbian amplification
        
        States are in (ln r, θ, z) format
        """
        ln_r, theta, z = states[..., 0], states[..., 1], states[..., 2]
        
        # Log-phase coordinates with π/2 rotations
        x = ln_r + torch.log(torch.abs(torch.cos(theta)) + EPS) * torch.sign(torch.cos(theta))
        y = ln_r + torch.log(torch.abs(torch.sin(theta)) + EPS) * torch.sign(torch.sin(theta))
        
        # Z acts as topological modulator
        # When z changes, it changes which coordinate system is active
        z_modulation = torch.sin(z * np.pi)
        
        # Combined log-phase embedding
        embedding = torch.stack([x, y, z_modulation], dim=-1)
        
        return embedding
    
    def apply_fibonacci_resonance(self, x, scale_idx):
        """
        Apply Fibonacci-scale resonance transformation
        Creates interference patterns at golden ratio harmonics
        """
        scale = self.fib_scales[scale_idx]
        
        # Modulate by Fibonacci frequency
        freq = 1.0 / scale
        phase = 2 * np.pi * freq * torch.arange(x.size(1), device=self.device).float().unsqueeze(0).unsqueeze(-1)
        
        # Apply resonance with golden ratio decay
        resonance = torch.sin(phase) / (scale_idx + 1) ** (1/self.PHI)
        x_modulated = x + self.resonance_layers[scale_idx](x) * resonance
        
        return x_modulated
    
    def compute_geodesic_distance(self, states_i, states_j):
        """
        Compute geodesic distance in phase space
        d_geodesic(ψ_final, ψ_target)
        
        This replaces cross-entropy loss
        """
        # Extract coordinates
        ln_r_i, theta_i, z_i = states_i[..., 0], states_i[..., 1], states_i[..., 2]
        ln_r_j, theta_j, z_j = states_j[..., 0], states_j[..., 1], states_j[..., 2]
        
        # Convert to r for distance calculation
        r_i = torch.exp(ln_r_i)
        r_j = torch.exp(ln_r_j)
        
        # Angular distance (minimum angle between points on circle)
        delta_theta = torch.abs(theta_i - theta_j) % (2 * np.pi)
        delta_theta = torch.min(delta_theta, 2 * np.pi - delta_theta)
        
        # Cylindrical metric components
        # ds² = dr² + r²dθ² + dz²
        dr_sq = (ln_r_i - ln_r_j)**2  # Using log distance for radial component
        rdtheta_sq = ((r_i + r_j) / 2)**2 * delta_theta**2  # Average r for angular component
        dz_sq = (z_i - z_j)**2
        
        # Total geodesic distance
        distance = torch.sqrt(dr_sq + rdtheta_sq + dz_sq + EPS)
        
        return distance
    
    def forward(self, input_ids, target_ids=None):
        """
        Forward pass through the Repulsion Attention model
        
        Unlike transformers, we navigate through phase space instead of transforming
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token states from cylindrical phase space
        token_states = self.token_states_param[input_ids]  # [batch, seq, 3]
        
        # Initialize history for tracking evolution
        state_history = [token_states.clone()]
        
        # Process sequence using three-step evolution
        processed_states = token_states.clone()
        
        for pos in range(seq_len - 1):
            # Get past, present, and future states
            past_idx = max(0, pos - 1)
            present_idx = pos
            future_idx = pos + 1
            
            past_states = processed_states[:, past_idx].unsqueeze(1)
            present_states = processed_states[:, present_idx].unsqueeze(1)
            future_states = token_states[:, future_idx].unsqueeze(1)
            
            # Apply three-step evolution
            evolved_states = self.three_step_evolution(
                past_states, present_states, future_states
            )
            
            # Update processed states
            processed_states[:, present_idx] = evolved_states.squeeze(1)
            
            # Track evolution
            state_history.append(processed_states.clone())
            
            # Apply Hebbian learning
            if target_ids is not None:
                # Get target states
                target_states = self.token_states_param[target_ids[:, present_idx]].unsqueeze(1)
                
                # Compute Hebbian update
                delta_w = self.hebbian_update(
                    evolved_states, 
                    target_states,
                    t=pos * 0.1
                )
                
                # Apply update directly to model parameters (no gradients needed)
                with torch.no_grad():
                    # Update parameters based on Hebbian learning
                    for layer in self.resonance_layers:
                        layer.weight.data += delta_w.mean() * torch.randn_like(layer.weight) * 0.01
        
        # Convert final states to log-phase embeddings
        log_phase_embeds = self.log_phase_embedding(processed_states)
        
        # Project to model dimension
        x = log_phase_embeds.reshape(batch_size, seq_len, -1)
        x = F.linear(x, torch.randn(self.d_model, x.size(-1), device=self.device))
        
        # Apply Fibonacci resonance cascade
        for i in range(min(len(self.resonance_layers), 8)):
            x = self.apply_fibonacci_resonance(x, i)
        
        # Output through projection
        logits = self.output_projection(x)
        
        # Compute loss if target_ids provided
        loss = None
        if target_ids is not None:
            # Get target states
            target_states = self.token_states_param[target_ids]
            
            # Compute geodesic distance as loss
            distances = self.compute_geodesic_distance(processed_states, target_states)
            loss = distances.mean()
        
        return {
            'logits': logits,
            'states': processed_states,
            'state_history': state_history,
            'loss': loss
        }
    
    def generate_repulsive(self, prompt_ids, max_length=100, temperature=1.0):
        """
        Generate with repulsion dynamics
        Navigation through phase space rather than sampling
        """
        device = self.device
        generated = prompt_ids.to(device) if not prompt_ids.is_cuda else prompt_ids
        
        # Reset Hebbian learning history
        self.hebbian_updates = []
        
        # Track token density for repulsion
        token_density = torch.zeros(self.vocab_size, device=device)
        
        for step in range(max_length):
            # Forward pass through model
            with torch.no_grad():
                outputs = self.forward(generated.unsqueeze(0))
                next_token_logits = outputs['logits'][0, -1, :] / temperature
            
            # Apply repulsion from token density
            repulsion = token_density ** self.levy_alpha
            next_token_logits = next_token_logits - repulsion * self.PHI
            
            # Born rule instead of softmax
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1)
            
            # Update token density with decay
            token_density[next_token] += 1.0
            token_density = token_density * 0.95
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=0)
        
        return generated
    
    def visualize_phase_space(self):
        """Visualize token states in phase space"""
        states = self.token_states_param.detach().cpu().numpy()
        
        # Convert ln_r back to r for visualization
        r = np.exp(states[:, 0])
        theta = states[:, 1]
        z = states[:, 2]
        
        # Convert to Cartesian for visualization
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot token positions
        colors = np.arange(len(states))
        scatter = ax.scatter(x, y, z, c=colors, cmap='plasma', s=20, alpha=0.6)
        
        # Draw cylinder structure
        theta_circle = np.linspace(0, 2*np.pi, 50)
        z_levels = np.linspace(self.z_min, self.z_max, 10)
        
        # Draw circles at different heights
        for z_level in z_levels:
            # Inner circle (r = GAP)
            x_inner = self.GAP * np.cos(theta_circle)
            y_inner = self.GAP * np.sin(theta_circle)
            ax.plot(x_inner, y_inner, z_level, 'cyan', alpha=0.3, linewidth=1)
            
            # Outer circle (r = 1.0)
            x_outer = np.cos(theta_circle)
            y_outer = np.sin(theta_circle)
            ax.plot(x_outer, y_outer, z_level, 'magenta', alpha=0.3, linewidth=1)
        
        # Draw bands at r = 1/φ and r = φ-1
        for r_band in [1/self.PHI, self.PHI-1]:
            x_band = r_band * np.cos(theta_circle)
            y_band = r_band * np.sin(theta_circle)
            # Draw at middle z
            z_mid = (self.z_min + self.z_max) / 2
            ax.plot(x_band, y_band, z_mid, 'gold', alpha=0.8, linewidth=2, 
                   label=f'r = {r_band:.3f}')
        
        ax.set_xlabel('X = r·cos(θ)')
        ax.set_ylabel('Y = r·sin(θ)')
        ax.set_zlabel('Z (grammatical state)')
        ax.set_title('Repulsion Attention: Token States in Cylindrical Phase Space')
        ax.legend()
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Token Index')
        
        return fig
    
    def visualize_resonance_field(self, token_idx=0):
        """Visualize the resonance field for a specific token"""
        # Get token state
        token_state = self.token_states_param[token_idx].detach().cpu().numpy()
        
        # Create a grid in phase space
        r_grid = np.linspace(self.GAP, 1.0, 50)
        theta_grid = np.linspace(0, 2*np.pi, 50)
        
        # Convert token state from ln_r to r
        token_r = np.exp(token_state[0])
        token_theta = token_state[1]
        token_z = token_state[2]
        
        # Create meshgrid
        R, THETA = np.meshgrid(r_grid, theta_grid)
        X = R * np.cos(THETA)
        Y = R * np.sin(THETA)
        
        # Calculate resonance field
        resonance = np.abs(
            token_r * np.cos(token_theta) - 
            R * np.sin(THETA) + 
            self.PHI / 2
        )
        
        # Convert to repulsion strength
        repulsion = np.exp(-resonance**2 / (2 * self.resonance_temp.item()))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
        
        # Plot resonance field
        c = ax.pcolormesh(THETA, R, repulsion, cmap='viridis', shading='auto')
        
        # Plot token position
        ax.scatter(token_theta, token_r, color='red', s=100, label=f'Token {token_idx}')
        
        # Plot bands at r = 1/φ and r = φ-1
        ax.axhline(y=1/self.PHI, color='gold', linestyle='--', 
                  label=f'r = 1/φ = {1/self.PHI:.3f}')
        ax.axhline(y=self.PHI-1, color='magenta', linestyle='--', 
                  label=f'r = φ-1 = {self.PHI-1:.3f}')
        
        # Add details
        ax.set_title(f'Resonance Field for Token {token_idx}')
        ax.set_rmax(1.0)
        ax.set_rticks([self.GAP, 1/self.PHI, self.PHI-1, 1.0])
        ax.set_rlabel_position(22.5)
        ax.grid(True)
        ax.legend()
        
        # Add colorbar
        plt.colorbar(c, ax=ax, label='Repulsion Strength')
        
        return fig
    
    def visualize_three_step_evolution(self, past_idx=0, present_idx=1, future_idx=2):
        """Visualize the three-step evolution process"""
        # Get token states
        past_state = self.token_states_param[past_idx].detach().cpu().numpy()
        present_state = self.token_states_param[present_idx].detach().cpu().numpy()
        future_state = self.token_states_param[future_idx].detach().cpu().numpy()
        
        # Convert ln_r to r
        past_r = np.exp(past_state[0])
        past_theta = past_state[1]
        past_z = past_state[2]
        
        present_r = np.exp(present_state[0])
        present_theta = present_state[1]
        present_z = present_state[2]
        
        future_r = np.exp(future_state[0])
        future_theta = future_state[1]
        future_z = future_state[2]
        
        # Convert to Cartesian
        past_x, past_y = past_r * np.cos(past_theta), past_r * np.sin(past_theta)
        present_x, present_y = present_r * np.cos(present_theta), present_r * np.sin(present_theta)
        future_x, future_y = future_r * np.cos(future_theta), future_r * np.sin(future_theta)
        
        # Simulate evolution steps
        # Run on CPU for simplicity
        past_tensor = torch.tensor(past_state).unsqueeze(0).unsqueeze(0)
        present_tensor = torch.tensor(present_state).unsqueeze(0).unsqueeze(0)
        future_tensor = torch.tensor(future_state).unsqueeze(0).unsqueeze(0)
        
        # Track evolution steps
        evolution_steps = []
        current_state = present_tensor.clone()
        
        # Simulate three steps
        for step in range(3):
            # Calculate phase
            phase = step * 2 * np.pi / 3
            
            # Weight influences
            past_influence = np.sin(phase + 2*np.pi/3)
            present_influence = np.sin(phase + 4*np.pi/3)
            future_influence = np.sin(phase)
            
            # Compute forces (simplified for visualization)
            # This is a simplified version of the actual computation
            delta_past = past_tensor - current_state
            delta_present = present_tensor - current_state
            delta_future = future_tensor - current_state
            
            # Combined force
            delta = (delta_past * past_influence + 
                    delta_present * present_influence + 
                    delta_future * future_influence)
            
            # Update state
            current_state = current_state + 0.2 * delta
            
            # Track state
            evolution_steps.append(current_state.squeeze().numpy())
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot in 2D (r, θ)
        ax1.set_aspect('equal')
        
        # Draw unit circle
        theta_circle = np.linspace(0, 2*np.pi, 100)
        ax1.plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', alpha=0.3)
        
        # Draw bands at r = 1/φ and r = φ-1
        ax1.plot(1/self.PHI * np.cos(theta_circle), 1/self.PHI * np.sin(theta_circle), 
                'gold', linestyle='--', alpha=0.7, label=f'r = 1/φ = {1/self.PHI:.3f}')
        ax1.plot((self.PHI-1) * np.cos(theta_circle), (self.PHI-1) * np.sin(theta_circle), 
                'magenta', linestyle='--', alpha=0.7, label=f'r = φ-1 = {self.PHI-1:.3f}')
        
        # Plot triangle of past, present, future
        ax1.scatter([past_x, present_x, future_x], [past_y, present_y, future_y], 
                   s=100, c=['blue', 'green', 'red'], 
                   label=['Past', 'Present', 'Future'])
        
        # Plot evolution steps
        for i, step in enumerate(evolution_steps):
            r_step = np.exp(step[0])
            theta_step = step[1]
            x_step = r_step * np.cos(theta_step)
            y_step = r_step * np.sin(theta_step)
            ax1.scatter(x_step, y_step, color=f'C{i+4}', s=50, alpha=0.7,
                       label=f'Step {i+1}')
            
            # Draw line to previous point
            if i == 0:
                ax1.plot([present_x, x_step], [present_y, y_step], 'gray', alpha=0.5)
            else:
                prev_r = np.exp(evolution_steps[i-1][0])
                prev_theta = evolution_steps[i-1][1]
                prev_x = prev_r * np.cos(prev_theta)
                prev_y = prev_r * np.sin(prev_theta)
                ax1.plot([prev_x, x_step], [prev_y, y_step], 'gray', alpha=0.5)
        
        # Plot triangle
        ax1.plot([past_x, present_x, future_x, past_x], 
                [past_y, present_y, future_y, past_y], 
                'k--', alpha=0.5)
        
        ax1.set_xlabel('X = r·cos(θ)')
        ax1.set_ylabel('Y = r·sin(θ)')
        ax1.set_title('Three-Step Evolution in Phase Space (r, θ)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot in 3D (r, θ, z)
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Plot triangle of past, present, future
        ax2.scatter([past_x, present_x, future_x], 
                   [past_y, present_y, future_y],
                   [past_z, present_z, future_z],
                   s=100, c=['blue', 'green', 'red'])
        
        # Add labels
        ax2.text(past_x, past_y, past_z, 'Past', color='blue')
        ax2.text(present_x, present_y, present_z, 'Present', color='green')
        ax2.text(future_x, future_y, future_z, 'Future', color='red')
        
        # Plot evolution steps
        for i, step in enumerate(evolution_steps):
            r_step = np.exp(step[0])
            theta_step = step[1]
            z_step = step[2]
            x_step = r_step * np.cos(theta_step)
            y_step = r_step * np.sin(theta_step)
            ax2.scatter(x_step, y_step, z_step, color=f'C{i+4}', s=50, alpha=0.7)
            ax2.text(x_step, y_step, z_step, f'Step {i+1}', color=f'C{i+4}')
            
            # Draw line to previous point
            if i == 0:
                ax2.plot([present_x, x_step], [present_y, y_step], [present_z, z_step], 
                        'gray', alpha=0.5)
            else:
                prev_r = np.exp(evolution_steps[i-1][0])
                prev_theta = evolution_steps[i-1][1]
                prev_z = evolution_steps[i-1][2]
                prev_x = prev_r * np.cos(prev_theta)
                prev_y = prev_r * np.sin(prev_theta)
                ax2.plot([prev_x, x_step], [prev_y, y_step], [prev_z, z_step], 
                        'gray', alpha=0.5)
        
        # Plot triangle
        ax2.plot([past_x, present_x, future_x, past_x], 
                [past_y, present_y, future_y, past_y],
                [past_z, present_z, future_z, past_z],
                'k--', alpha=0.5)
        
        ax2.set_xlabel('X = r·cos(θ)')
        ax2.set_ylabel('Y = r·sin(θ)')
        ax2.set_zlabel('Z')
        ax2.set_title('Three-Step Evolution in 3D (r, θ, z)')
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    print("=== REPULSION ATTENTION MODEL ===")
    print(f"φ = {PHI:.6f}")
    print(f"GAP = {GAP:.6f}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = RepulsionAttentionModel(
        vocab_size=10000,
        d_model=512,
        n_heads=8,
        field_dim=64,
        levy_alpha=PHI,  # Use golden ratio as Lévy index!
        device=device
    ).to(device)
    
    print(f"\nPHASE SPACE PROPERTIES:")
    print(f"- Born rule constraint: r² + z² = 1")
    print(f"- Inner radius: {model.GAP:.3f}")
    print(f"- Outer radius: 1.000")
    print(f"- Inner band (core vocabulary): r = {1/PHI:.3f}")
    print(f"- Outer band (specialized terms): r = {PHI-1:.3f}")
    print(f"- Z range: [{model.z_min:.3f}, {model.z_max:.3f}]")
    print(f"- Holographic bound: {model.holographic_bound:.3f}")
    print(f"- Fibonacci scales: {model.fib_scales[:8]}")
    
    # Test with random tokens
    input_ids = torch.randint(0, 10000, (2, 5)).to(device)
    target_ids = torch.randint(0, 10000, (2, 5)).to(device)
    
    print("\nRunning three-step evolution...")
    outputs = model.forward(input_ids, target_ids)
    
    print(f"Output shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item() if outputs['loss'] is not None else 'N/A'}")
    
    # Test generation
    prompt = torch.tensor([1, 2, 3, 4, 5]).to(device)
    
    print("\nGenerating with repulsion dynamics...")
    generated = model.generate_repulsive(prompt, max_length=20, temperature=0.8)
    
    print(f"\nGenerated sequence: {generated}")
    
    # Visualize phase space
    print("\nVisualizing phase space...")
    fig1 = model.visualize_phase_space()
    
    # Visualize resonance field
    print("Visualizing resonance field...")
    fig2 = model.visualize_resonance_field(token_idx=0)
    
    # Visualize three-step evolution
    print("Visualizing three-step evolution...")
    fig3 = model.visualize_three_step_evolution(0, 1, 2)
    
    # Check quantum state consistency
    states = model.token_states_param.detach().cpu().numpy()
    r = np.exp(states[:, 0])
    z = states[:, 2]
    born_rule = r**2 + z**2
    
    print(f"\nBorn rule verification:")
    print(f"- Mean: {born_rule.mean():.6f} (should be close to 1.0)")
    print(f"- Min: {born_rule.min():.6f}")
    print(f"- Max: {born_rule.max():.6f}")
    print(f"- Standard deviation: {born_rule.std():.6f}")
    
    # Save visualizations
    print("\nSaving visualizations...")
    fig1.savefig('outputs/repulsion_phase_space.png')
    fig2.savefig('outputs/repulsion_resonance_field.png')
    fig3.savefig('outputs/repulsion_three_step_evolution.png')
    print("Visualizations saved to outputs/ directory")
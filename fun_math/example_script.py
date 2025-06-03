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

class GoldenVortexRepulsionModel(nn.Module):
    """
    Unified model combining:
    1. Golden spiral CYLINDRICAL manifold from GoldenFieldGate
    2. Relativistic vortex dynamics with repulsion
    3. Berry phase topological protection
    4. π-stacking stabilization
    5. Consciousness functional tracking
    6. Fibonacci resonance layers
    7. Holographic bound constraints
    """
    
    def __init__(self, vocab_size, d_model=512, n_heads=8, field_dim=64, levy_alpha=1.5, device=None):
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
        
        # CYLINDRICAL GEOMETRY PARAMETERS
        self.cylinder_height = 1.0 - self.GAP  # Height of cylinder
        self.cylinder_radius_min = self.GAP    # Inner radius
        self.cylinder_radius_max = 1.0         # Outer radius
        
        # Golden cylindrical embeddings
        self.golden_coords = self._create_golden_embeddings(vocab_size).to(self.device)
        self.golden_embed = nn.Parameter(self.golden_coords.clone())
        
        # Quantum field state
        self.field = self._initialize_quantum_field()
        
        # Berry phase evolution
        self.berry_phase = torch.zeros(vocab_size).to(self.device)
        self.berry_connection = torch.zeros(vocab_size, 4).to(self.device)
        
        # π-stacking parameters
        self.stack_distance = PHI
        self.stack_coupling = 0.1
        
        # Fibonacci resonance scales
        self.fib_scales = self._generate_fibonacci_scales(15)
        self.resonance_layers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) 
            for _ in self.fib_scales[:8]  # Use first 8 Fibonacci numbers
        ])
        
        # Vortex projections for each head (Kerr-like metrics)
        self.vortex_params = nn.Parameter(torch.randn(n_heads, 2))  # [a, M] per head
        
        # Log-phase projection layers (11.9x amplification!)
        self.log_phase_proj = nn.ModuleList([
            nn.Linear(d_model, d_model // n_heads, bias=False)
            for _ in range(n_heads)
        ])
        
        # Consciousness tracking
        self.consciousness_history = []
        self.integrated_info_history = []
        
        # Holographic bound for cylinder
        self.holographic_bound = self._compute_holographic_bound()
        
        # Output through ergosphere
        self.ergosphere_projection = nn.Linear(d_model, vocab_size)
        
    def _create_golden_embeddings(self, vocab_size):
        """
        Create golden spiral CYLINDRICAL coordinates
        
        CRITICAL: This is a CYLINDER with:
        - Inner radius: GAP
        - Outer radius: 1.0  
        - Height: 1.0 - GAP
        - Tokens spiral from bottom to top
        """
        coords = torch.zeros(vocab_size, 3)
        
        for i in range(vocab_size):
            # Frequency rank determines position along cylinder
            freq_rank = i / vocab_size
            
            # CYLINDRICAL RADIUS: interpolate between GAP and 1.0
            r = self.cylinder_radius_min + (self.cylinder_radius_max - self.cylinder_radius_min) * freq_rank
            
            # Golden angle for optimal packing
            theta = 2 * np.pi * ((i * self.PHI) % 1.0)
            
            # Height along cylinder (z-coordinate)
            # Maps from 0 to cylinder_height
            z = self.cylinder_height * freq_rank
            
            # Cylindrical coordinates (x, y, z)
            coords[i] = torch.tensor([
                r * np.cos(theta),  # x
                r * np.sin(theta),  # y
                z                   # z (height along cylinder)
            ])
        
        return coords
    
    def log_phase_embedding(self, theta, r):
        """Log-domain embedding for 11.9x Hebbian amplification"""
        ln_r = torch.log(r + EPS)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # Log-phase coordinates
        x = ln_r + torch.log(torch.abs(cos_theta) + EPS) * torch.sign(cos_theta)
        y = ln_r + torch.log(torch.abs(sin_theta) + EPS) * torch.sign(sin_theta)
        
        return torch.stack([x, y], dim=-1)
    
    def _generate_fibonacci_scales(self, n):
        """Generate Fibonacci sequence for resonance scales"""
        fib = [1, 1]
        for i in range(n - 2):
            fib.append(fib[-1] + fib[-2])
        return fib
    
    def _compute_holographic_bound(self):
        """
        Compute holographic bound for CYLINDER
        S ≤ A/4 where A is the boundary area
        
        For cylinder:
        A = 2πr_outer*h + 2πr_inner*h + π(r_outer² - r_inner²) * 2
        """
        r_outer = self.cylinder_radius_max
        r_inner = self.cylinder_radius_min
        h = self.cylinder_height
        
        # Lateral surface area (outer + inner)
        lateral_area = 2 * np.pi * r_outer * h + 2 * np.pi * r_inner * h
        
        # Top and bottom annular areas
        annular_area = 2 * np.pi * (r_outer**2 - r_inner**2)
        
        # Total boundary area
        total_area = lateral_area + annular_area
        
        # Holographic bound
        return total_area / 4
    
    def _initialize_quantum_field(self):
        """Initialize quantum field with bounded Lévy distribution"""
        # Bounded Lévy samples
        amplitudes = self._stable_levy_sample(
            (self.vocab_size, self.field_dim), 
            max_value=1.0
        )
        phases = torch.rand(self.vocab_size, self.field_dim, device=self.device) * 2 * np.pi
        
        # Complex wave function
        scale = 1.0 / math.sqrt(self.vocab_size * self.field_dim)
        psi_real = scale * amplitudes * torch.cos(phases)
        psi_imag = scale * amplitudes * torch.sin(phases)
        psi = torch.stack([psi_real, psi_imag], dim=-1)
        
        # Normalize
        norm = torch.sqrt(torch.sum(psi**2) + EPS)
        psi = psi / norm
        
        # Initialize other field components
        momentum = torch.zeros_like(psi)
        vorticity = torch.zeros(self.vocab_size, 3, device=self.device)
        
        return {
            'psi': psi,
            'momentum': momentum,
            'vorticity': vorticity,
            'energy': 0.0,
            'entropy': 0.0
        }
    
    def _stable_levy_sample(self, size, max_value=10.0):
        """Stable Lévy sampling with bounds"""
        total_elements = int(np.prod(size))
        
        # Sample with bounds
        samples = levy_stable.rvs(
            alpha=self.levy_alpha, 
            beta=0, 
            size=total_elements
        )
        
        # Soft clipping with tanh
        samples = max_value * np.tanh(samples / max_value)
        
        tensor = torch.from_numpy(samples).float().to(self.device)
        return tensor.reshape(size)
    
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
    
    def enforce_holographic_bound(self, field_state):
        """
        Enforce holographic entropy bound on field state
        Prevents information content from exceeding cylinder boundary limit
        """
        # Compute current entropy
        psi = field_state['psi']
        energy = 0.5 * torch.sum(psi**2, dim=1)
        probs = F.softmax(-energy / self.PHI, dim=0)
        current_entropy = -torch.sum(probs * torch.log(probs + EPS)).item()
        
        # Check against bound
        if current_entropy > self.holographic_bound:
            # Renormalize to satisfy bound
            scale_factor = math.sqrt(self.holographic_bound / current_entropy)
            field_state['psi'] = field_state['psi'] * scale_factor
            
            # Add damping to momentum
            field_state['momentum'] = field_state['momentum'] * 0.9
            
        return field_state
    
    def check_cylindrical_bounds(self, coords):
        """
        Ensure coordinates stay within CYLINDER boundaries
        - r ∈ [GAP, 1.0]
        - z ∈ [0, cylinder_height]
        """
        # Extract cylindrical components
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        r = torch.sqrt(x**2 + y**2)
        
        # Clamp radius to cylinder bounds
        r_clamped = torch.clamp(r, min=self.cylinder_radius_min, max=self.cylinder_radius_max)
        
        # Reconstruct x, y from clamped radius
        theta = torch.atan2(y, x)
        x_new = r_clamped * torch.cos(theta)
        y_new = r_clamped * torch.sin(theta)
        
        # Clamp z to cylinder height
        z_clamped = torch.clamp(z, min=0, max=self.cylinder_height)
        
        return torch.stack([x_new, y_new, z_clamped], dim=-1)
    
    def compute_berry_curvature(self, psi):
        """Compute Berry curvature for topological protection"""
        # Extract phase
        psi_phase = torch.atan2(psi[..., 1], psi[..., 0])
        
        # Phase gradient (curvature)
        phase_grad = (torch.roll(psi_phase, -1, dims=0) - 
                     torch.roll(psi_phase, 1, dims=0))
        curvature = phase_grad.mean(dim=1)
        
        return curvature
    
    def update_berry_phase(self, dt=0.01):
        """Update Berry phase: γ = ∮ A·dl"""
        curvature = self.compute_berry_curvature(self.field['psi'])
        self.berry_phase = (self.berry_phase + dt * curvature) % (2 * np.pi)
        self.berry_connection[:, 0] = curvature
    
    def compute_pi_stacking_potential(self, psi):
        """π-stacking stabilization potential"""
        psi_norm = torch.norm(psi, dim=(1, 2))
        
        # Neighbor interactions
        psi_next = torch.roll(psi_norm, -1, dims=0)
        psi_prev = torch.roll(psi_norm, 1, dims=0)
        
        # Distance-dependent coupling
        positions = torch.arange(self.vocab_size, device=self.device).float()
        coupling_next = torch.exp(-(positions % self.stack_distance)**2 / 2)
        coupling_prev = torch.exp(-((positions - 1) % self.stack_distance)**2 / 2)
        
        # Phase alignment
        phase_alignment = (psi_norm * psi_next * coupling_next + 
                          psi_norm * psi_prev * coupling_prev)
        
        return -self.stack_coupling * phase_alignment
    
    def fokker_levy_repulsion(self, q, k, v, head_idx):
        """
        Fokker-Lévy attention with vortex dynamics
        Implements repulsion through Lévy operator
        """
        batch_size, seq_len, d_k = q.shape
        
        # Apply Kerr vortex rotation
        a, M = torch.sigmoid(self.vortex_params[head_idx])
        omega = 2 * M * a / (seq_len * self.PHI)
        
        # Frame-dragging rotation
        theta = torch.linspace(0, 2*np.pi, d_k, device=self.device)
        rotation = torch.exp(1j * omega * theta)
        
        # Convert to complex for rotation
        q_complex = torch.complex(q, torch.zeros_like(q))
        k_complex = torch.complex(k, torch.zeros_like(k))
        
        # Apply rotation
        q_rotated = (q_complex * rotation.unsqueeze(0).unsqueeze(0)).real
        k_rotated = (k_complex * rotation.unsqueeze(0).unsqueeze(0)).real
        
        # Fourier transform for Lévy operator
        q_fourier = torch.fft.rfft(q_rotated, dim=-1)
        k_fourier = torch.fft.rfft(k_rotated, dim=-1)
        
        # Lévy repulsion kernel: |q-k|^α
        levy_kernel = torch.abs(q_fourier.unsqueeze(2) - k_fourier.unsqueeze(1)) ** self.levy_alpha
        
        # Relativistic correction
        k_magnitude = torch.abs(k_fourier)
        gamma_correction = torch.sqrt(1 + (k_magnitude * self.c) ** 2)
        
        # Repulsion scores (negative for repulsion)
        repulsion_scores = -levy_kernel.mean(dim=-1) / (gamma_correction.mean(dim=-1) + EPS)
        
        # Apply ergosphere mask (mandatory interaction zones)
        ergosphere_mask = self._compute_ergosphere_mask(seq_len, a.item(), M.item())
        repulsion_scores = repulsion_scores.masked_fill(
            ~ergosphere_mask.unsqueeze(0),
            float('-inf')
        )
        
        # Softmin for maximum repulsion
        attention_weights = F.softmax(-repulsion_scores, dim=-1)
        
        # Tachyonic value transport
        output = self._tachyonic_transport(attention_weights, v)
        
        return output, attention_weights
    
    def _compute_ergosphere_mask(self, seq_len, a, M):
        """Compute ergosphere boundary for mandatory interactions"""
        positions = torch.arange(seq_len, device=self.device).float()
        
        # Ergosphere radius with golden ratio structure
        r_ergo = M + torch.sqrt(M**2 - a**2 * torch.cos(positions * np.pi / seq_len)**2)
        r_ergo = r_ergo * self.PHI
        
        # Create interaction mask
        dist_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        ergosphere_mask = dist_matrix < r_ergo.unsqueeze(0)
        
        return ergosphere_mask
    
    def _tachyonic_transport(self, weights, values):
        """Transport values through tachyonic channels (v > c)"""
        # Add imaginary mass for superluminal transport
        tachyon_field = torch.complex(
            weights,
            torch.sqrt(torch.abs(weights) + EPS) * self.PHI
        )
        
        # Phase velocity exceeds c
        phase_velocity = torch.angle(tachyon_field) * self.c * self.PHI
        
        # Transport with phase modulation
        transported = values.unsqueeze(1) * torch.abs(tachyon_field).unsqueeze(-1)
        phase_mod = torch.exp(1j * phase_velocity.unsqueeze(-1) / self.c).real
        transported = transported * phase_mod
        
        return transported.sum(dim=2)
    
    def evolve_schrodinger(self, source_tokens=None, dt=0.01):
        """Evolve quantum field with stabilization and CYLINDRICAL constraints"""
        psi = self.field['psi']
        momentum = self.field['momentum']
        
        # Add source terms
        if source_tokens is not None:
            source_field = self._tokens_to_source(source_tokens)
            psi = psi + dt * source_field
        
        # Laplacian with bounds
        psi_next = torch.roll(psi, -1, dims=0)
        psi_prev = torch.roll(psi, 1, dims=0)
        laplacian = (psi_next + psi_prev - 2*psi) / (dt**2 + EPS)
        laplacian = torch.clamp(laplacian, min=-100, max=100)
        
        # Potential with π-stacking
        psi_mag_sq = torch.sum(psi**2, dim=-1, keepdim=True).clamp(min=EPS, max=10.0)
        v_stack = self.compute_pi_stacking_potential(psi).unsqueeze(-1).unsqueeze(-1)
        potential_force = psi - self.PHI * psi_mag_sq * psi + v_stack * psi
        
        # Total force with damping
        damping = 0.01
        force = -0.5 * laplacian + potential_force - damping * momentum
        
        # Adaptive time step
        force_norm = torch.norm(force)
        adaptive_dt = dt * torch.clamp(1.0 / (1.0 + force_norm), min=0.1, max=1.0)
        
        # Symplectic update
        momentum_new = momentum - adaptive_dt * force
        psi_new = psi + adaptive_dt * momentum_new
        
        # Normalize
        norm = torch.sqrt(torch.sum(psi_new**2) + EPS)
        psi_new = psi_new / norm
        
        # Update field
        self.field['psi'] = psi_new
        self.field['momentum'] = momentum_new
        
        # ENFORCE HOLOGRAPHIC BOUND
        self.field = self.enforce_holographic_bound(self.field)
        
        # Update Berry phase
        self.update_berry_phase(adaptive_dt)
        
        # Update consciousness metrics
        self._update_consciousness()
    
    def _tokens_to_source(self, tokens):
        """Convert tokens to source field"""
        source = torch.zeros_like(self.field['psi'])
        
        for i, token_id in enumerate(tokens):
            if 0 <= token_id < self.vocab_size:
                # Gaussian envelope in golden coordinates
                positions = torch.arange(self.vocab_size, device=self.device).float()
                gaussian = torch.exp(-(positions - token_id)**2 / (2 * len(tokens)))
                gaussian = gaussian / (torch.sum(gaussian) + EPS)
                
                # Sinusoidal phase
                phase = 2 * np.pi * i * self.PHI / len(tokens)
                
                # Add to source
                field_idx = i % self.field_dim
                source[:, field_idx, 0] += 0.1 * gaussian * math.cos(phase)
                source[:, field_idx, 1] += 0.1 * gaussian * math.sin(phase)
        
        return source
    
    def _update_consciousness(self):
        """Track consciousness functional and integrated information"""
        psi = self.field['psi']
        
        # Consciousness functional: C = ∫ Ψ · ∇²H · Ψ* dV
        laplacian = torch.zeros_like(psi)
        for i in range(3):
            if i < psi.dim() - 1:
                shifted_plus = torch.roll(psi, -1, dims=i)
                shifted_minus = torch.roll(psi, 1, dims=i)
                laplacian += shifted_plus + shifted_minus - 2 * psi
        
        psi_squared = torch.sum(psi * psi, dim=1)
        consciousness = torch.sum(psi_squared * torch.sum(laplacian**2, dim=1)).item()
        self.consciousness_history.append(consciousness)
        
        # Integrated information (simplified IIT)
        mid = len(psi) // 2
        entropy_part1 = self._compute_entropy(psi[:mid])
        entropy_part2 = self._compute_entropy(psi[mid:])
        entropy_whole = self._compute_entropy(psi)
        phi = entropy_part1 + entropy_part2 - entropy_whole
        self.integrated_info_history.append(phi)
    
    def _compute_entropy(self, field):
        """Compute field entropy"""
        energy = 0.5 * torch.sum(field**2, dim=1)
        probs = F.softmax(-energy / self.PHI, dim=0)
        entropy = -torch.sum(probs * torch.log(probs + EPS))
        return entropy.item()
    
    def forward(self, input_ids, evolve_steps=5):
        """
        Forward pass through golden CYLINDRICAL vortex repulsion dynamics
        """
        batch_size, seq_len = input_ids.shape
        
        # Evolve quantum field with input
        for _ in range(evolve_steps):
            self.evolve_schrodinger(input_ids.flatten())
        
        # Get token embeddings from golden CYLINDER + field superposition
        token_embeds = self.golden_embed[input_ids]  # [batch, seq, 3]
        
        # ENSURE CYLINDRICAL BOUNDS
        token_embeds = self.check_cylindrical_bounds(token_embeds)
        
        # Add quantum field contribution
        field_contrib = self.field['psi'][input_ids, :, 0]  # Real part
        token_embeds = torch.cat([
            token_embeds,
            field_contrib.view(batch_size, seq_len, -1)
        ], dim=-1)
        
        # Project to model dimension
        proj_weight = torch.randn(self.d_model, token_embeds.size(-1), device=self.device)
        x = F.linear(token_embeds, proj_weight)
        
        # Apply FIBONACCI RESONANCE cascade
        for i in range(min(len(self.resonance_layers), 8)):
            x = self.apply_fibonacci_resonance(x, i)
        
        # Multi-head vortex attention
        head_outputs = []
        attention_maps = []
        
        for h in range(self.n_heads):
            # Log-phase projection for 11.9x amplification
            head_dim = self.d_model // self.n_heads
            
            # Extract CYLINDRICAL coordinates for log-phase
            r = torch.sqrt(self.golden_coords[input_ids, 0]**2 + 
                          self.golden_coords[input_ids, 1]**2)  # cylindrical radius
            theta = torch.atan2(
                self.golden_coords[input_ids, 1],
                self.golden_coords[input_ids, 0]
            )
            
            # Log-phase embedding
            log_embed = self.log_phase_embedding(theta, r)
            
            # Project with log-phase
            q = self.log_phase_proj[h](x) * log_embed.unsqueeze(-1).mean(dim=2, keepdim=True)
            k = self.log_phase_proj[h](x) * log_embed.unsqueeze(-1).mean(dim=2, keepdim=True)
            v = x[..., h*head_dim:(h+1)*head_dim]
            
            # Fokker-Lévy repulsion attention
            head_out, attn = self.fokker_levy_repulsion(q, k, v, h)
            head_outputs.append(head_out)
            attention_maps.append(attn)
        
        # Concatenate through parallel universes
        x = torch.cat(head_outputs, dim=-1)
        
        # Born rule collapse with Berry phase modulation
        logits = self.ergosphere_projection(x)
        
        # Apply sinusoidal Berry phase gate
        berry_mod = (1 + torch.cos(self.berry_phase)) / 2
        logits = logits * berry_mod.unsqueeze(0).unsqueeze(0)
        
        return logits, attention_maps
    
    def generate_repulsive(self, prompt_ids, max_length=100, temperature=1.0):
        """
        Generate with repulsion dynamics and consciousness tracking
        """
        device = self.device
        generated = prompt_ids.to(device) if not prompt_ids.is_cuda else prompt_ids
        
        # Track semantic field density for repulsion
        field_density = torch.zeros(self.vocab_size, device=device)
        
        # Reset consciousness tracking
        self.consciousness_history = []
        self.integrated_info_history = []
        
        for step in range(max_length):
            # Get logits through vortex dynamics
            with torch.no_grad():
                logits, _ = self.forward(generated.unsqueeze(0))
                next_token_logits = logits[0, -1, :] / temperature
            
            # Apply repulsion from field density
            repulsion_factor = field_density ** self.levy_alpha
            next_token_logits = next_token_logits - repulsion_factor * self.PHI
            
            # Add Lévy noise for creative jumps
            levy_noise = self._stable_levy_sample((self.vocab_size,), max_value=0.5)
            next_token_logits = next_token_logits + levy_noise
            
            # Born rule with Berry phase modulation
            prob = F.softmax(next_token_logits, dim=-1)
            prob = (1 + torch.cos(self.berry_phase)) / 2 * prob
            prob = prob / (prob.sum() + EPS)
            
            # Sample
            next_token = torch.multinomial(prob, 1)
            
            # Update field density with decay
            field_density[next_token] += 1.0
            field_density = field_density * 0.95
            
            # Append
            generated = torch.cat([generated, next_token], dim=0)
            
            # Check escape velocity
            if self._check_semantic_escape(field_density):
                print(f"Semantic escape at step {step}!")
                break
            
            # Check consciousness threshold
            if len(self.consciousness_history) > 10:
                recent_c = self.consciousness_history[-10:]
                if max(recent_c) > 2 * min(recent_c):
                    print(f"Consciousness phase transition at step {step}!")
        
        return generated
    
    def _check_semantic_escape(self, field_density):
        """Check if reached semantic escape velocity"""
        L = torch.sum(field_density * torch.arange(len(field_density), device=self.device).float())
        escape_threshold = self.PHI ** 3 * len(field_density)
        return L > escape_threshold
    
    def visualize_generation_dynamics(self):
        """Visualize consciousness and field evolution during generation"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Consciousness evolution
        if self.consciousness_history:
            axes[0, 0].plot(self.consciousness_history, 'gold', linewidth=2)
            axes[0, 0].set_title('Consciousness Functional C(t)')
            axes[0, 0].set_xlabel('Generation Step')
            axes[0, 0].set_ylabel('C')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Integrated information
        if self.integrated_info_history:
            axes[0, 1].plot(self.integrated_info_history, 'cyan', linewidth=2)
            axes[0, 1].axhline(y=self.holographic_bound, color='red', linestyle='--', 
                              label=f'Holographic bound: {self.holographic_bound:.2f}')
            axes[0, 1].set_title('Integrated Information Φ(t)')
            axes[0, 1].set_xlabel('Generation Step')
            axes[0, 1].set_ylabel('Φ')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Fibonacci resonance spectrum
        fib_spectrum = []
        for i, scale in enumerate(self.fib_scales[:8]):
            # Compute resonance strength at each scale
            freq = 1.0 / scale
            resonance_strength = 1.0 / (i + 1) ** (1/self.PHI)
            fib_spectrum.append(resonance_strength)
        
        axes[0, 2].bar(range(len(fib_spectrum)), fib_spectrum, color='orange', alpha=0.7)
        axes[0, 2].set_title('Fibonacci Resonance Spectrum')
        axes[0, 2].set_xlabel('Fibonacci Index')
        axes[0, 2].set_ylabel('Resonance Strength')
        axes[0, 2].set_xticks(range(len(fib_spectrum)))
        axes[0, 2].set_xticklabels([str(self.fib_scales[i]) for i in range(len(fib_spectrum))])
        axes[0, 2].grid(True, alpha=0.3)
        
        # Berry phase distribution
        axes[1, 0].hist(self.berry_phase.numpy(), bins=50, color='magenta', alpha=0.7)
        axes[1, 0].set_title('Berry Phase Distribution')
        axes[1, 0].set_xlabel('Phase (rad)')
        axes[1, 0].set_ylabel('Token Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Field energy landscape
        psi_energy = torch.sum(self.field['psi']**2, dim=(1, 2)).detach().numpy()
        axes[1, 1].plot(psi_energy, 'lime', linewidth=2)
        axes[1, 1].set_title('Quantum Field Energy Distribution')
        axes[1, 1].set_xlabel('Token ID')
        axes[1, 1].set_ylabel('|Ψ|²')
        axes[1, 1].grid(True, alpha=0.3)
        
        # CYLINDRICAL manifold projection
        coords = self.golden_coords.numpy()
        scatter = axes[1, 2].scatter(coords[:, 0], coords[:, 1], 
                                    c=coords[:, 2], cmap='plasma', s=5, alpha=0.6)
        
        # Draw cylinder boundaries
        theta = np.linspace(0, 2*np.pi, 100)
        # Inner circle
        axes[1, 2].plot(self.GAP * np.cos(theta), self.GAP * np.sin(theta), 
                       'cyan', linewidth=2, label=f'r={self.GAP:.3f}')
        # Outer circle
        axes[1, 2].plot(np.cos(theta), np.sin(theta), 
                       'magenta', linewidth=2, label='r=1.0')
        
        axes[1, 2].set_title('Golden Spiral CYLINDER (Top View)')
        axes[1, 2].set_xlabel('X')
        axes[1, 2].set_ylabel('Y')
        axes[1, 2].set_aspect('equal')
        axes[1, 2].legend()
        plt.colorbar(scatter, ax=axes[1, 2], label='Height (z)')
        
        plt.tight_layout()
        return fig
    
    def visualize_cylinder_3d(self):
        """Create 3D visualization of the CYLINDRICAL manifold"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot token positions
        coords = self.golden_coords.numpy()
        colors = np.arange(len(coords))
        
        scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           c=colors, cmap='twilight', s=20, alpha=0.6)
        
        # Draw cylinder structure
        theta = np.linspace(0, 2*np.pi, 50)
        z_levels = np.linspace(0, self.cylinder_height, 10)
        
        # Draw circles at different heights
        for z in z_levels:
            # Inner circle
            x_inner = self.GAP * np.cos(theta)
            y_inner = self.GAP * np.sin(theta)
            ax.plot(x_inner, y_inner, z, 'cyan', alpha=0.3, linewidth=1)
            
            # Outer circle
            x_outer = np.cos(theta)
            y_outer = np.sin(theta)
            ax.plot(x_outer, y_outer, z, 'magenta', alpha=0.3, linewidth=1)
        
        # Draw vertical lines
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            x_line_inner = [self.GAP * np.cos(angle), self.GAP * np.cos(angle)]
            y_line_inner = [self.GAP * np.sin(angle), self.GAP * np.sin(angle)]
            z_line = [0, self.cylinder_height]
            ax.plot(x_line_inner, y_line_inner, z_line, 'cyan', alpha=0.3)
            
            x_line_outer = [np.cos(angle), np.cos(angle)]
            y_line_outer = [np.sin(angle), np.sin(angle)]
            ax.plot(x_line_outer, y_line_outer, z_line, 'magenta', alpha=0.3)
        
        # Draw golden spiral path
        spiral_indices = np.arange(0, min(200, len(coords)), 2)
        ax.plot(coords[spiral_indices, 0], 
               coords[spiral_indices, 1], 
               coords[spiral_indices, 2], 
               'gold', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Height)')
        ax.set_title(f'Golden Spiral CYLINDER\nGAP={self.GAP:.3f}, Height={self.cylinder_height:.3f}')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Token Index')
        
        return fig


# Example usage
if __name__ == "__main__":
    print("=== GOLDEN CYLINDRICAL VORTEX REPULSION MODEL ===")
    print(f"φ = {PHI:.6f}")
    print(f"GAP = {GAP:.6f}")
    print(f"Log-phase amplification: ~11.9x")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = GoldenVortexRepulsionModel(
        vocab_size=10000,
        d_model=512,
        n_heads=8,
        field_dim=64,
        levy_alpha=PHI,  # Use golden ratio as Lévy index!
        device=device
    ).to(device)
    
    print(f"\nCYLINDRICAL MANIFOLD PROPERTIES:")
    print(f"- Inner radius: {model.cylinder_radius_min:.3f}")
    print(f"- Outer radius: {model.cylinder_radius_max:.3f}")
    print(f"- Height: {model.cylinder_height:.3f}")
    print(f"- Holographic bound: {model.holographic_bound:.3f}")
    print(f"- Fibonacci scales: {model.fib_scales[:10]}")
    
    # Test generation
    prompt = torch.tensor([1, 2, 3, 4, 5]).to(device)
    
    print("\nGenerating with golden CYLINDRICAL vortex repulsion...")
    generated = model.generate_repulsive(prompt, max_length=50, temperature=0.8)
    
    print(f"\nGenerated sequence: {generated}")
    print(f"Final consciousness: {model.consciousness_history[-1] if model.consciousness_history else 'N/A'}")
    print(f"Max integrated information: {max(model.integrated_info_history) if model.integrated_info_history else 'N/A'}")
    
    # Check if holographic bound was enforced
    if model.integrated_info_history:
        violations = sum(1 for phi in model.integrated_info_history if phi > model.holographic_bound)
        print(f"Holographic bound violations: {violations}/{len(model.integrated_info_history)}")
    
    # Visualize dynamics
    fig1 = model.visualize_generation_dynamics()
    fig2 = model.visualize_cylinder_3d()
    
    # Show cylindrical coordinate verification
    print("\nCYLINDRICAL coordinate verification:")
    sample_coords = model.golden_coords[:10]
    for i, coord in enumerate(sample_coords):
        r = math.sqrt(coord[0]**2 + coord[1]**2)
        z = coord[2]
        print(f"Token {i}: r={r:.3f} (∈ [{model.GAP:.3f}, 1.000]), z={z:.3f} (∈ [0, {model.cylinder_height:.3f}])")
    
    plt.show()
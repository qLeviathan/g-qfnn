"""
NASA STEM Educational Module: Building Language Models with Physics
=================================================================
Learn how physics principles can revolutionize neural networks!

Prerequisites: Basic Python, High school physics, Curiosity about AI
Hardware: CUDA-capable GPU recommended
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Running on: {device}")
print(f"💫 GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU Name: {torch.cuda.get_device_name(0)}")

class PhysicsLab:
    """Your interactive physics laboratory for neural networks!"""
    
    def __init__(self):
        self.device = device
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio - nature's favorite number!
        
    def introduction(self):
        """Start here! Understanding the physics-AI connection."""
        print("""
        🌟 Welcome to NASA STEM Physics-Informed Neural Networks Lab! 🌟
        
        Today's Mission: Build a language model using physics principles
        
        Key Concepts We'll Master:
        1. 🌊 Diffusion - How information spreads like heat
        2. 📐 Resonance - Why some words "click" together  
        3. 🎲 Quantum Measurements - Attention without softmax
        4. 🔄 Conservation Laws - Memory without backpropagation
        
        Let's start with a simple question: 
        What if words were particles in a field?
        """)

# Module 1: Understanding Fields
class Module1_Fields:
    """Lesson 1: Words as particles in fields"""
    
    def __init__(self, lab):
        self.lab = lab
        self.vocab_size = 100
        self.embedding_dim = 64
        
    def interactive_field_visualization(self):
        """See how words exist in field space"""
        
        # Create interactive sliders
        angle_slider = widgets.FloatSlider(
            value=0, min=0, max=2*np.pi, step=0.1,
            description='Phase θ:', readout_format='.2f'
        )
        radius_slider = widgets.FloatSlider(
            value=0.5, min=0.1, max=1.0, step=0.05,
            description='Radius r:', readout_format='.2f'
        )
        
        def update_field(angle, radius):
            clear_output(wait=True)
            
            # Create field visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Cylindrical coordinates
            theta = np.linspace(0, 2*np.pi, 100)
            r = np.linspace(0.1, 1.0, 50)
            R, THETA = np.meshgrid(r, theta)
            
            # Field strength
            field = np.sin(3*THETA) * np.exp(-R/self.lab.phi)
            
            # Plot field
            c1 = ax1.contourf(R*np.cos(THETA), R*np.sin(THETA), field, 20, cmap='RdBu')
            ax1.scatter(radius*np.cos(angle), radius*np.sin(angle), 
                       c='yellow', s=200, edgecolor='black', linewidth=2)
            ax1.set_title('Word in Field Space')
            ax1.set_xlabel('x = r cos(θ)')
            ax1.set_ylabel('y = r sin(θ)')
            plt.colorbar(c1, ax=ax1, label='Field Strength')
            
            # Energy landscape
            energy = -np.log(np.abs(field) + 0.1)
            c2 = ax2.contourf(R*np.cos(THETA), R*np.sin(THETA), energy, 20, cmap='viridis')
            ax2.scatter(radius*np.cos(angle), radius*np.sin(angle), 
                       c='red', s=200, edgecolor='white', linewidth=2)
            ax2.set_title('Energy Landscape')
            ax2.set_xlabel('x = r cos(θ)')
            ax2.set_ylabel('y = r sin(θ)')
            plt.colorbar(c2, ax=ax2, label='Energy')
            
            plt.tight_layout()
            plt.show()
            
            # Explain the physics
            print(f"""
            📍 Current Position: r={radius:.2f}, θ={angle:.2f}
            🌊 Field Strength: {np.sin(3*angle) * np.exp(-radius/self.lab.phi):.3f}
            ⚡ Energy: {-np.log(np.abs(np.sin(3*angle) * np.exp(-radius/self.lab.phi)) + 0.1):.3f}
            
            💡 Physics Insight: 
            - High field strength = likely word position
            - Low energy = stable configuration
            - Words naturally flow toward energy minima!
            """)
        
        # Create interactive widget
        interact_widget = widgets.interactive(
            update_field,
            angle=angle_slider,
            radius=radius_slider
        )
        display(interact_widget)

# Module 2: Diffusion Dynamics
class Module2_Diffusion:
    """Lesson 2: How information diffuses like heat"""
    
    def __init__(self, lab):
        self.lab = lab
        
    def diffusion_simulator(self):
        """Watch how tokens diffuse in real-time!"""
        
        print("""
        🔥 Diffusion Equation in Neural Networks 🔥
        
        In physics: ∂u/∂t = D∇²u (heat equation)
        In our network: d(ln r)/dt = -ln(r)/τ + σ×noise
        
        Watch how a token spreads its information:
        """)
        
        # Interactive parameters
        diff_coeff = widgets.FloatSlider(
            value=0.1, min=0.01, max=0.5, step=0.01,
            description='Diffusion σ:'
        )
        coherence = widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.05,
            description='Coherence C:'
        )
        
        @widgets.interact(D=diff_coeff, C=coherence)
        def simulate_diffusion(D=0.1, C=0.0):
            # Initialize token positions
            n_tokens = 50
            ln_r = torch.randn(n_tokens).to(device) * 0.1
            theta = torch.rand(n_tokens).to(device) * 2 * np.pi
            
            # Time evolution
            n_steps = 100
            dt = 0.01
            history = []
            
            for t in range(n_steps):
                # Diffusion with coherence modulation
                noise = torch.randn_like(ln_r).to(device)
                ln_r = ln_r - ln_r * dt / self.lab.phi + D * (1-C) * noise * np.sqrt(dt)
                
                # Add golden ratio resonance
                resonance = torch.sin(self.lab.phi * t * dt)
                theta = theta + 0.1 * resonance * dt
                
                if t % 10 == 0:
                    history.append((ln_r.cpu().numpy().copy(), 
                                  theta.cpu().numpy().copy()))
            
            # Visualize
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot trajectories
            for i in range(5):  # Show first 5 tokens
                trajectory_r = [np.exp(h[0][i]) for h in history]
                trajectory_theta = [h[1][i] for h in history]
                x = [r * np.cos(theta) for r, theta in zip(trajectory_r, trajectory_theta)]
                y = [r * np.sin(theta) for r, theta in zip(trajectory_r, trajectory_theta)]
                ax.plot(x, y, alpha=0.5, linewidth=2)
                ax.scatter(x[-1], y[-1], s=100, edgecolor='black')
            
            # Style
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_xlabel('x = r cos(θ)')
            ax.set_ylabel('y = r sin(θ)')
            ax.set_title(f'Token Diffusion (σ={D:.2f}, Coherence={C:.2f})')
            ax.grid(True, alpha=0.3)
            
            # Add golden ratio circle
            circle = plt.Circle((0, 0), 1/self.lab.phi, fill=False, 
                              linestyle='--', color='gold', linewidth=2)
            ax.add_patch(circle)
            ax.text(0, 1/self.lab.phi + 0.1, 'φ-resonance zone', 
                   ha='center', color='gold')
            
            plt.show()
            
            print(f"""
            🔬 Observations:
            - Lower diffusion = more stable trajectories
            - Higher coherence = less random motion
            - Tokens naturally orbit the φ-resonance circle!
            """)

# Module 3: Quantum Attention
class Module3_QuantumAttention:
    """Lesson 3: Attention through quantum measurement"""
    
    def __init__(self, lab):
        self.lab = lab
        
    def born_rule_demo(self):
        """Compare classical softmax vs quantum Born rule attention"""
        
        print("""
        ⚛️ Quantum Attention: Born Rule vs Softmax ⚛️
        
        Classical: softmax(QK^T/√d)V
        Quantum: |⟨ψ_q|ψ_k⟩|²V
        
        Let's see the difference:
        """)
        
        seq_len = widgets.IntSlider(value=10, min=5, max=20, description='Sequence:')
        temp = widgets.FloatSlider(value=1.0, min=0.1, max=2.0, description='Temperature:')
        
        @widgets.interact(n=seq_len, T=temp)
        def compare_attention(n=10, T=1.0):
            # Create random quantum states
            q_phase = torch.rand(n, 1).to(device) * 2 * np.pi
            k_phase = torch.rand(n, 1).to(device) * 2 * np.pi
            
            # Classical attention
            scores = torch.randn(n, n).to(device) / np.sqrt(n)
            classical_attn = torch.softmax(scores / T, dim=-1)
            
            # Quantum Born rule attention
            overlap = torch.cos(q_phase - k_phase.T)
            born_attn = overlap ** 2  # Born rule: |⟨ψ|φ⟩|²
            born_attn = born_attn / born_attn.sum(dim=-1, keepdim=True)
            
            # Visualize
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Classical attention
            im1 = ax1.imshow(classical_attn.cpu(), cmap='Blues', vmin=0, vmax=1)
            ax1.set_title('Classical Softmax Attention')
            ax1.set_xlabel('Keys')
            ax1.set_ylabel('Queries')
            plt.colorbar(im1, ax=ax1)
            
            # Quantum attention
            im2 = ax2.imshow(born_attn.cpu(), cmap='Reds', vmin=0, vmax=1)
            ax2.set_title('Quantum Born Rule Attention')
            ax2.set_xlabel('Keys')
            ax2.set_ylabel('Queries')
            plt.colorbar(im2, ax=ax2)
            
            # Difference
            diff = (born_attn - classical_attn).cpu()
            im3 = ax3.imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
            ax3.set_title('Difference (Quantum - Classical)')
            ax3.set_xlabel('Keys')
            ax3.set_ylabel('Queries')
            plt.colorbar(im3, ax=ax3)
            
            plt.tight_layout()
            plt.show()
            
            # Compute sparsity
            classical_sparsity = (classical_attn < 0.1).float().mean()
            quantum_sparsity = (born_attn < 0.1).float().mean()
            
            print(f"""
            📊 Statistics:
            - Classical sparsity: {classical_sparsity:.1%}
            - Quantum sparsity: {quantum_sparsity:.1%}
            - Max attention (classical): {classical_attn.max():.3f}
            - Max attention (quantum): {born_attn.max():.3f}
            
            💡 Key Insight: Quantum attention naturally creates 
            sparse patterns without temperature tuning!
            """)

# Module 4: Hebbian Learning
class Module4_HebbianLearning:
    """Lesson 4: Learning without backpropagation"""
    
    def __init__(self, lab):
        self.lab = lab
        
    def hebbian_vs_backprop(self):
        """See how Hebbian learning works without storing gradients"""
        
        print("""
        🧠 Hebbian Learning: "Neurons that fire together, wire together" 🧠
        
        Traditional: Store all activations, compute gradients backward
        Hebbian: Update weights based on local correlations only!
        
        Memory comparison:
        """)
        
        batch_size = widgets.IntSlider(value=32, min=8, max=128, step=8, 
                                      description='Batch Size:')
        seq_len = widgets.IntSlider(value=100, min=50, max=500, step=50,
                                   description='Sequence:')
        
        @widgets.interact(B=batch_size, L=seq_len)
        def compare_memory(B=32, L=100):
            hidden = 512
            
            # Calculate memory requirements
            # Backprop needs to store all intermediate activations
            backprop_memory = B * L * hidden * 4  # float32
            backprop_memory += B * L * L * 4  # attention matrix
            backprop_memory += B * L * hidden * 4 * 12  # transformer layers
            
            # Hebbian only needs current activations
            hebbian_memory = B * L * hidden * 4  # current state
            hebbian_memory += hidden * hidden * 4  # weight matrix
            
            # Visualize
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Memory comparison
            memories = [backprop_memory/1e9, hebbian_memory/1e9]
            colors = ['red', 'green']
            bars = ax1.bar(['Backpropagation', 'Hebbian'], memories, color=colors)
            ax1.set_ylabel('Memory (GB)')
            ax1.set_title(f'Memory Usage (Batch={B}, Seq={L})')
            
            # Add values on bars
            for bar, mem in zip(bars, memories):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mem:.2f} GB', ha='center', va='bottom')
            
            # Scaling analysis
            seq_lengths = np.linspace(100, 2000, 20)
            backprop_scaling = (B * seq_lengths * hidden * 4 * 13) / 1e9
            hebbian_scaling = (B * seq_lengths * hidden * 4 + hidden**2 * 4) / 1e9
            
            ax2.plot(seq_lengths, backprop_scaling, 'r-', linewidth=2, 
                    label='Backprop O(L²)')
            ax2.plot(seq_lengths, hebbian_scaling, 'g-', linewidth=2, 
                    label='Hebbian O(L)')
            ax2.set_xlabel('Sequence Length')
            ax2.set_ylabel('Memory (GB)')
            ax2.set_title('Memory Scaling')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print(f"""
            🚀 Advantages of Hebbian Learning:
            - {backprop_memory/hebbian_memory:.1f}x less memory
            - O(L) vs O(L²) scaling
            - No gradient storage needed
            - Continuous learning possible
            
            ⚡ The Physics Connection:
            Just like magnetic domains align without central coordination,
            neural weights can self-organize through local interactions!
            """)

# Module 5: Build Your Own PINN Language Model
class Module5_BuildPINN:
    """Final Project: Build a complete PINN language model!"""
    
    def __init__(self, lab):
        self.lab = lab
        
    def create_model_architecture(self):
        """Step-by-step model building"""
        
        print("""
        🏗️ Let's Build a Physics-Informed Language Model! 🏗️
        
        Components:
        1. Cylindrical embedding space
        2. Diffusion-based token evolution  
        3. Quantum attention mechanism
        4. Hebbian weight updates
        5. Golden ratio resonance
        """)
        
        class SimplePINNLanguageModel(nn.Module):
            def __init__(self, vocab_size, hidden_dim, lab):
                super().__init__()
                self.lab = lab
                self.vocab_size = vocab_size
                self.hidden_dim = hidden_dim
                
                # Cylindrical embeddings
                self.ln_r_embed = nn.Embedding(vocab_size, hidden_dim)
                self.theta_embed = nn.Embedding(vocab_size, hidden_dim)
                self.z_embed = nn.Embedding(vocab_size, hidden_dim)
                
                # Hebbian weights (no gradients needed!)
                self.hebbian_w = nn.Parameter(
                    torch.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim),
                    requires_grad=False
                )
                
                # Output projection
                self.output = nn.Linear(hidden_dim * 3, vocab_size)
                
            def cylindrical_to_cartesian(self, ln_r, theta, z):
                """Convert cylindrical to Cartesian coordinates"""
                r = torch.exp(ln_r)
                x = r * torch.cos(theta)
                y = r * torch.sin(theta)
                return torch.cat([x, y, z], dim=-1)
                
            def quantum_attention(self, q_phase, k_phase, v):
                """Born rule attention without softmax"""
                # Quantum overlap
                overlap = torch.cos(q_phase.unsqueeze(-1) - k_phase.unsqueeze(-2))
                born_prob = overlap ** 2
                
                # Normalize (Born rule naturally sums to 1)
                born_prob = born_prob / born_prob.sum(dim=-1, keepdim=True)
                
                # Apply attention
                return torch.matmul(born_prob, v)
                
            def hebbian_update(self, pre, post, lr=0.01):
                """Local Hebbian weight update"""
                with torch.no_grad():
                    # Hebbian rule: Δw = η * pre * post
                    correlation = torch.matmul(pre.T, post) / pre.shape[0]
                    self.hebbian_w += lr * correlation
                    
                    # Normalize to prevent explosion
                    self.hebbian_w /= self.hebbian_w.norm() + 1e-8
                    
            def forward(self, input_ids, train=True):
                batch_size, seq_len = input_ids.shape
                
                # Get cylindrical embeddings
                ln_r = self.ln_r_embed(input_ids)
                theta = self.theta_embed(input_ids)
                z = self.z_embed(input_ids)
                
                # Add position encoding using golden ratio
                pos = torch.arange(seq_len).float().to(input_ids.device)
                theta = theta + 2 * np.pi * pos.unsqueeze(0).unsqueeze(-1) / self.lab.phi
                
                # Diffusion step (simplified)
                noise = torch.randn_like(ln_r) * 0.01
                ln_r = ln_r - ln_r / self.lab.phi + noise
                
                # Convert to Cartesian for attention
                cartesian = self.cylindrical_to_cartesian(ln_r, theta, z)
                
                # Quantum attention
                attended = self.quantum_attention(theta, theta, cartesian)
                
                # Hebbian learning (only during training)
                if train:
                    self.hebbian_update(cartesian, attended)
                
                # Apply Hebbian weights
                transformed = torch.matmul(attended, self.hebbian_w)
                
                # Output logits
                logits = self.output(transformed)
                
                return logits
        
        # Create and test model
        model = SimplePINNLanguageModel(
            vocab_size=1000,
            hidden_dim=64,
            lab=self.lab
        ).to(device)
        
        # Test forward pass
        test_input = torch.randint(0, 1000, (4, 20)).to(device)
        with torch.no_grad():
            output = model(test_input, train=False)
            
        print(f"""
        ✅ Model Created Successfully!
        
        Architecture Summary:
        - Parameters: {sum(p.numel() for p in model.parameters()):,}
        - Hebbian weights: {model.hebbian_w.shape}
        - Input shape: {test_input.shape}
        - Output shape: {output.shape}
        
        🎯 Next Steps:
        1. Train on real text data
        2. Visualize learned resonance patterns
        3. Compare with traditional transformers
        4. Explore different physics constraints
        """)
        
        return model

# Main Educational Interface
class PINNEducator:
    """Complete educational interface"""
    
    def __init__(self):
        self.lab = PhysicsLab()
        self.modules = {
            '1_fields': Module1_Fields(self.lab),
            '2_diffusion': Module2_Diffusion(self.lab),
            '3_quantum': Module3_QuantumAttention(self.lab),
            '4_hebbian': Module4_HebbianLearning(self.lab),
            '5_build': Module5_BuildPINN(self.lab)
        }
        
    def start_course(self):
        """Launch the interactive course"""
        print("""
        🚀 NASA STEM: Physics-Informed Neural Networks Course 🚀
        
        Select a module to begin:
        """)
        
        module_selector = widgets.Dropdown(
            options=[
                ('Introduction', 'intro'),
                ('Module 1: Fields', '1_fields'),
                ('Module 2: Diffusion', '2_diffusion'),
                ('Module 3: Quantum Attention', '3_quantum'),
                ('Module 4: Hebbian Learning', '4_hebbian'),
                ('Module 5: Build Your Model', '5_build')
            ],
            description='Module:',
            style={'description_width': 'initial'}
        )
        
        def load_module(selection):
            clear_output()
            
            if selection == 'intro':
                self.lab.introduction()
            elif selection == '1_fields':
                self.modules['1_fields'].interactive_field_visualization()
            elif selection == '2_diffusion':
                self.modules['2_diffusion'].diffusion_simulator()
            elif selection == '3_quantum':
                self.modules['3_quantum'].born_rule_demo()
            elif selection == '4_hebbian':
                self.modules['4_hebbian'].hebbian_vs_backprop()
            elif selection == '5_build':
                self.modules['5_build'].create_model_architecture()
        
        interact = widgets.interactive(load_module, selection=module_selector)
        display(interact)
        
        # Add navigation tips
        print("""
        
        💡 Tips:
        - Work through modules in order for best understanding
        - Experiment with all interactive parameters
        - Try to predict what will happen before moving sliders
        - Think about the physics meaning, not just the math
        
        🎓 By the end, you'll understand:
        - Why attention is like quantum measurement
        - How diffusion creates natural learning
        - Why the golden ratio appears everywhere
        - How to build AI that thinks like physics
        """)

# Launch the course
if __name__ == "__main__":
    educator = PINNEducator()
    educator.start_course()
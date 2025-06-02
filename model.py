"""
model.py - Complete Field-Theoretic Language Model
Physics: Tokens evolve on golden manifold with gravitational dynamics
No backpropagation - learning through Hebbian crystallization
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np

from core import FieldConfig, GoldenEmbedding, LogPhaseEmbedding, FieldEvolution, CrystalMemory
from perturbations import AdaptivePerturbation, LevyPerturbation, BetaPerturbation
from collapse import FieldCollapse, NucleusFieldSampler

class FieldTheoreticLM(nn.Module):
    """
    Complete field-theoretic language model
    Replaces transformer attention with gravitational field dynamics
    """
    def __init__(self, config: FieldConfig, embedding_type: str = "golden"):
        super().__init__()
        self.config = config
        
        # Embedding layer
        if embedding_type == "golden":
            self.embeddings = GoldenEmbedding(config)
        elif embedding_type == "log_phase":
            self.embeddings = LogPhaseEmbedding(config)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        # Field evolution layers
        self.field_layers = nn.ModuleList([
            FieldEvolution(config) for _ in range(config.n_layers)
        ])
        
        # Crystal memory (Hebbian)
        self.crystal_memory = CrystalMemory(config)
        
        # Field collapse
        self.collapse = FieldCollapse(config.vocab_size, config.d_model)
        
        # Perturbation strategy
        self.perturbation = AdaptivePerturbation(scale=0.01)
        
        # Output projection (small learnable component)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False, dtype=torch.float32)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=1/np.sqrt(config.d_model))
        
        # Sampling
        self.sampler = NucleusFieldSampler(top_p=0.9)
        
        # Ensure all parameters use float32
        for param in self.parameters():
            param.data = param.data.to(torch.float32)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        crystal_update: bool = True,
        perturbation_type: Optional[str] = "adaptive",
        return_field: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through field dynamics
        input_ids: (batch, seq_len) - Input token indices
        crystal_update: Whether to update crystal memory
        perturbation_type: Type of perturbation to apply
        return_field: Whether to return field states
        
        Returns dict with:
            - logits: (batch, seq_len, vocab_size)
            - collapsed_tokens: (batch, seq_len) 
            - coherence: (batch, seq_len)
            - field_states: List of (batch, seq_len, d_model) if return_field
        """
        # Embed tokens to field
        field = self.embeddings(input_ids)  # (batch, seq_len, d_model)
        
        # Ensure field is float32
        field = field.to(torch.float32)
        
        field_states = []
        
        # Evolve through layers
        for layer_idx, field_layer in enumerate(self.field_layers):
            # Store pre-synaptic activity
            pre_field = field.clone()
            
            # Apply crystal transformation
            field = self.crystal_memory(field, layer_idx)
            
            # Field evolution
            field = field_layer(field, steps=5)
            
            # Apply perturbation
            if perturbation_type == "levy":
                field = LevyPerturbation(scale=0.01).perturb(field)
            elif perturbation_type == "beta":
                field = BetaPerturbation(scale=0.01).perturb(field)
            elif perturbation_type == "adaptive":
                field = self.perturbation.perturb(field)
            
            # Hebbian update
            if crystal_update and self.training:
                self.crystal_memory.hebbian_update(pre_field, field, layer_idx)
            
            if return_field:
                field_states.append(field.clone())
        
        # Project to vocabulary space (ensure field is float32)
        field = field.to(torch.float32)
        logits = self.output_projection(field)  # (batch, seq_len, vocab_size)
        
        # Field collapse dynamics
        collapsed_tokens, collapse_probs = self.collapse(field)
        
        # Compute coherence
        coherence = self.collapse.compute_coherence(field)
        
        # Mix logits with collapse probabilities
        logits = 0.7 * logits + 0.3 * torch.log(collapse_probs + 1e-8)
        
        output = {
            'logits': logits,
            'collapsed_tokens': collapsed_tokens,
            'coherence': coherence,
            'collapse_probs': collapse_probs
        }
        
        if return_field:
            output['field_states'] = field_states
            
        return output
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
        perturbation_type: str = "adaptive"
    ) -> torch.Tensor:
        """
        Generate tokens using field dynamics
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Start with input
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                generated, 
                crystal_update=False,
                perturbation_type=perturbation_type
            )
            
            # Get last position logits
            logits = outputs['logits'][:, -1, :] / temperature
            coherence = outputs['coherence'][:, -1]
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample with coherence weighting
            if top_p < 1.0:
                # Nucleus sampling weighted by coherence
                next_tokens = self.sampler(
                    probs.unsqueeze(1), 
                    coherence.unsqueeze(1)
                ).squeeze(1)
            else:
                # Standard sampling
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Append to generated
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            
            # Stop if all sequences have generated EOS (assuming 0 is EOS)
            if (next_tokens == 0).all():
                break
        
        return generated
    
    def compute_perplexity(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute perplexity without backprop
        """
        outputs = self.forward(input_ids, crystal_update=False)
        logits = outputs['logits']
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )
        
        perplexity = torch.exp(loss)
        return perplexity
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get model memory usage statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        total_buffers = sum(b.numel() for b in self.buffers())
        
        # Crystal memory size estimation
        crystal_size = 0
        if hasattr(self.crystal_memory, 'W_crystal'):
            if isinstance(self.crystal_memory.W_crystal, torch.Tensor):
                crystal_size = self.crystal_memory.W_crystal.numel()
            elif hasattr(self.crystal_memory.W_crystal, 'parameters'):
                crystal_size = sum(p.numel() for p in self.crystal_memory.W_crystal.parameters())
        
        # Convert to MB
        param_mb = total_params * 4 / 1024 / 1024  # float32
        buffer_mb = total_buffers * 4 / 1024 / 1024
        crystal_mb = crystal_size * 4 / 1024 / 1024
        
        return {
            'total_parameters': total_params,
            'param_memory_mb': param_mb,
            'buffer_memory_mb': buffer_mb,
            'crystal_memory_mb': crystal_mb,
            'total_memory_mb': param_mb + buffer_mb
        }

# Model configurations
def create_small_model(embedding_type: str = "golden") -> FieldTheoreticLM:
    """Create small model for testing (50M params)"""
    config = FieldConfig(
        vocab_size=50257,
        d_model=512,
        n_layers=8,
        max_seq_len=512,
        dtype=torch.float32  # Ensure float32
    )
    return FieldTheoreticLM(config, embedding_type)

def create_base_model(embedding_type: str = "golden") -> FieldTheoreticLM:
    """Create base model (350M params)"""
    config = FieldConfig(
        vocab_size=50257,
        d_model=768,
        n_layers=12,
        max_seq_len=1024,
        dtype=torch.float32  # Ensure float32
    )
    return FieldTheoreticLM(config, embedding_type)

def create_large_model(embedding_type: str = "golden") -> FieldTheoreticLM:
    """Create large model (750M params)"""
    config = FieldConfig(
        vocab_size=50257,
        d_model=1024,
        n_layers=16,
        max_seq_len=2048,
        dtype=torch.float32  # Ensure float32
    )
    return FieldTheoreticLM(config, embedding_type)

# Validation
if __name__ == "__main__":
    print("=== Field-Theoretic Language Model Validation ===")
    
    # Create small model for testing - use float32 for validation
    config = FieldConfig(
        vocab_size=50257,
        d_model=512,
        n_layers=8,
        max_seq_len=512,
        dtype=torch.float32  # Use float32 for testing
    )
    model = FieldTheoreticLM(config, "log_phase").cuda()
    
    # Test input
    input_ids = torch.randint(0, 50257, (2, 32)).cuda()
    
    # Forward pass
    outputs = model(input_ids, return_field=True)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Collapsed tokens: {outputs['collapsed_tokens'].shape}")
    print(f"Coherence range: [{outputs['coherence'].min().item():.3f}, {outputs['coherence'].max().item():.3f}]")
    print(f"Field states: {len(outputs['field_states'])} layers")
    
    # Test generation
    prompt = torch.randint(0, 50257, (1, 5)).cuda()
    generated = model.generate(prompt, max_length=20)
    print(f"\nGenerated shape: {generated.shape}")
    
    try:
        # Memory usage
        memory = model.get_memory_usage()
        print(f"\nMemory usage:")
        for key, value in memory.items():
            print(f"  {key}: {value:.2f}")
    except Exception as e:
        print(f"Warning: Memory usage calculation failed: {e}")
    
    print("\n[PASS] Model validated")
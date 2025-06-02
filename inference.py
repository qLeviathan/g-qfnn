"""
inference.py - Inference utilities and analysis tools
Includes generation, field visualization, and performance profiling
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path

from model import FieldTheoreticLM
from transformers import AutoTokenizer

class FieldInference:
    """
    Inference engine for field-theoretic language models
    Handles generation, analysis, and visualization
    """
    def __init__(
        self,
        model: FieldTheoreticLM,
        tokenizer_name: str = "gpt2",
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        num_samples: int = 1,
        perturbation_type: str = "adaptive"
    ) -> List[str]:
        """Generate text from prompt"""
        self.model.eval()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        generated_texts = []
        
        for _ in range(num_samples):
            # Generate
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                perturbation_type=perturbation_type
            )
            
            # Decode
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
            
        return generated_texts
    
    @torch.no_grad()
    def analyze_field_dynamics(
        self,
        text: str,
        return_evolution: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Analyze field dynamics for given text"""
        self.model.eval()
        
        # Tokenize
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # Forward with field tracking
        outputs = self.model(input_ids, crystal_update=False, return_field=True)
        
        # Extract dynamics
        analysis = {
            "tokens": input_ids[0],
            "coherence": outputs["coherence"][0],
            "collapsed_tokens": outputs["collapsed_tokens"][0],
            "final_logits": outputs["logits"][0]
        }
        
        if return_evolution:
            # Stack field states
            field_evolution = torch.stack(outputs["field_states"])  # (n_layers, seq_len, d_model)
            analysis["field_evolution"] = field_evolution
            
            # Compute layer-wise metrics
            layer_coherence = []
            layer_entropy = []
            
            for layer_field in field_evolution:
                # Coherence
                coh = self.model.collapse.compute_coherence(layer_field.unsqueeze(0))
                layer_coherence.append(coh[0])
                
                # Entropy
                energy = 0.5 * torch.sum(layer_field**2, dim=-1)
                probs = torch.softmax(-energy / 1.618, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                layer_entropy.append(entropy)
                
            analysis["layer_coherence"] = torch.stack(layer_coherence)
            analysis["layer_entropy"] = torch.stack(layer_entropy)
            
        return analysis
    
    def visualize_field_evolution(
        self,
        text: str,
        save_path: Optional[str] = None
    ):
        """Visualize field evolution through layers"""
        analysis = self.analyze_field_dynamics(text)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Field Evolution: "{text[:50]}..."', fontsize=14)
        
        # Token display
        tokens = [self.tokenizer.decode([t.item()]) for t in analysis["tokens"]]
        
        # 1. Coherence evolution
        coherence = analysis["layer_coherence"].cpu().numpy()
        im1 = axes[0, 0].imshow(coherence, aspect='auto', cmap='viridis')
        axes[0, 0].set_xlabel('Token Position')
        axes[0, 0].set_ylabel('Layer')
        axes[0, 0].set_title('Coherence Evolution')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Entropy evolution
        entropy = analysis["layer_entropy"].cpu().numpy()
        im2 = axes[0, 1].imshow(entropy, aspect='auto', cmap='plasma')
        axes[0, 1].set_xlabel('Token Position')
        axes[0, 1].set_ylabel('Layer')
        axes[0, 1].set_title('Entropy Evolution')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. Final coherence
        final_coherence = analysis["coherence"].cpu().numpy()
        axes[1, 0].bar(range(len(final_coherence)), final_coherence)
        axes[1, 0].axhline(y=0.91, color='r', linestyle='--', label='Collapse Threshold')
        axes[1, 0].set_xlabel('Token Position')
        axes[1, 0].set_ylabel('Coherence')
        axes[1, 0].set_title('Final Layer Coherence')
        axes[1, 0].legend()
        
        # 4. Token probabilities at collapse points
        collapsed = analysis["collapsed_tokens"].cpu().numpy()
        collapse_positions = np.where(collapsed >= 0)[0]
        
        if len(collapse_positions) > 0:
            pos = collapse_positions[0]
            logits = analysis["final_logits"][pos].cpu().numpy()
            probs = np.exp(logits) / np.sum(np.exp(logits))
            top_k = 10
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_tokens = [self.tokenizer.decode([i]) for i in top_indices]
            top_probs = probs[top_indices]
            
            axes[1, 1].barh(range(top_k), top_probs)
            axes[1, 1].set_yticks(range(top_k))
            axes[1, 1].set_yticklabels(top_tokens)
            axes[1, 1].set_xlabel('Probability')
            axes[1, 1].set_title(f'Top Tokens at Position {pos}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
    def profile_inference(
        self,
        batch_sizes: List[int] = [1, 4, 8, 16],
        seq_lengths: List[int] = [128, 256, 512]
    ) -> Dict[str, Dict[str, float]]:
        """Profile inference performance"""
        self.model.eval()
        results = {}
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                config_name = f"batch{batch_size}_seq{seq_len}"
                
                # Create dummy input
                input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=self.device)
                
                # Warmup
                for _ in range(3):
                    _ = self.model(input_ids, crystal_update=False)
                    
                torch.cuda.synchronize()
                
                # Time inference
                start_time = time.time()
                n_runs = 10
                
                for _ in range(n_runs):
                    outputs = self.model(input_ids, crystal_update=False)
                    
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                
                # Compute metrics
                avg_time = elapsed / n_runs
                tokens_per_sec = (batch_size * seq_len) / avg_time
                
                # Memory usage
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                torch.cuda.reset_peak_memory_stats()
                
                results[config_name] = {
                    "batch_size": batch_size,
                    "seq_length": seq_len,
                    "avg_time_ms": avg_time * 1000,
                    "tokens_per_second": tokens_per_sec,
                    "memory_mb": memory_mb
                }
                
                print(f"{config_name}: {avg_time*1000:.1f}ms, "
                      f"{tokens_per_sec:.0f} tok/s, {memory_mb:.0f}MB")
                
        return results
    
    def compare_perturbations(
        self,
        prompt: str,
        perturbation_types: List[str] = ["levy", "beta", "adaptive"],
        num_samples: int = 3
    ) -> Dict[str, List[str]]:
        """Compare generation with different perturbations"""
        results = {}
        
        for ptype in perturbation_types:
            print(f"\nGenerating with {ptype} perturbation...")
            generations = self.generate_text(
                prompt,
                max_length=100,
                num_samples=num_samples,
                perturbation_type=ptype
            )
            results[ptype] = generations
            
        return results

def load_checkpoint(checkpoint_path: str, device: str = "cuda") -> Tuple[FieldTheoreticLM, Dict]:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model
    from model import FieldConfig, FieldTheoreticLM
    config = FieldConfig(**checkpoint["config"])
    
    # Ensure float32 dtype
    if not hasattr(config, 'dtype'):
        config.dtype = torch.float32
    
    model = FieldTheoreticLM(config)
    
    # Load state
    model.load_state_dict(checkpoint["model_state"])
    model.crystal_memory.W_crystal = checkpoint["crystal_state"]
    
    # Ensure all parameters use float32
    for param in model.parameters():
        param.data = param.data.to(torch.float32)
    
    return model, checkpoint

# Validation
if __name__ == "__main__":
    print("=== Inference Module Validation ===")
    
    # Create small model for testing
    from model import create_small_model
    model = create_small_model("log_phase")
    
    # Create inference engine
    inference = FieldInference(model)
    
    # Test generation
    print("\nTesting generation...")
    prompt = "The quantum field"
    generations = inference.generate_text(prompt, max_length=30, num_samples=2)
    for i, text in enumerate(generations):
        print(f"Sample {i+1}: {text}")
        
    # Test field analysis
    print("\nTesting field analysis...")
    analysis = inference.analyze_field_dynamics("Hello world")
    print(f"Coherence shape: {analysis['coherence'].shape}")
    print(f"Field evolution shape: {analysis['field_evolution'].shape}")
    
    # Test performance profiling
    print("\nTesting performance profiling...")
    profile = inference.profile_inference(
        batch_sizes=[1, 4],
        seq_lengths=[128]
    )
    
    print("\n[PASS] Inference module validated")
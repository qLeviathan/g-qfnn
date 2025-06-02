"""
trainer.py - Field-theoretic training without backpropagation
Physics: Crystal memory formation through Hebbian updates
"""

import torch
import numpy as np
from typing import Dict, Optional, List
import time
from pathlib import Path
import json
from dataclasses import dataclass, asdict

from model import FieldTheoreticLM
from data import FieldDataLoader

@dataclass
class TrainingMetrics:
    """Metrics tracked during training"""
    step: int
    epoch: float
    loss: float
    perplexity: float
    coherence: float
    crystal_norm: float
    tokens_per_second: float
    gpu_memory_mb: float

class FieldTrainer:
    """
    Trains field-theoretic LM without backpropagation
    Uses Hebbian crystallization for weight formation
    """
    def __init__(
        self,
        model: FieldTheoreticLM,
        data_loader: FieldDataLoader,
        output_dir: str = "outputs",
        log_every: int = 100,
        save_every: int = 1000,
        eval_every: int = 500
    ):
        self.model = model
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.log_every = log_every
        self.save_every = save_every
        self.eval_every = eval_every
        
        # Metrics tracking
        self.metrics_history: List[TrainingMetrics] = []
        self.start_time = time.time()
        self.total_tokens = 0
        
    def train_step(self, batch: Dict[str, torch.Tensor], perturbation_type: str = "adaptive") -> Dict[str, float]:
        """
        Single training step with Hebbian updates
        No gradient computation needed
        """
        # Ensure all tensors use float32
        input_ids = batch["input_ids"]
        
        # Forward pass with crystal updates
        outputs = self.model(
            input_ids,
            crystal_update=True,  # Enable Hebbian updates
            perturbation_type=perturbation_type
        )
        
        # Compute metrics (no backprop)
        with torch.no_grad():
            # Cross-entropy loss for monitoring
            logits = outputs["logits"]
            targets = batch["target_ids"]
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1)
            )
            
            perplexity = torch.exp(loss)
            
            # Field metrics
            mean_coherence = outputs["coherence"].mean()
            
            # Crystal norm
            crystal_norm = torch.norm(self.model.crystal_memory.W_crystal)
            
        return {
            "loss": loss.item(),
            "perplexity": perplexity.item(),
            "coherence": mean_coherence.item(),
            "crystal_norm": crystal_norm.item()
        }
    
    def evaluate(self, max_batches: int = 50) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        total_coherence = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in self.data_loader.iterate_batches("validation", max_batches):
                outputs = self.model(
                    batch["input_ids"],
                    crystal_update=False  # No updates during eval
                )
                
                # Loss
                logits = outputs["logits"]
                targets = batch["target_ids"]
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += shift_targets.numel()
                total_coherence += outputs["coherence"].mean().item()
                n_batches += 1
        
        avg_loss = total_loss / total_tokens
        avg_perplexity = np.exp(avg_loss)
        avg_coherence = total_coherence / n_batches
        
        self.model.train()
        
        return {
            "val_loss": avg_loss,
            "val_perplexity": avg_perplexity,
            "val_coherence": avg_coherence
        }
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"checkpoint_{step}.pt"
        
        checkpoint = {
            "step": step,
            "model_state": self.model.state_dict(),
            "crystal_state": self.model.crystal_memory.W_crystal,
            "config": asdict(self.model.config) if hasattr(self.model.config, '__dict__') else self.model.config,
            "metrics_history": self.metrics_history[-100:]  # Last 100 metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
    def train(
        self,
        num_steps: int,
        perturbation_schedule: Optional[Dict[int, str]] = None
    ):
        """
        Main training loop
        perturbation_schedule: Dict mapping step -> perturbation type
        """
        if perturbation_schedule is None:
            perturbation_schedule = {
                0: "beta",      # Start with local exploration
                5000: "adaptive",  # Switch to adaptive
                10000: "levy"     # Heavy-tailed exploration later
            }
        
        self.model.train()
        step = 0
        
        print(f"Starting training for {num_steps} steps...")
        print(f"Model memory: {self.model.get_memory_usage()['total_memory_mb']:.1f} MB")
        
        while step < num_steps:
            # Determine perturbation type
            perturbation_type = "adaptive"
            for threshold, ptype in sorted(perturbation_schedule.items()):
                if step >= threshold:
                    perturbation_type = ptype
            
            # Get batch with error handling
            try:
                batch = self.data_loader.get_batch("train")
            except Exception as e:
                print(f"Error getting batch: {e}. Retrying...")
                time.sleep(1)
                continue
                
            batch_start = time.time()
            
            # Training step
            metrics = self.train_step(batch, perturbation_type)
            
            # Track tokens/sec
            batch_time = time.time() - batch_start
            batch_tokens = batch["input_ids"].numel()
            self.total_tokens += batch_tokens
            tokens_per_sec = batch_tokens / batch_time
            
            # GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                torch.cuda.reset_peak_memory_stats()
            else:
                gpu_memory = 0
            
            # Log metrics
            if step % self.log_every == 0:
                elapsed = time.time() - self.start_time
                print(f"Step {step} | "
                      f"Loss: {metrics['loss']:.3f} | "
                      f"PPL: {metrics['perplexity']:.1f} | "
                      f"Coherence: {metrics['coherence']:.3f} | "
                      f"Crystal: {metrics['crystal_norm']:.1f} | "
                      f"Tokens/s: {tokens_per_sec:.0f} | "
                      f"GPU: {gpu_memory:.0f}MB | "
                      f"Perturb: {perturbation_type}")
                
                # Store metrics
                self.metrics_history.append(TrainingMetrics(
                    step=step,
                    epoch=self.total_tokens / 1e6,  # Millions of tokens
                    loss=metrics['loss'],
                    perplexity=metrics['perplexity'],
                    coherence=metrics['coherence'],
                    crystal_norm=metrics['crystal_norm'],
                    tokens_per_second=tokens_per_sec,
                    gpu_memory_mb=gpu_memory
                ))
            
            # Evaluate
            if step % self.eval_every == 0 and step > 0:
                print("\nEvaluating...")
                eval_metrics = self.evaluate()
                print(f"Validation - Loss: {eval_metrics['val_loss']:.3f} | "
                      f"PPL: {eval_metrics['val_perplexity']:.1f} | "
                      f"Coherence: {eval_metrics['val_coherence']:.3f}\n")
            
            # Save checkpoint
            if step % self.save_every == 0 and step > 0:
                self.save_checkpoint(step)
                self.save_metrics()
            
            step += 1
        
        # Final save
        self.save_checkpoint(step)
        self.save_metrics()
        print(f"\nTraining complete! Total time: {time.time() - self.start_time:.1f}s")
    
    def save_metrics(self):
        """Save training metrics to JSON"""
        metrics_path = self.output_dir / "metrics.json"
        
        metrics_data = {
            "metrics": [asdict(m) for m in self.metrics_history],
            "total_tokens": self.total_tokens,
            "total_time": time.time() - self.start_time
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

class AblationTrainer:
    """
    Specialized trainer for ablation studies
    Tests different configurations systematically
    """
    def __init__(self, base_config: Dict, output_dir: str = "ablations"):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def run_ablation(
        self,
        name: str,
        embedding_types: List[str] = ["golden", "log_phase"],
        perturbation_types: List[str] = ["levy", "beta", "adaptive"],
        datasets: List[str] = ["wikitext-2"],
        num_steps: int = 1000
    ):
        """Run ablation study"""
        results = {}
        
        for dataset in datasets:
            print(f"\n=== Dataset: {dataset} ===")
            data_loader = FieldDataLoader(dataset, batch_size=8, seq_length=256, max_retries=5)
            
            for embed_type in embedding_types:
                for perturb_type in perturbation_types:
                    config_name = f"{dataset}_{embed_type}_{perturb_type}"
                    print(f"\nRunning: {config_name}")
                    
                    # Create model
                    from model import create_small_model
                    model = create_small_model(embed_type).cuda()
                    
                    # Create trainer
                    trainer = FieldTrainer(
                        model, 
                        data_loader,
                        output_dir=self.output_dir / config_name,
                        log_every=50,
                        eval_every=200
                    )
                    
                    # Train with fixed perturbation
                    schedule = {0: perturb_type}
                    trainer.train(num_steps, schedule)
                    
                    # Final evaluation
                    final_metrics = trainer.evaluate(max_batches=100)
                    
                    results[config_name] = {
                        "dataset": dataset,
                        "embedding": embed_type,
                        "perturbation": perturb_type,
                        "final_perplexity": final_metrics["val_perplexity"],
                        "final_coherence": final_metrics["val_coherence"],
                        "crystal_norm": model.crystal_memory.W_crystal.norm().item(),
                        "training_time": time.time() - trainer.start_time
                    }
        
        # Save results
        results_path = self.output_dir / f"{name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n=== Ablation Results ===")
        for config, metrics in results.items():
            print(f"{config}: PPL={metrics['final_perplexity']:.1f}, "
                  f"Coherence={metrics['final_coherence']:.3f}")
        
        return results

# Validation
if __name__ == "__main__":
    print("=== Trainer Module Validation ===")
    
    # Quick test
    from model import create_small_model
    from data import FieldDataLoader
    
    model = create_small_model("golden").cuda()
    
    # Create data loader with retry logic
    data_loader = FieldDataLoader("wikitext-2", batch_size=4, seq_length=128, max_retries=5)
    
    trainer = FieldTrainer(
        model, 
        data_loader,
        log_every=10,
        eval_every=50
    )
    
    # Train for a few steps
    print("\nRunning quick training test...")
    trainer.train(num_steps=50)
    
    print("\n[PASS] Trainer validated")
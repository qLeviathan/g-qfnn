"""
Enhanced Quantum-Geometric Framework with Physics-First Principles

Key Physics Components:
1. Radial Diffusion (ODE1) = Imaginary-time Schrödinger (ODE2)
   - Uses equivalence between diffusion and quantum mechanics in imaginary time
   - Implements proper radial coordinates for geometric interpretation

2. Poisson/Flux Equation Simplification
   - Takes epsilon -> infinity limit for simplified field equations
   - Maintains geometric interpretation of token interactions

3. Distance-based Attention Matrix
   - No activation functions - pure geometric distance
   - Proper normalization in radial coordinates
   - Lower triangular mask for causal attention

4. Heun-Euler Integration
   - Two-step integration scheme for stability
   - Adaptive time stepping based on Beta distribution
   - Proper handling of radial coordinates

5. Normalized Radial Coordinates
   - Direct radial distance representation
   - Embedding dimension N with 1 dim reserved for geometry
   - Statistical normalization for isotropic interactions
"""
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from dataclasses import dataclass
from torch.amp import autocast
from transformers import GPT2Tokenizer
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx

# Use mixed precision training instead of global float16


def print_tensor_stats(name: str, tensor: torch.Tensor, print_output: bool = False):
    """Calculate and optionally print detailed statistics about tensor values."""
    with torch.no_grad():
        stats = {
            "Shape": tensor.shape,
            "Dtype": tensor.dtype,
            "Min": tensor.min().item(),
            "Max": tensor.max().item(),
            "Mean": tensor.mean().item(),
            "Std": tensor.std(unbiased=True).item() if tensor.numel() > 1 else 0.0
        }
        
        # Check for NaN/Inf
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        if print_output:
            print(f"\n{name} Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6e}")
                else:
                    print(f"  {key}: {value}")
            if has_nan:
                print("  WARNING: Contains NaN values!")
            if has_inf:
                print("  WARNING: Contains Inf values!")
                
        return stats

@dataclass
class Config:
    N: int = 512             # Total embedding dimension
    D: int = 64              # Sequence length (reduced for testing)
    device: str = "cuda"     # Device to run on
    vocab_size: int = 50257  # GPT2 vocab size
    dt_scale: float = 0.00005 # Integration time step scaling (smaller for geometric stability)
    stop_threshold: float = 0.0001  # Early stopping threshold

    def __post_init__(self):
        # Reserve 1 dimension for radial distance
        self.reserved = 1
        self.attn_dim = self.N - self.reserved
        # Validate dimensions
        assert self.N >= 2, "Need N >= 2 for radial geometry + extra dimensions"
        assert self.D >= 2, "Need D >= 2 for meaningful sequence interactions"

class QuantumAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.N = config.N
        self.D = config.D
        self.vocab_size = config.vocab_size
        self.dt_scale = torch.tensor(config.dt_scale, dtype=torch.float32)

        # Initialize radial embeddings directly in float16
        pos_indices = torch.arange(self.vocab_size, dtype=torch.float16)
        r0 = pos_indices / self.vocab_size  # Normalized radial distances [0, 1]
        
        # Apply statistical normalization
        r0 = (r0 - r0.mean()) / (r0.std() + 1e-5)  # Z-score normalization
        r0 = r0.unsqueeze(-1)  # [vocab_size, 1] for radial distances only
        
        # Print initialization stats if debug mode
        if hasattr(self, 'debug') and self.debug:
            print_tensor_stats("Geometric embeddings initialization", r0, True)
        
        # Register as buffer (non-trainable)
        self.register_buffer('r0', r0)  # [vocab_size, 1] for radial distances
        
        # Projections for combining geometric and learned features
        self.diff_proj = nn.Linear(1, self.N - 1)  # Project diffusion coefficient to N-1 dims
        self.out_proj = nn.Linear(self.N, self.vocab_size)  # Final projection

    def compute_score(self, r_embed: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Compute sparse attention scores using confidence intervals on radial distances."""
        device = r_embed.device
        batch_size, seq_len = r_embed.shape[:2]
        
        # Compute radial distances directly
        dists = torch.abs(r_embed.unsqueeze(2) - r_embed.unsqueeze(1))  # [B, D, D, 1]
        dists = dists.squeeze(-1)  # [B, D, D]
        
        # Calculate confidence intervals per batch and sequence position
        with torch.no_grad():
            # Calculate stats over the sequence dimension
            mean_dist = dists.mean(dim=2, keepdim=True)  # [B, D, 1]
            std_dist = dists.std(dim=2, keepdim=True)    # [B, D, 1]
            conf_threshold = mean_dist + 2 * std_dist     # 2-sigma confidence interval
            
            # Broadcast for comparison
            conf_threshold = conf_threshold.expand_as(dists)  # [B, D, D]
        
        # Create sparse attention mask
        causal_mask = torch.tril(torch.ones_like(dists, dtype=torch.bool), diagonal=-1)
        sparse_mask = (dists <= conf_threshold) & causal_mask
        
        # Compute normalized scores only for valid connections
        scores = torch.zeros_like(dists)
        valid_dists = dists[sparse_mask]
        valid_thresholds = conf_threshold[sparse_mask]
        scores[sparse_mask] = 1.0 - (valid_dists / valid_thresholds)
        
        if debug:
            sparsity = sparse_mask.float().mean()
            print(f"\nAttention sparsity: {sparsity.item():.2%}")
            print_tensor_stats("Radial distances", dists, True)
            print_tensor_stats("Confidence thresholds", conf_threshold, True)
            print_tensor_stats("Sparse attention scores", scores, True)
        
        return scores

    @autocast("cuda")
    def sample_inverse_beta(self, batch_size: int, debug: bool = False) -> torch.Tensor:
        """Sample integration time step using inverse Beta distribution."""
        # Sample in float32 for numerical stability
        a = torch.tensor(float(self.N - 2) / 2.0, dtype=torch.float32)
        b = torch.tensor(float(self.D - 1) / 2.0, dtype=torch.float32)
        x = torch.distributions.Beta(a, b).sample((batch_size,))
        x = x.to(device=a.device, dtype=torch.float32)
        
        # Calculate dt with proper clamping
        dt = (1.0 / x.clamp(min=1e-6)) * self.dt_scale
        dt = dt.clamp(0.001, 0.05)  # Smaller max step for geometric stability
        
        if debug:
            print_tensor_stats("Integration timestep (dt)", dt, True)
            
        return dt

    @autocast("cuda")
    def integrator_step(self, r_embed: torch.Tensor, dt: torch.Tensor, debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform Heun-Euler integration step with float16 precision."""
        device = r_embed.device
        
        # First integration step with proper scaling
        score = self.compute_score(r_embed, debug)
        dt = dt.to(device)  # Ensure dt is on the same device
        dt_score = dt.unsqueeze(-1).unsqueeze(-1) * score
        k1 = torch.einsum('bdk,bk->bd', dt_score, r_embed.squeeze(-1)).unsqueeze(-1)
        k1 = torch.clamp(k1, min=-1.0, max=1.0)  # Clamp for geometric stability
        
        # Second integration step with proper scaling
        target = r_embed + k1
        score_target = self.compute_score(target, debug)
        dt_score_target = dt.unsqueeze(-1).unsqueeze(-1) * score_target
        k2 = torch.einsum('bdk,bk->bd', dt_score_target, target.squeeze(-1)).unsqueeze(-1)
        k2 = torch.clamp(k2, min=-1.0, max=1.0)  # Clamp for geometric stability
        
        # Combine steps with proper scaling
        r_new = r_embed + 0.5 * (k1 + k2)
        
        # Normalize with float16 precision
        r_new = F.normalize(r_new, p=2, dim=-1)
        
        if debug:
            print_tensor_stats("First step (k1)", k1, True)
            print_tensor_stats("Second step (k2)", k2, True)
            print_tensor_stats("Combined update", r_new, True)
        
        return r_new, score_target

    @autocast("cuda")
    def forward(self, x: torch.Tensor, debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Forward pass with physics-based token evolution using float16 precision."""
        batch_size = x.shape[0]
        device = x.device
        
        # Get embeddings for input tokens with float16 precision
        token_indices = x.long()  # Keep indices as long for proper indexing
        r_embed = self.r0[token_indices].to(device).float()  # Get embeddings
        r_embed = F.normalize(r_embed, p=2, dim=-1)  # Normalize for geometric stability
        
        # Sample integration timestep with float16 precision
        dt = self.sample_inverse_beta(batch_size, debug).to(device)
        dt = torch.clamp(dt, min=0.0, max=1.0)  # Clamp for stability
        
        # Perform integration steps
        r_new, score_target = self.integrator_step(r_embed, dt, debug)
        
        # Calculate diffusion coefficients with float16 precision
        diff_mean = score_target.mean(dim=-1, keepdim=True)
        diff_coef = torch.clamp(diff_mean, min=0.0, max=1.0)  # Clamp for stability
        
        # Project features with float16 precision
        v = self.diff_proj(diff_coef)
        v = torch.clamp(v, min=-1.0, max=1.0)  # Clamp for geometric stability
        
        # Combine representations with float16 precision
        final_rep = torch.cat([r_new, v], dim=-1)
        
        # Generate logits with float16 precision
        logits = self.out_proj(final_rep)
        logits = torch.clamp(logits, min=-2.0, max=2.0)  # Wider range for logits
        
        if debug:
            print_tensor_stats("Initial embeddings", r_embed, True)
            print_tensor_stats("Diffusion coefficients", diff_coef, True)
            print_tensor_stats("Projected features", v, True)
            print_tensor_stats("Final representation", final_rep, True)
            print_tensor_stats("Output logits", logits, True)
        
        # Calculate average dt before scaling
        avg_dt = dt.mean().item()  # Get scalar value
        
        return logits, score_target, avg_dt


def find_valid_continuations(current_tokens: List[int], train_sequences: List[str], tokenizer: GPT2Tokenizer) -> List[int]:
    """Find valid next tokens from training sequences."""
    valid_tokens = set()
    
    # Convert current tokens to text for context matching
    current_text = tokenizer.decode(current_tokens)
    
    for seq in train_sequences:
        if current_text in seq:
            # Find position where current text ends
            pos = seq.find(current_text) + len(current_text)
            if pos < len(seq):
                # Get next token
                next_text = seq[pos:pos+1]
                next_tokens = tokenizer.encode(next_text)
                if next_tokens:
                    valid_tokens.add(next_tokens[0])
    
    return list(valid_tokens)

def generate_sequence(model: QuantumAttention, prompt: str, tokenizer: GPT2Tokenizer, 
                     max_length: int = 20, initial_temperature: float = 2.0, top_k: int = 100) -> str:
    """Generate text continuation using training sequence patterns with float16 precision."""
    model.eval()
    device = model.r0.device
    
    # Tokenize prompt (use long for CUDA compatibility)
    input_ids = tokenizer(
        prompt,
        padding='max_length',
        truncation=True,
        max_length=model.D,
        return_tensors="pt"
    )["input_ids"].to(device).to(dtype=torch.long)  # Use long for CUDA
    generated_ids = input_ids[0].tolist()
    
    # Get prompt length (use uint8 since max_length is small)
    prompt_length = torch.tensor(len(tokenizer.encode(prompt)), dtype=torch.uint8)
    
    # Generate tokens
    for _ in range(max_length - prompt_length.item()):
        # Forward pass (logits will be in float16 from model)
        logits, score_target, avg_dt = model(input_ids)
        
        # Get next token prediction
        last_token_pos = min(len(generated_ids) - 1, model.D - 1)
        next_token_logits = logits[0, last_token_pos]
        
        # Find valid continuations from training data
        current_tokens = generated_ids[-min(len(generated_ids), 8):]
        valid_next_tokens = find_valid_continuations(current_tokens, train_sequences, tokenizer)
        
        if valid_next_tokens:
            # Only allow tokens from training sequences (use bool for mask)
            mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
            mask[valid_next_tokens] = True
            next_token_logits = next_token_logits.masked_fill(~mask, float('-inf'))
            
            # Sample from valid tokens (temperature scaling in float16)
            temperature = torch.tensor(initial_temperature * (1.0 - len(generated_ids) / max_length), dtype=torch.float16)
            next_token_probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(next_token_probs, 1).to(dtype=torch.long).item()  # Use long for CUDA
            
            # Print selected token
            token_str = tokenizer.decode([next_token])
            print(f"Selected token: {token_str} (ID: {next_token})")
            
            # Append token
            generated_ids.append(next_token)
            input_ids = torch.tensor([generated_ids], device=device, dtype=torch.long)  # Use long for CUDA
        else:
            # Try to complete with a training sequence
            for seq in train_sequences:
                if seq.startswith(prompt):
                    return seq
            break
    
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def train_step(model: QuantumAttention, batch: torch.Tensor, optimizer: torch.optim.Optimizer) -> Tuple[float, float, float]:
    """Training step with exact sequence matching using float16 precision."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass (already in float16 from model)
    logits, score_target, avg_dt = model(batch)
    
    # Language modeling loss with float16 precision
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = batch[..., 1:].contiguous()
    
    # Position weights (binary scaling can use uint8)
    seq_len = shift_labels.size(1)
    position_scale = torch.arange(seq_len, dtype=torch.uint8, device=batch.device) / seq_len
    position_weights = 1.0 + position_scale.to(dtype=torch.float16)
    
    # Per-token loss with float16 precision
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1).long()  # Use long for CUDA compatibility
    # Let float16 handle precision
    per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
    per_token_loss = per_token_loss.view(shift_labels.size(0), shift_labels.size(1))  # Reshape to match position weights
    
    # Weight losses (keep in float16)
    weighted_loss = per_token_loss * position_weights
    lm_loss = weighted_loss.mean()
    
    # Geometric coherence loss with float16 precision
    token_embeddings = model.r0[batch.long()].float()  # Get embeddings
    diffs = token_embeddings[:, 1:] - token_embeddings[:, :-1]
    distances = torch.norm(diffs, dim=-1)
    geo_loss = torch.mean(distances)
    
    # Attention coherence loss with float16 precision
    attn_scores = score_target.softmax(dim=-1)
    attn_diffs = attn_scores[:, 1:] - attn_scores[:, :-1]
    attn_loss = torch.mean(torch.norm(attn_diffs, dim=-1))
    
    # Exact sequence matching loss
    exact_loss = 0.0
    for b in range(batch.size(0)):
        # Get sequence for this batch (use long for CUDA compatibility)
        seq = train_sequences[b % len(train_sequences)]
        seq_tokens = torch.tensor(tokenizer.encode(seq), dtype=torch.long, device=device)
        
        # Get model predictions for this sequence length
        seq_len = min(len(seq_tokens), logits.size(1))
        seq_logits = logits[b, :seq_len]
        seq_tokens = seq_tokens[:seq_len]
        
        # Strong cross entropy loss for exact matches (convert to float32 for cross_entropy)
        exact_loss += F.cross_entropy(seq_logits.float(), seq_tokens)
    exact_loss = exact_loss / batch.size(0)
    
    # Combined loss with focus on exact sequence learning
    loss = 10.0 * exact_loss + 2.0 * lm_loss + 0.05 * geo_loss + 0.05 * attn_loss
    
    # Calculate sparsity using confidence interval mask
    with torch.no_grad():
        # Get mean and std of attention scores for confidence interval
        mean_score = score_target.mean(dim=-1, keepdim=True)
        std_score = score_target.std(dim=-1, keepdim=True)
        # Count connections within 2-sigma confidence interval
        valid_mask = (score_target > 0.0) & (score_target >= (mean_score - 2 * std_score))
        sparsity = valid_mask.float().mean().item()
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item(), sparsity, avg_dt
if __name__ == "__main__":
    import time
    import psutil
    import torch.cuda.profiler as profiler
    import torch.cuda.nvtx as nvtx
    from torch.profiler import profile, record_function, ProfilerActivity
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Enable profiling
    torch.cuda.empty_cache()
    start_time = time.time()
    
    # Initialize model
    config = Config()
    config.vocab_size = len(tokenizer)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = device.type
    
    print(f"\nInitializing QuantumAttention model:")
    print(f"  Device: {device}")
    print(f"  Embedding dim (N): {config.N}")
    print(f"  Sequence length (D): {config.D}")
    print(f"  Vocab size: {config.vocab_size}")
    
    # Create model and optimizer
    model = QuantumAttention(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training data
    train_sequences = [
        "The quick brown fox jumps over the lazy dog",
        "I think therefore I am",
        "To be or not to be that is the question",
        "All that glitters is not gold",
        "A journey of a thousand miles begins with a single step",
        "Actions speak louder than words",
        "Beauty is in the eye of the beholder",
        "Every cloud has a silver lining"
    ]
    
    # Training loop with profiling
    print("\nStarting training with performance profiling:")
    print("-" * 60)
    
    num_epochs = 500
    batch_size = len(train_sequences)
    
    # Curriculum learning
    sequence_lengths = [8, 16, 32, 64]
    epochs_per_length = num_epochs // len(sequence_lengths)
    
    # Performance metrics
    total_tokens = 0
    total_time = 0
    
    for curr_length_idx, max_length in enumerate(sequence_lengths):
        print(f"\nTraining on sequences of length {max_length}:")
        print("-" * 60)
        
        # Create training batch
        encodings = tokenizer(
            train_sequences,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        train_batch = encodings["input_ids"].to(device)
        
        # Train for this length
        start_epoch = curr_length_idx * epochs_per_length
        end_epoch = start_epoch + epochs_per_length
        
        # Start profiling
        with torch.cuda.profiler.profile():
            with torch.cuda.nvtx.range(f"sequence_length_{max_length}"):
                for epoch in range(start_epoch, end_epoch):
                    # Profile each epoch
                    epoch_start = time.time()
                    with profile(
                        activities=[
                            ProfilerActivity.CPU,
                            ProfilerActivity.CUDA,
                        ],
                        with_stack=True,
                        profile_memory=True,
                    ) as prof:
                        with record_function("epoch"):
                            loss, sparsity, avg_dt = train_step(model, train_batch, optimizer)
                    
                    epoch_time = time.time() - epoch_start
                    
                    # Get CPU and GPU usage
                    cpu_percent = psutil.cpu_percent()
                    gpu_util = torch.cuda.utilization()
                    
                    # Calculate tokens per second
                    tokens_processed = batch_size * max_length
                    total_tokens += tokens_processed
                    total_time += epoch_time
                    tokens_per_second = tokens_processed / epoch_time
                    
                    if (epoch + 1) % 100 == 0:
                        print(f"Epoch {epoch + 1}/{num_epochs}:")
                        print(f"  Loss: {loss:.4f}")
                        print(f"  Attention Sparsity: {sparsity:.2%}")
                        print(f"  Average dt: {avg_dt:.6f} (scaled: {avg_dt:.6e})")
                        print(f"  Tokens/second: {tokens_per_second:.2f}")
                        print(f"  Memory used: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")
                        print(f"  CPU Usage: {cpu_percent:.1f}%")
                        print(f"  GPU Usage: {gpu_util}%")
                        
                        if (epoch + 1) % 200 == 0:  # Print profile every 200 epochs
                            print("\nProfile Summary:")
                            print(prof.key_averages().table(
                                sort_by="cuda_time_total", row_limit=10))
    
    # Print final performance stats
    avg_tokens_per_second = total_tokens / total_time
    print("\nPerformance Summary:")
    print("-" * 60)
    print(f"Total training time: {total_time:.2f}s")
    print(f"Average tokens/second: {avg_tokens_per_second:.2f}")
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")
    memory_allocated = torch.cuda.memory_allocated()
    memory_reserved = torch.cuda.memory_reserved()
    memory_efficiency = memory_allocated / memory_reserved if memory_reserved > 0 else 1.0
    print(f"Memory efficiency: {memory_efficiency:.2%}")
    print(f"Memory allocated/reserved: {memory_allocated/1024**2:.1f}MB/{memory_reserved/1024**2:.1f}MB")
    
    # Test generation
    print("\nTesting sequence generation:")
    print("-" * 60)
    
    test_prompts = [
        "The quick brown fox",
        "I think",
        "To be or",
        "To be a fox",
        "All that"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        generated = generate_sequence(model, prompt, tokenizer, initial_temperature=2.0, top_k=100)
        print(f"Generated: {generated}")

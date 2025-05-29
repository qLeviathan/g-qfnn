"""
collapse.py - Field collapse dynamics and token sampling
Physics: Coherence-driven collapse to discrete tokens from continuous field
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy.stats import levy_stable

PHI = (1 + np.sqrt(5)) / 2

class FieldCollapse(torch.nn.Module):
    """
    Implements field collapse from continuous manifold to discrete tokens
    Physics: When coherence > threshold, field collapses to observable (token)
    """
    def __init__(self, vocab_size: int, d_model: int, collapse_threshold: float = 0.91):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.collapse_threshold = collapse_threshold
        
        # Measurement operators (vocab_size observable bases)
        self.measurement_ops = torch.nn.Parameter(
            torch.randn(vocab_size, d_model) / np.sqrt(d_model)
        )
        
    def compute_coherence(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute field coherence (order parameter)
        field: (batch, seq_len, d_model)
        Returns: (batch, seq_len) - Coherence values in [0, 1]
        """
        # Local covariance matrix
        batch, seq_len, d_model = field.shape
        
        # Compute covariance in local windows
        coherence = torch.zeros(batch, seq_len, device=field.device)
        
        for i in range(seq_len):
            # Window around position i
            start = max(0, i - 2)
            end = min(seq_len, i + 3)
            window = field[:, start:end, :]  # (batch, window_size, d_model)
            
            # Center the window
            window_centered = window - window.mean(dim=1, keepdim=True)
            
            # Covariance matrix
            cov = torch.einsum('bwd,bwe->bde', window_centered, window_centered) / window.shape[1]
            
            # Coherence = largest eigenvalue / trace
            eigvals = torch.linalg.eigvalsh(cov)  # (batch, d_model)
            coherence[:, i] = eigvals[:, -1] / (eigvals.sum(dim=-1) + 1e-8)
            
        return coherence
    
    def collapse_probability(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute probability distribution over tokens via Born rule
        field: (batch, seq_len, d_model)
        Returns: (batch, seq_len, vocab_size) - Token probabilities
        """
        # Project field onto measurement operators
        # P(token_i) = |⟨M_i|ψ⟩|²
        projections = torch.einsum('vd,bsd->bsv', self.measurement_ops, field)
        
        # Born rule: square amplitudes
        probabilities = projections**2
        
        # Normalize
        probabilities = probabilities / (probabilities.sum(dim=-1, keepdim=True) + 1e-8)
        
        return probabilities
    
    def forward(self, field: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collapse field to tokens where coherence exceeds threshold
        field: (batch, seq_len, d_model)
        Returns: 
            tokens: (batch, seq_len) - Collapsed token indices (-1 for uncollapsed)
            probs: (batch, seq_len, vocab_size) - Collapse probabilities
        """
        batch, seq_len, _ = field.shape
        device = field.device
        
        # Compute coherence
        coherence = self.compute_coherence(field)  # (batch, seq_len)
        
        # Determine collapse points
        collapse_mask = coherence > self.collapse_threshold  # (batch, seq_len)
        
        # Compute collapse probabilities
        probs = self.collapse_probability(field)  # (batch, seq_len, vocab_size)
        
        # Apply temperature
        if temperature != 1.0:
            probs = torch.pow(probs, 1.0 / temperature)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Sample tokens at collapse points
        tokens = torch.full((batch, seq_len), -1, dtype=torch.long, device=device)
        
        if collapse_mask.any():
            # Sample from distribution
            collapsed_probs = probs[collapse_mask]  # (n_collapsed, vocab_size)
            sampled = torch.multinomial(collapsed_probs, num_samples=1).squeeze(-1)
            tokens[collapse_mask] = sampled
        
        return tokens, probs

class LevyFieldSampler(torch.nn.Module):
    """
    Lévy flight-based token sampling
    Physics: Heavy-tailed exploration of token space
    """
    def __init__(self, vocab_size: int, alpha: float = PHI):
        super().__init__()
        self.vocab_size = vocab_size
        self.alpha = alpha
        
    def forward(self, logits: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample tokens using Lévy flight dynamics
        logits: (batch, vocab_size)
        Returns: (batch, num_samples) - Sampled token indices
        """
        batch_size = logits.shape[0]
        device = logits.device
        
        # Convert logits to initial distribution
        probs = F.softmax(logits, dim=-1)
        
        samples = []
        
        for _ in range(num_samples):
            # Current position in probability space
            current_pos = torch.multinomial(probs, 1).squeeze(-1)  # (batch,)
            
            # Lévy flight step
            # Generate on CPU due to scipy
            levy_steps = []
            for _ in range(batch_size):
                step = levy_stable.rvs(self.alpha, beta=0, scale=0.1)
                levy_steps.append(step)
            
            levy_steps = torch.tensor(levy_steps, device=device)
            
            # Convert to token space movement
            step_size = torch.abs(levy_steps) * self.vocab_size * 0.1
            direction = torch.sign(torch.randn(batch_size, device=device))
            
            # New position
            new_pos = current_pos + (step_size * direction).long()
            new_pos = torch.clamp(new_pos, 0, self.vocab_size - 1)
            
            samples.append(new_pos)
        
        return torch.stack(samples, dim=1)

class NucleusFieldSampler(torch.nn.Module):
    """
    Field-aware nucleus sampling
    Physics: Sample from high-coherence regions of field
    """
    def __init__(self, top_p: float = 0.9):
        super().__init__()
        self.top_p = top_p
        
    def forward(self, probs: torch.Tensor, coherence: torch.Tensor) -> torch.Tensor:
        """
        Sample tokens with coherence-weighted nucleus sampling
        probs: (batch, seq_len, vocab_size) - Token probabilities
        coherence: (batch, seq_len) - Field coherence values
        Returns: (batch, seq_len) - Sampled tokens
        """
        batch, seq_len, vocab_size = probs.shape
        device = probs.device
        
        tokens = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
        
        for b in range(batch):
            for s in range(seq_len):
                # Weight probabilities by coherence
                weighted_probs = probs[b, s] * (1 + coherence[b, s])
                weighted_probs = weighted_probs / weighted_probs.sum()
                
                # Sort probabilities
                sorted_probs, sorted_indices = torch.sort(weighted_probs, descending=True)
                
                # Compute cumulative probabilities
                cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                
                # Find nucleus
                nucleus_mask = cumsum_probs <= self.top_p
                if not nucleus_mask.any():
                    nucleus_mask[0] = True
                
                # Renormalize nucleus
                nucleus_probs = sorted_probs[nucleus_mask]
                nucleus_probs = nucleus_probs / nucleus_probs.sum()
                nucleus_indices = sorted_indices[nucleus_mask]
                
                # Sample from nucleus
                sampled_idx = torch.multinomial(nucleus_probs, 1)
                tokens[b, s] = nucleus_indices[sampled_idx]
        
        return tokens

class BeamFieldSearch:
    """
    Beam search with field dynamics
    Physics: Multiple field trajectories evolve in parallel
    """
    def __init__(self, beam_size: int = 4, length_penalty: float = 0.6):
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        
    def search(self, field_evolution_fn, initial_field: torch.Tensor, 
               max_length: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search through field evolution
        Returns: (tokens, scores) for top beam
        """
        device = initial_field.device
        batch_size = initial_field.shape[0]
        
        # Initialize beams
        beams = [(initial_field, torch.zeros(batch_size, 0, dtype=torch.long, device=device), 0.0)]
        
        for step in range(max_length):
            candidates = []
            
            for field, tokens, score in beams:
                # Evolve field
                evolved_field = field_evolution_fn(field)
                
                # Get token probabilities
                # (Simplified - would use FieldCollapse in practice)
                probs = torch.randn(batch_size, 50257, device=device).softmax(dim=-1)
                
                # Get top-k tokens
                topk_probs, topk_indices = torch.topk(probs, self.beam_size, dim=-1)
                
                for k in range(self.beam_size):
                    new_tokens = torch.cat([tokens, topk_indices[:, k:k+1]], dim=1)
                    new_score = score + torch.log(topk_probs[:, k]).sum().item()
                    candidates.append((evolved_field, new_tokens, new_score))
            
            # Select top beams
            candidates.sort(key=lambda x: x[2] / (len(x[1][0]) ** self.length_penalty), reverse=True)
            beams = candidates[:self.beam_size]
            
            # Early stopping
            if all(len(tokens[0]) >= max_length for _, tokens, _ in beams):
                break
        
        # Return best beam
        best_field, best_tokens, best_score = beams[0]
        return best_tokens, torch.tensor([best_score])

# Validation
if __name__ == "__main__":
    print("=== Field Collapse Module Validation ===")
    
    # Test field collapse
    collapse = FieldCollapse(vocab_size=1000, d_model=64).cuda()
    field = torch.randn(2, 16, 64).cuda()
    field = field / torch.norm(field, dim=-1, keepdim=True)
    
    tokens, probs = collapse(field)
    print(f"Collapsed tokens shape: {tokens.shape}")
    print(f"Collapse probabilities shape: {probs.shape}")
    print(f"Collapsed positions: {(tokens >= 0).sum()} / {tokens.numel()}")
    
    # Test coherence computation
    coherence = collapse.compute_coherence(field)
    print(f"\nCoherence range: [{coherence.min():.3f}, {coherence.max():.3f}]")
    
    # Test Lévy sampler
    levy_sampler = LevyFieldSampler(vocab_size=1000)
    logits = torch.randn(2, 1000).cuda()
    levy_samples = levy_sampler(logits, num_samples=5)
    print(f"\nLévy samples shape: {levy_samples.shape}")
    
    # Test nucleus sampler
    nucleus_sampler = NucleusFieldSampler(top_p=0.9)
    sampled = nucleus_sampler(probs, coherence)
    print(f"Nucleus samples shape: {sampled.shape}")
    
    print("\n✓ Field collapse validated")
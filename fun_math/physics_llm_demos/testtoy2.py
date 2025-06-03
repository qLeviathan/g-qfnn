import numpy as np
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
from enum import Enum

# Constants
PHI = (1 + 5 ** 0.5) / 2  # Golden ratio ≈ 1.618
OPTIMAL_SPARSITY = 4 * PHI / (3 * np.pi * np.sqrt(np.e))  # ≈ 0.4165
EPSILON = 1e-8  # Small constant for numerical stability

class AttentionMechanism(Enum):
    COSINE_PHASE = "cos_phase"              # 0.5 + 0.5*cos(θi - θj)
    NEGATIVE_DISTANCE = "neg_distance"      # -||ψi - ψj||
    INNER_PRODUCT = "inner_product"         # <ψi, ψj>
    FLUX_ATTENTION = "flux_attention"       # (0.5 + 0.5*cos(θi - θj))/(||ψi - ψj|| + ε)
    FLUX_WITH_AMPLITUDE = "flux_with_amp"   # (0.5 + 0.5*cos(θi - θj))*ri*rj/(||ψi - ψj|| + ε)

class RadialEncoding(Enum):
    GOLDEN_RATIO = "golden_ratio"           # θv = 2π · ((φ · v) % 1)
    UNIFORM = "uniform"                     # θv = 2π · (v/V)
    FIBONACCI_MOD1 = "fib_mod1"             # θv = 2π · (Fib(v) % 1)

class CylinderField(Enum):
    STANDARD = "standard"                   # No additional modulation
    HARMONIC = "harmonic"                   # Harmonic oscillator potential
    COULOMB = "coulomb"                     # 1/r potential
    GAUSSIAN = "gaussian"                   # Gaussian well potential

@dataclass
class QFNNConfig:
    """Configuration for QFNN ablation tests."""
    vocab_size: int = 1000
    embedding_dim: int = 4     # N
    sequence_length: int = 16  # D
    radial_min: float = 0.3
    radial_max: float = 0.9
    attention_mechanism: AttentionMechanism = AttentionMechanism.FLUX_ATTENTION
    radial_encoding: RadialEncoding = RadialEncoding.GOLDEN_RATIO
    cylinder_field: CylinderField = CylinderField.STANDARD
    dt_scale: float = 0.01
    integration_steps: int = 3
    sparsity: float = OPTIMAL_SPARSITY

def generate_embeddings(tokens: List[int], config: QFNNConfig) -> np.ndarray:
    """Generate token embeddings based on specified encoding."""
    D = len(tokens)
    N = config.embedding_dim
    embeddings = np.zeros((D, N))
    
    # Calculate radii based on token position
    if D > 1:
        radii = np.linspace(config.radial_min, config.radial_max, D)
    else:
        radii = np.array([config.radial_min])
    
    # Calculate phase angles based on encoding
    if config.radial_encoding == RadialEncoding.GOLDEN_RATIO:
        thetas = np.array([2 * np.pi * ((PHI * token) % 1) for token in tokens])
    
    elif config.radial_encoding == RadialEncoding.UNIFORM:
        thetas = np.array([2 * np.pi * (token % config.vocab_size) / config.vocab_size for token in tokens])
    
    elif config.radial_encoding == RadialEncoding.FIBONACCI_MOD1:
        # Fibonacci-based distribution
        fib = [0, 1]
        while len(fib) <= max(tokens) + 1:
            fib.append(fib[-1] + fib[-2])
        thetas = np.array([2 * np.pi * (fib[token % len(fib)] % 1) for token in tokens])
    
    # Apply cylinder field modulation to radii
    if config.cylinder_field == CylinderField.HARMONIC:
        # Harmonic oscillator potential: V(r) = k*r²
        radii = radii * (1.0 - 0.2 * np.sin(2 * thetas))
    
    elif config.cylinder_field == CylinderField.COULOMB:
        # Coulomb-like potential: V(r) = 1/r
        radii = radii * (1.0 + 0.2 * np.cos(3 * thetas))
    
    elif config.cylinder_field == CylinderField.GAUSSIAN:
        # Gaussian well potential: V(r) = exp(-r²/2σ²)
        radii = radii * (1.0 + 0.15 * np.sin(5 * thetas) * np.cos(2 * thetas))
    
    # Convert polar to cartesian coordinates (first 2 dimensions)
    for i in range(D):
        embeddings[i, 0] = radii[i] * np.cos(thetas[i])
        embeddings[i, 1] = radii[i] * np.sin(thetas[i])
    
    return embeddings

def compute_attention(embeddings: np.ndarray, config: QFNNConfig) -> np.ndarray:
    """Compute attention matrix based on specified mechanism."""
    D = embeddings.shape[0]
    attention = np.zeros((D, D))
    
    # Extract circular components (first 2 dimensions)
    circular = embeddings[:, :2]
    
    # Calculate angles and radii
    thetas = np.arctan2(circular[:, 1], circular[:, 0])
    radii = np.sqrt(circular[:, 0]**2 + circular[:, 1]**2)
    
    # Compute attention based on mechanism
    if config.attention_mechanism == AttentionMechanism.COSINE_PHASE:
        # Phase-based attention (0.5 + 0.5*cos(θi - θj))
        theta_i = thetas.reshape(-1, 1)
        theta_j = thetas.reshape(1, -1)
        phase_diff = theta_i - theta_j
        attention = 0.5 + 0.5 * np.cos(phase_diff)
    
    elif config.attention_mechanism == AttentionMechanism.NEGATIVE_DISTANCE:
        # Negative distance attention (-||ψi - ψj||)
        for i in range(D):
            for j in range(D):
                attention[i, j] = -np.linalg.norm(embeddings[i] - embeddings[j])
        
        # Normalize to [0, 1]
        if D > 1:
            attention = (attention - attention.min()) / (attention.max() - attention.min() + EPSILON)
    
    elif config.attention_mechanism == AttentionMechanism.INNER_PRODUCT:
        # Inner product attention (<ψi, ψj>)
        attention = np.dot(embeddings, embeddings.T)
        
        # Normalize to [0, 1]
        if D > 1:
            attention = (attention - attention.min()) / (attention.max() - attention.min() + EPSILON)
    
    elif config.attention_mechanism == AttentionMechanism.FLUX_ATTENTION:
        # Magnetic flux attention (0.5 + 0.5*cos(θi - θj))/(||ψi - ψj|| + ε)
        theta_i = thetas.reshape(-1, 1)
        theta_j = thetas.reshape(1, -1)
        phase_diff = theta_i - theta_j
        phase_sim = 0.5 + 0.5 * np.cos(phase_diff)
        
        # Compute distances
        distances = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                distances[i, j] = np.linalg.norm(circular[i] - circular[j]) + EPSILON
        
        # Final attention
        attention = phase_sim / distances
    
    elif config.attention_mechanism == AttentionMechanism.FLUX_WITH_AMPLITUDE:
        # Magnetic flux with amplitude: (0.5 + 0.5*cos(θi - θj))*ri*rj/(||ψi - ψj|| + ε)
        theta_i = thetas.reshape(-1, 1)
        theta_j = thetas.reshape(1, -1)
        phase_diff = theta_i - theta_j
        phase_sim = 0.5 + 0.5 * np.cos(phase_diff)
        
        # Amplitude product
        r_i = radii.reshape(-1, 1)
        r_j = radii.reshape(1, -1)
        amplitude_product = r_i * r_j
        
        # Compute distances
        distances = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                distances[i, j] = np.linalg.norm(circular[i] - circular[j]) + EPSILON
        
        # Final attention with amplitude product
        attention = (phase_sim * amplitude_product) / distances
    
    return attention

def apply_sparsity(attention: np.ndarray, config: QFNNConfig) -> np.ndarray:
    """Apply sparsity to attention matrix."""
    D = attention.shape[0]
    sparse_attention = attention.copy()
    
    # Vectorized thresholding
    flat_attn = attention.reshape(-1)
    k = max(1, int((1.0 - config.sparsity) * flat_attn.size))
    if k < len(flat_attn):
        threshold = np.partition(flat_attn, -k)[-k]
        sparse_attention = (attention >= threshold).astype(float) * attention
    
    # Ensure self-attention (diagonal) is preserved
    np.fill_diagonal(sparse_attention, np.diag(attention))
    
    # Row-normalize
    row_sums = sparse_attention.sum(axis=1, keepdims=True)
    sparse_attention = sparse_attention / (row_sums + EPSILON)
    
    return sparse_attention

def heun_integration(embeddings: np.ndarray, attention: np.ndarray, 
                     config: QFNNConfig) -> np.ndarray:
    """Apply Heun-Euler integration to evolve embeddings."""
    # Set dt based on embedding_dim and sequence_length (N/2, D/2)
    dt = config.dt_scale * np.sqrt(config.embedding_dim / config.sequence_length)
    
    # Store original embeddings for integration
    original_embeddings = embeddings.copy()
    
    # Store original norms for energy conservation
    original_norms = np.linalg.norm(original_embeddings, axis=1, keepdims=True)
    
    # Apply integration steps
    for _ in range(config.integration_steps):
        # Step 1: k1 = dt * F(ψ)
        k1 = dt * np.dot(attention, original_embeddings)
        
        # Step 2: ψ* = ψ + k1
        psi_star = original_embeddings + k1
        
        # Normalize intermediate state to preserve energy
        psi_star_norms = np.linalg.norm(psi_star, axis=1, keepdims=True)
        psi_star = psi_star * (original_norms / (psi_star_norms + EPSILON))
        
        # Recompute attention for intermediate state
        attention_star = compute_attention(psi_star, config)
        attention_star = apply_sparsity(attention_star, config)
        
        # Step 3: k2 = dt * F(ψ*)
        k2 = dt * np.dot(attention_star, psi_star)
        
        # Step 4: ψ' = ψ + (1/2)(k1 + k2)
        original_embeddings = original_embeddings + 0.5 * (k1 + k2)
        
        # Final normalization to preserve energy
        current_norms = np.linalg.norm(original_embeddings, axis=1, keepdims=True)
        original_embeddings = original_embeddings * (original_norms / (current_norms + EPSILON))
    
    return original_embeddings

def evaluate_sentence_understanding(original_embeddings: np.ndarray, 
                                   evolved_embeddings: np.ndarray,
                                   tokens_str: List[str]) -> dict:
    """
    Evaluate sentence understanding metrics based on linguistic patterns.
    
    This examines if:
    1. Semantically related words move closer together
    2. Sequential word order is preserved
    3. Natural language patterns emerge
    """
    D = len(tokens_str)
    
    # Extract circular components for phase analysis
    orig_circular = original_embeddings[:, :2]
    evol_circular = evolved_embeddings[:, :2]
    
    # Calculate pairwise distances
    orig_dists = np.zeros((D, D))
    evol_dists = np.zeros((D, D))
    
    for i in range(D):
        for j in range(D):
            orig_dists[i, j] = np.linalg.norm(orig_circular[i] - orig_circular[j])
            evol_dists[i, j] = np.linalg.norm(evol_circular[i] - evol_circular[j])
    
    # Semantic connections: common word types should cluster
    # Simple approach: detect nouns, verbs, adjectives, etc.
    word_types = {}
    nouns = ["quantum", "flux", "networks"]  # Example types
    verbs = ["like"]
    
    for i, token in enumerate(tokens_str):
        if token.lower() in nouns:
            word_types[i] = "noun"
        elif token.lower() in verbs:
            word_types[i] = "verb"
        else:
            word_types[i] = "other"
    
    # Calculate average distances between same word types
    same_type_dists_orig = []
    same_type_dists_evol = []
    
    for i in range(D):
        for j in range(i+1, D):
            if word_types[i] == word_types[j]:
                same_type_dists_orig.append(orig_dists[i, j])
                same_type_dists_evol.append(evol_dists[i, j])
    
    # Semantic clustering score
    semantic_score = 1.0
    if len(same_type_dists_orig) > 0:
        avg_orig = np.mean(same_type_dists_orig)
        avg_evol = np.mean(same_type_dists_evol)
        semantic_score = avg_orig / (avg_evol + EPSILON)
    
    # Sequential ordering: check if adjacent tokens remain close
    seq_dists_orig = []
    seq_dists_evol = []
    
    for i in range(D-1):
        seq_dists_orig.append(orig_dists[i, i+1])
        seq_dists_evol.append(evol_dists[i, i+1])
    
    # Sequential preservation score
    seq_score = 1.0
    if len(seq_dists_orig) > 0:
        avg_seq_orig = np.mean(seq_dists_orig)
        avg_seq_evol = np.mean(seq_dists_evol)
        seq_score = avg_seq_orig / (avg_seq_evol + EPSILON)
    
    # Natural language score: subject-verb-object structure
    # Simple approach: assume first token is subject, verb follows, then object
    nl_score = 1.0
    if D >= 3:
        subject_idx = 0
        verb_idx = 1
        object_idx = 2
        
        subject_verb_dist_orig = orig_dists[subject_idx, verb_idx]
        subject_verb_dist_evol = evol_dists[subject_idx, verb_idx]
        
        verb_object_dist_orig = orig_dists[verb_idx, object_idx]
        verb_object_dist_evol = evol_dists[verb_idx, object_idx]
        
        # Check if structure is preserved/enhanced
        sv_score = subject_verb_dist_orig / (subject_verb_dist_evol + EPSILON)
        vo_score = verb_object_dist_orig / (verb_object_dist_evol + EPSILON)
        nl_score = (sv_score + vo_score) / 2
    
    return {
        "semantic_score": semantic_score,
        "sequential_score": seq_score,
        "language_score": nl_score,
        "overall_understanding": (semantic_score + seq_score + nl_score) / 3
    }

def evaluate_embeddings(original_embeddings: np.ndarray, 
                        evolved_embeddings: np.ndarray,
                        tokens_str: List[str] = None) -> dict:
    """Evaluate embedding quality metrics."""
    # Extract circular components
    circular_orig = original_embeddings[:, :2]
    circular_evol = evolved_embeddings[:, :2]
    
    # Calculate phase coherence
    theta_orig = np.arctan2(circular_orig[:, 1], circular_orig[:, 0])
    theta_evol = np.arctan2(circular_evol[:, 1], circular_evol[:, 0])
    
    # Phase coherence is measured by how aligned the phases are
    orig_complex = np.exp(1j * theta_orig)
    evol_complex = np.exp(1j * theta_evol)
    
    orig_coherence = np.abs(np.mean(orig_complex))
    evol_coherence = np.abs(np.mean(evol_complex))
    
    # Energy conservation
    orig_energy = np.mean(np.linalg.norm(original_embeddings, axis=1))
    evol_energy = np.mean(np.linalg.norm(evolved_embeddings, axis=1))
    energy_conservation = evol_energy / (orig_energy + EPSILON)
    
    # Distance preservation
    D = original_embeddings.shape[0]
    orig_dists = np.zeros((D, D))
    evol_dists = np.zeros((D, D))
    
    for i in range(D):
        for j in range(D):
            orig_dists[i, j] = np.linalg.norm(original_embeddings[i] - original_embeddings[j])
            evol_dists[i, j] = np.linalg.norm(evolved_embeddings[i] - evolved_embeddings[j])
    
    # Flatten and compute correlation
    orig_dists_flat = orig_dists.flatten()
    evol_dists_flat = evol_dists.flatten()
    distance_corr = np.corrcoef(orig_dists_flat, evol_dists_flat)[0, 1]
    
    results = {
        "original_coherence": orig_coherence,
        "evolved_coherence": evol_coherence,
        "coherence_gain": evol_coherence - orig_coherence,
        "energy_conservation": energy_conservation,
        "distance_correlation": distance_corr
    }
    
    # Add sentence understanding metrics if tokens provided
    if tokens_str is not None:
        understanding_metrics = evaluate_sentence_understanding(
            original_embeddings, evolved_embeddings, tokens_str)
        results.update(understanding_metrics)
    
    return results

def run_ablation_test(tokens: List[int], 
                      tokens_str: List[str],
                      attention_mechanisms: List[AttentionMechanism],
                      radial_encodings: List[RadialEncoding],
                      cylinder_fields: List[CylinderField],
                      base_config: QFNNConfig) -> dict:
    """Run ablation tests across different configurations."""
    results = {}
    
    for attn_mech in attention_mechanisms:
        for rad_enc in radial_encodings:
            for cyl_field in cylinder_fields:
                # Create config variant
                variant_config = QFNNConfig(
                    vocab_size=base_config.vocab_size,
                    embedding_dim=base_config.embedding_dim,
                    sequence_length=base_config.sequence_length,
                    radial_min=base_config.radial_min,
                    radial_max=base_config.radial_max,
                    attention_mechanism=attn_mech,
                    radial_encoding=rad_enc,
                    cylinder_field=cyl_field,
                    dt_scale=base_config.dt_scale,
                    integration_steps=base_config.integration_steps,
                    sparsity=base_config.sparsity
                )
                
                # Generate embeddings
                original_embeddings = generate_embeddings(tokens, variant_config)
                
                # Compute attention
                attention = compute_attention(original_embeddings, variant_config)
                
                # Apply sparsity
                sparse_attention = apply_sparsity(attention, variant_config)
                
                # Apply integration
                evolved_embeddings = heun_integration(original_embeddings, sparse_attention, variant_config)
                
                # Evaluate results
                metrics = evaluate_embeddings(original_embeddings, evolved_embeddings, tokens_str)
                
                # Store results
                key = (attn_mech.value, rad_enc.value, cyl_field.value)
                results[key] = {
                    "metrics": metrics,
                    "original_embeddings": original_embeddings,
                    "evolved_embeddings": evolved_embeddings,
                    "attention": attention,
                    "sparse_attention": sparse_attention
                }
    
    return results

def visualize_results(results: dict, tokens_str: List[str], base_config: QFNNConfig):
    """Visualize ablation test results."""
    # Create figure for metrics comparison
    metrics_to_plot = [
        ("coherence_gain", "Coherence Gain"),
        ("energy_conservation", "Energy Conservation"),
        ("distance_correlation", "Distance Correlation"),
        ("overall_understanding", "Sentence Understanding")
    ]
    
    # Number of metrics to show
    n_metrics = len(metrics_to_plot)
    
    fig_metrics, ax_metrics = plt.subplots(1, n_metrics, figsize=(20, 5))
    
    # Extract configurations and metrics
    configs = list(results.keys())
    config_labels = [f"{cfg[0]}\n{cfg[1]}\n{cfg[2]}" for cfg in configs]
    
    # Plot each metric
    for i, (metric_key, metric_title) in enumerate(metrics_to_plot):
        if metric_key in results[configs[0]]["metrics"]:
            metric_values = [results[cfg]["metrics"].get(metric_key, 0) for cfg in configs]
            ax_metrics[i].bar(range(len(configs)), metric_values)
            ax_metrics[i].set_title(metric_title)
            ax_metrics[i].set_xticks(range(len(configs)))
            ax_metrics[i].set_xticklabels(config_labels, rotation=90)
    
    plt.tight_layout()
    
    # Find the best configuration for visualization
    best_configs = {}
    for metric_key, _ in metrics_to_plot:
        if metric_key in results[configs[0]]["metrics"]:
            metric_values = [results[cfg]["metrics"].get(metric_key, 0) for cfg in configs]
            best_idx = np.argmax(metric_values)
            best_configs[metric_key] = configs[best_idx]
    
    # Create figure for embeddings visualization
    best_config = best_configs.get("overall_understanding", configs[0])
    best_result = results[best_config]
    
    fig_emb, ax_emb = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original and evolved embeddings (first 2 dimensions)
    orig_emb = best_result["original_embeddings"]
    evol_emb = best_result["evolved_embeddings"]
    
    # Original embeddings
    ax_emb[0].scatter(orig_emb[:, 0], orig_emb[:, 1])
    for i, token in enumerate(tokens_str):
        ax_emb[0].annotate(token, (orig_emb[i, 0], orig_emb[i, 1]))
    ax_emb[0].set_title(f"Original Embeddings\n{best_config}")
    
    # Add unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax_emb[0].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    ax_emb[0].set_xlim(-1.5, 1.5)
    ax_emb[0].set_ylim(-1.5, 1.5)
    ax_emb[0].grid(True)
    
    # Evolved embeddings
    ax_emb[1].scatter(evol_emb[:, 0], evol_emb[:, 1])
    for i, token in enumerate(tokens_str):
        ax_emb[1].annotate(token, (evol_emb[i, 0], evol_emb[i, 1]))
    
    understanding = best_result["metrics"].get("overall_understanding", 0)
    ax_emb[1].set_title(f"Evolved Embeddings\nUnderstanding: {understanding:.4f}")
    
    # Add unit circle
    ax_emb[1].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    ax_emb[1].set_xlim(-1.5, 1.5)
    ax_emb[1].set_ylim(-1.5, 1.5)
    ax_emb[1].grid(True)
    
    plt.tight_layout()
    
    # Create figure for attention visualization
    fig_attn, ax_attn = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot attention matrices
    im1 = ax_attn[0].imshow(best_result["attention"], cmap="viridis")
    ax_attn[0].set_title("Attention Matrix")
    ax_attn[0].set_xticks(range(len(tokens_str)))
    ax_attn[0].set_yticks(range(len(tokens_str)))
    ax_attn[0].set_xticklabels(tokens_str)
    ax_attn[0].set_yticklabels(tokens_str)
    plt.colorbar(im1, ax=ax_attn[0])
    
    im2 = ax_attn[1].imshow(best_result["sparse_attention"], cmap="viridis")
    ax_attn[1].set_title(f"Sparse Attention (s={base_config.sparsity:.4f})")
    ax_attn[1].set_xticks(range(len(tokens_str)))
    ax_attn[1].set_yticks(range(len(tokens_str)))
    ax_attn[1].set_xticklabels(tokens_str)
    ax_attn[1].set_yticklabels(tokens_str)
    plt.colorbar(im2, ax=ax_attn[1])
    
    plt.tight_layout()
    plt.show()

def main():
    # Example sentence
    sentence = ["I", "like", "quantum", "flux", "neural", "networks"]
    tokens = list(range(len(sentence)))  # Convert to indices
    
    # Base configuration
    base_config = QFNNConfig(
        vocab_size=1000,
        embedding_dim=4,
        sequence_length=len(tokens),
        radial_min=0.3,
        radial_max=0.9,
        dt_scale=0.1,
        integration_steps=3,
        sparsity=OPTIMAL_SPARSITY
    )
    
    # Components to test
    attention_mechanisms = [
        AttentionMechanism.COSINE_PHASE,
        AttentionMechanism.NEGATIVE_DISTANCE,
        AttentionMechanism.FLUX_ATTENTION,
        AttentionMechanism.FLUX_WITH_AMPLITUDE
    ]
    
    radial_encodings = [
        RadialEncoding.GOLDEN_RATIO,
        RadialEncoding.FIBONACCI_MOD1
    ]
    
    cylinder_fields = [
        CylinderField.STANDARD,
        CylinderField.HARMONIC
    ]
    
    # Run ablation tests
    results = run_ablation_test(
        tokens, sentence, 
        attention_mechanisms, 
        radial_encodings,
        cylinder_fields,
        base_config
    )
    
    # Print summary of results
    print("Ablation Test Results:")
    print("======================")
    
    for (attn_mech, rad_enc, cyl_field), result in results.items():
        metrics = result["metrics"]
        print(f"Attention: {attn_mech}, Encoding: {rad_enc}, Field: {cyl_field}")
        print(f"  Coherence Gain: {metrics['coherence_gain']:.4f}")
        print(f"  Energy Conservation: {metrics['energy_conservation']:.4f}")
        print(f"  Distance Correlation: {metrics['distance_correlation']:.4f}")
        
        if "overall_understanding" in metrics:
            print(f"  Semantic Score: {metrics.get('semantic_score', 0):.4f}")
            print(f"  Sequential Score: {metrics.get('sequential_score', 0):.4f}")
            print(f"  Language Score: {metrics.get('language_score', 0):.4f}")
            print(f"  Overall Understanding: {metrics.get('overall_understanding', 0):.4f}")
        print()
    
    # Visualize results
    visualize_results(results, sentence, base_config)

if __name__ == "__main__":
    main()
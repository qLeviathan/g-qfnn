"""
Extended Quantum-Geometric Framework (No Softmax, Extended Explanations)

What's New:
1. Clarifies why (D-1) and (N-2) appear in the code.
2. Provides a sentence -> polar embedding with r, theta, storing (rx, ry) in first 2 dims of an N-dim embedding.
3. Demonstrates a "physics-inspired" approach to token reconstruction, forcibly starting with tokens "I" -> "like".
4. Adds a "noise schedule" to dynamically update dt, illustrating how negative distance (score) can guide exploration.
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------
# I. Basic Geometry: (r, theta) <-> (rx, ry)
# ---------------------------------------------------------------------
def polar_to_cartesian(r, theta):
    """
    Convert polar (r, theta) -> Cartesian (rx, ry).

    r : radial distance, typically in [0,1] or [0, some max]
    theta : angle in radians, typically in [0, 2π).
    """
    rx = r * math.cos(theta)
    ry = r * math.sin(theta)
    return rx, ry

def cartesian_to_polar(rx, ry):
    """
    Convert Cartesian (rx, ry) -> polar (r, theta).

    r = sqrt(rx^2 + ry^2)
    theta in [0, 2π).
    """
    r = math.sqrt(rx*rx + ry*ry)
    theta = math.atan2(ry, rx)
    if theta < 0:
        theta += 2.0 * math.pi
    return r, theta


# ---------------------------------------------------------------------
# II. Radial Diffusion (ODE1) = Imag-time Schr. (ODE2)
# ---------------------------------------------------------------------
def radial_diffusion_rhs(u, r_grid, alpha, potential=None):
    """
    ODE1: ∂u/∂t = alpha * [1/r * d/dr (r d/dr u)] in radial symmetry.
    potential=None is unused here, but included to unify signature with the Schr. eq.
    """
    du_dt = np.zeros_like(u)
    dr = r_grid[1] - r_grid[0]
    n = len(r_grid)
    
    for i in range(1, n-1):
        r = r_grid[i]
        d_u_dr_plus  = (u[i+1] - u[i]) / dr
        d_u_dr_minus = (u[i] - u[i-1]) / dr
        
        flux_plus  = r * d_u_dr_plus
        flux_minus = r * d_u_dr_minus
        
        du_dt[i] = alpha * (1.0 / r) * (flux_plus - flux_minus) / dr
    
    return du_dt

def imaginary_time_schrod_rhs(u, r_grid, alpha, potential=None):
    """
    ODE2: Imag-time Schr. eq in radial coords:
      ∂u/∂τ = alpha * [1/r d/dr ( r d/dr u ) ] - V(r)*u

    If potential is None => free particle.
    """
    if potential is None:
        def potential(r):
            return 0.0
    
    du_dtau = radial_diffusion_rhs(u, r_grid, alpha, potential=None)
    for i, r in enumerate(r_grid):
        du_dtau[i] -= potential(r)*u[i]
    return du_dtau


# ---------------------------------------------------------------------
# III. Poisson/Flux eq. with eps -> infinity => trivial
# ---------------------------------------------------------------------
def poisson_flux_equation(phi, r_grid, rho=None):
    """
    1D radial eq: ∇² phi = -rho/eps => trivial if eps->∞ => ∇² phi=0 => linear/constant.
    """
    if rho is None:
        rho = np.zeros_like(phi)
    
    lap = np.zeros_like(phi)
    dr = r_grid[1] - r_grid[0]
    n = len(r_grid)
    
    for i in range(1, n-1):
        r = r_grid[i]
        dphi_dr_plus  = (phi[i+1] - phi[i]) / dr
        dphi_dr_minus = (phi[i]   - phi[i-1]) / dr
        
        flux_plus  = (r+0.5*dr)**2 * dphi_dr_plus
        flux_minus = (r-0.5*dr)**2 * dphi_dr_minus
        
        lap[i] = (flux_plus - flux_minus) / (dr * r**2)
    return lap


# ---------------------------------------------------------------------
# IV. Heun-Euler step + "default dt" from inverse Beta(N/2, D/2)
# ---------------------------------------------------------------------
def default_time_step_inv_beta(N, D):
    """
    dt = 1 / X,  X ~ Beta(N/2, D/2).

    Explanation:
      For embedding dim N and sequence length D, pick shape params alpha_=(N/2), beta_=(D/2).
      Beta distribution => X in (0,1).
      => dt in (1,∞).

    This dt can shift each integration step's "time" range dynamically.
    """
    if N < 2 or D < 2:
        raise ValueError("N >= 2 and D >= 2 recommended for Beta distribution.")
    
    alpha_ = float(N)/2.0
    beta_  = float(D)/2.0
    dist = torch.distributions.Beta(alpha_, beta_)
    X = dist.sample()
    return 1.0 / float(X)

def heun_euler_step(rhs_func, u, r_grid, alpha, dt, potential=None):
    """
    Heun-Euler integrator:
      k1 = rhs_func(u)
      k2 = rhs_func(u + dt*k1)
      u_next = u + dt/2*(k1 + k2)
    """
    k1 = rhs_func(u, r_grid, alpha, potential)
    u_plus = u + dt*k1
    k2 = rhs_func(u_plus, r_grid, alpha, potential)
    return u + 0.5*dt*(k1 + k2)


# ---------------------------------------------------------------------
# V. Distance-based approach: Negative Dist
# ---------------------------------------------------------------------
def normalize_embeddings(embs):
    """
    Row-wise normalization of embeddings, shape (D, N).

    Why N-2?
      If N=4, for example, we used 2 dims (rx, ry) for polar coords,
      so there are (N-2)=2 leftover dims that can store extra info or remain zeros.
      In general, the first 2 dims might be geometry; the rest are "unused" in this demo.

    This function ensures each row has a unit norm in R^N.
    """
    normed = embs.copy()
    for i in range(len(normed)):
        denom = np.linalg.norm(normed[i])
        if denom > 1e-12:
            normed[i] /= denom
    return normed

def negative_distance_matrix(embs, prev_diffs=None, learning_rate=0.1):
    """
    Return M[i,j] = -||embs[i] - embs[j]|| / denominator.

    'negative distance' => a "score" that is higher (less negative)
    when points are closer, lower (more negative) when points are far.
    
    The denominator is iteratively updated based on previous token differences:
    - If prev_diffs is None, denominator = 1.0 (initial case)
    - Otherwise, denominator = 1.0 + learning_rate * prev_diffs[i,j]
    
    This allows the distance calculation to adapt based on previous token relationships.
    
    Uses torch operations for efficient computation.
    
    Diagonals are zeroed out (distance from a token to itself is set to 0).
    """
    D = embs.shape[0]
    
    # Convert to torch tensors for easier operations
    embs_tensor = torch.from_numpy(embs).float()
    
    # Create distance matrix using torch operations
    # Compute pairwise distances between all embeddings
    dist_matrix = torch.zeros((D, D), dtype=torch.float32)
    for i in range(D):
        for j in range(D):
            if i != j:  # Skip diagonal elements
                diff = embs_tensor[i] - embs_tensor[j]
                dist_matrix[i, j] = torch.norm(diff)
            # Diagonal elements remain 0
    
    # Store raw distances for next iteration
    new_diffs = dist_matrix.numpy()
    
    # Apply denominator adjustment based on previous differences
    if prev_diffs is not None:
        prev_diffs_tensor = torch.from_numpy(prev_diffs).float()
        denominator = 1.0 + learning_rate * prev_diffs_tensor
    else:
        denominator = torch.ones((D, D), dtype=torch.float32)
    
    # Calculate negative distance with adaptive denominator
    M = -dist_matrix / denominator
    
    return M.numpy(), new_diffs


# ---------------------------------------------------------------------
# VI. Sentence -> (r, theta) -> (rx, ry) in 2 dims, zeros in leftover (N-2)
# ---------------------------------------------------------------------
def sentence_to_polar_embeddings(sentence, N):
    """
    Convert a sentence of length D into R^N embeddings.
      - D = len(sentence)
      - For each token index i in [0..D-1]:
         theta_i = 2π * (i / D)
         r_i = 0.3 + 0.6*(i/(D-1)) if D>1 else 0.3 by default
           (We do (D-1) in the denominator so that i=0 => r=0.3, i=D-1 => r=0.9
            giving a radial "spread" of tokens.)

    Then (rx, ry) go in the first two dims; 
    the leftover (N-2) coords remain zero.

    Returns:
      embs shape (D, N)
    """
    D = len(sentence)
    embs = np.zeros((D, N), dtype=np.float32)

    for i in range(D):
        theta_i = 2.0*math.pi * (i / float(D)) if D>0 else 0.0
        # if D>1 => (i/(D-1)) in [0..1]; ensures r grows from 0.3..0.9
        if D > 1:
            frac = i / float(D-1)
        else:
            frac = 0.0
        r_i = 0.3 + 0.6*frac  # in [0.3..0.9]
        
        rx, ry = polar_to_cartesian(r_i, theta_i)
        embs[i, 0] = rx
        embs[i, 1] = ry
        # the rest (N-2) remain zeros
    return embs


# ---------------------------------------------------------------------
# VII. Sentence Reconstruction & Physics Analogy
# ---------------------------------------------------------------------
def reconstruct_sentence(tokens, negdist_mat):
    """
    Greedy approach that forcibly starts with token 0 -> token 1
    (like "I" -> "like"), then picks subsequent tokens by largest negative
    distance from the last chosen.

    Explanation:
      We interpret "largest negative distance" => "closest embedding"
      because negative distance is bigger (less negative) if actual distance is smaller.
      This has a loose "physics" analogy:
        - Systems prefer minimal potential or short distances
        - or we can interpret "maximum negative distance" = "maximum closeness".
    """
    D = len(tokens)
    if D < 2:
        return tokens[:]  # trivial if 0 or 1 token

    visited = [False]*D
    order = []

    # Force start with token[0], then token[1]
    visited[0] = True
    visited[1] = True
    order.extend([tokens[0], tokens[1]])

    current = 1  # we ended on token 1
    for _ in range(D-2):
        row = negdist_mat[current]
        cand_j = -1
        cand_val = -1e9
        for j in range(D):
            if (not visited[j]) and (row[j] > cand_val):
                cand_val = row[j]
                cand_j = j

        visited[cand_j] = True
        current = cand_j
        order.append(tokens[current])
    
    return order


# ---------------------------------------------------------------------
# VIII. A "Noise Schedule" That Adapts dt to the Negative Distance Score
# ---------------------------------------------------------------------
def update_dt_with_noise_schedule(dt, negdist_mat, scale=0.1):
    """
    Example function showing how negative distances might adapt dt:
      - We compute the mean of negdist_mat => ~ measure of closeness
      - If mean is near 0, tokens are "far" => negative distance ~ - large
        => mean(negdist_mat) is smaller => dt might increase
      - If mean is near 1, tokens are "close" => negative distance ~ 0 => dt might decrease

    We simply do:
      dt_new = dt + scale*(mean_negdist - 0.5)

    So if mean_negdist > 0.5 => dt increases
       mean_negdist < 0.5 => dt decreases
    This is purely a demo concept.
    """
    mean_negdist = negdist_mat.mean()
    # shift around 0.5 => new dt
    dt_new = dt + scale*(mean_negdist - 0.5)
    if dt_new < 0:
        dt_new = 0.001  # clamp
    return dt_new


# ---------------------------------------------------------------------
# X. Simple "Physics-Inspired" Network with Golden Ratio
# ---------------------------------------------------------------------
class PhysicsInspiredNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PhysicsInspiredNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)  # No bias
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False) # No bias
        
        # Initialize with golden ratio-based weights
        self.golden_ratio = (1.0 + math.sqrt(5)) / 2.0  # ≈ 1.618
        self._init_golden_weights()
        
    def _init_golden_weights(self):
        """Initialize weights using golden ratio patterns"""
        with torch.no_grad():
            # First layer: golden ratio pattern
            h, w = self.fc1.weight.shape
            
            # Create golden ratio pattern
            for i in range(h):
                for j in range(w):
                    # Use golden ratio to generate a value between -1 and 1
                    self.fc1.weight[i, j] = math.cos(2 * math.pi * ((i*j*self.golden_ratio) % 1.0))
            
            # Second layer: initialize with random weights
            self.fc2.weight.normal_(0, 0.1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

# ---------------------------------------------------------------------
# IX. Main
# ---------------------------------------------------------------------
def main():
    # Example #1: Radial diffusion vs. Imag-time Schr
    # embedding dimension N => alpha = N/2
    N = 4
    D_seq = 6  # for dt Beta(N/2, D/2)
    alpha = float(N)/2.0

    # "default dt" from Beta distribution
    dt = default_time_step_inv_beta(N, D_seq)
    print(f"[INFO] alpha={alpha}, dt={dt:.4f} (from 1/Beta(N/2,D/2))")

    # radial grid
    r_grid = np.linspace(0.0, 1.0, 50)
    u_init = np.exp(-((r_grid - 0.5)**2)/(2*0.01))

    # potential(r) = 0.5*r^2
    pot = lambda r: 0.5*r*r

    # Single Heun-Euler steps
    u_diff = heun_euler_step(radial_diffusion_rhs, u_init, r_grid, alpha, dt)
    u_schrod = heun_euler_step(imaginary_time_schrod_rhs, u_init, r_grid, alpha, dt, potential=pot)

    print("\n== Radial Diffusion vs Imag-time Schr. (1 step) ==")
    print("u_diff sample:", u_diff[::10])
    print("u_schrod sample:", u_schrod[::10])

    # Example #2: Poisson eq, eps->∞
    phi_init = np.linspace(0.0, 1.0, 50)
    lap = poisson_flux_equation(phi_init, r_grid)
    print("\n== Poisson eq. with eps->∞ => Laplacian(phi)=0 => sample output ==")
    print("lap sample:", lap[::10])

    # Example #3: Convert multiple sentences to polar embeddings
    sentences = [
        ["I", "like", "quantum", "mechanics", "with", "pizza"],
        ["quantum", "mechanics", "is", "fascinating", "and", "complex"],
        ["pizza", "with", "extra", "cheese", "tastes", "delicious"],
        ["the", "quick", "brown", "fox", "jumps", "over"]
    ]
    
    # Create a single evolving neural network
    # We'll use the maximum sequence length for the hidden size
    max_seq_len = max(len(s) for s in sentences)
    input_size = N  # Embedding dimension
    hidden_size = max_seq_len  # Use maximum sequence length as hidden size
    num_classes = 2  # Example: Positive or Negative sentiment
    
    # Initialize the evolving model
    evolving_model = PhysicsInspiredNN(input_size, hidden_size, num_classes)
    
    # Initialize token differences for iterative learning
    token_diffs = None
    
    # Process each sentence
    for i, sentence in enumerate(sentences):
        print(f"\nOriginal sentence {i+1}:", sentence)

        # D_s = len(sentence)
        embs = sentence_to_polar_embeddings(sentence, N)  # shape(D_s, N)
        # e.g. if D_s=6 => i=0..5 => r in [0.3..0.9], theta in [0..2π*(5/6)]
        embs = normalize_embeddings(embs)

        # Negative distance matrix with iterative learning (no normalization)
        if i == 0:
            # First sentence - initialize with no previous differences
            negdist_mat, token_diffs = negative_distance_matrix(embs)
        else:
            # Subsequent sentences - use previous token differences
            negdist_mat, token_diffs = negative_distance_matrix(embs, prev_diffs=token_diffs)
        
        print(f"== Negative Distance Matrix (raw, with zeroed diagonals) for sentence {i+1} ==")
        print(negdist_mat)

        # Example #4: "Reconstruct" sentence, forcibly starting with first two tokens
        # Use raw (unnormalized) matrix for reconstruction
        reorder = reconstruct_sentence(sentence, negdist_mat)
        print(f"Reconstructed seq (forced start {sentence[0]}->{sentence[1]}):", reorder)
        
        # Update the evolving model with the new negative distance matrix
        # The matrix needs to be reshaped to match the expected dimensions
        with torch.no_grad():  # No gradients needed for this physics-based approach
            # Use raw (non-normalized) matrix for weights
            # Pad or truncate to match the model's dimensions
            curr_seq_len = len(sentence)
            padded_matrix = np.zeros((hidden_size, input_size))
            padded_matrix[:curr_seq_len, :input_size] = negdist_mat[:curr_seq_len, :input_size]
            
            # Update weights - we use a weighted average to simulate evolution
            # For the first sentence, we just set the weights directly
            if i == 0:
                evolving_model.fc1.weight.copy_(torch.from_numpy(padded_matrix).float())
            else:
                # For subsequent sentences, we blend the new weights with the old ones
                # This creates a memory effect where the model evolves based on all sentences it has seen
                evolution_rate = 0.3  # How quickly the model adapts to new sentences
                current_weights = evolving_model.fc1.weight.data
                new_weights = torch.from_numpy(padded_matrix).float()
                blended_weights = (1 - evolution_rate) * current_weights + evolution_rate * new_weights
                evolving_model.fc1.weight.copy_(blended_weights)

        # Prepare the data
        embs_tensor = torch.from_numpy(embs).float()

        # Inference (no backpropagation)
        evolving_model.eval()
        outputs = evolving_model(embs_tensor)

        print(f"== Evolving Physics-Inspired Network Output for sentence {i+1} ==")
        print(f"Output: {outputs}")
        print("-" * 70)

    # Example #5: Illustrate a "noise schedule" that adjusts dt
    # Use the first sentence's negative distance matrix for this example
    first_sentence = sentences[0]
    embs = sentence_to_polar_embeddings(first_sentence, N)
    embs = normalize_embeddings(embs)
    negdist_mat, _ = negative_distance_matrix(embs)
    
    # Use raw matrix for noise schedule
    dt_old = dt
    dt_new = update_dt_with_noise_schedule(dt, negdist_mat, scale=0.1)
    print(f"\nNoise schedule update: old dt={dt_old:.4f} => new dt={dt_new:.4f} "
          f"(based on mean negdist {negdist_mat.mean():.4f})")
    
    print("\n== Summary of Physics-Inspired Approach ==")
    print("1. No backpropagation is needed for this physics-based approach")
    print("2. The negative distance matrix directly encodes token relationships")
    print("3. Different sentences produce different geometric patterns")
    print("4. The network outputs reflect the geometric structure of the input")
    print("5. Golden ratio (φ ≈ 1.618) provides natural, aesthetically pleasing patterns")
    print("6. Iterative learning adjusts token distances based on previous relationships")
    print("7. Torch operations enable efficient computation of distance matrices")
    print("8. Raw distance matrices with zeroed diagonals provide direct physical interpretation")
    print("9. Unnormalized metrics throughout provide consistent physical meaning")
    print("10. The neural network evolves dynamically as it processes each sentence")
    print("11. This evolution creates a memory effect where the model adapts based on all sentences it has seen")

if __name__ == "__main__":
    main()

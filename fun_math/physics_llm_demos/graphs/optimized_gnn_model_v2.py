"""
Optimized Quantum Flux Graph Neural Network for LLMs - Version 2

This implementation provides a highly vectorized version of the QuantumFlux GNN model
using einsum operations and tensor optimizations throughout. It eliminates loops
and maximizes parallel computation for improved performance.
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from transformers import GPT2Tokenizer


class VectorizedQuantumGeometricEmbedding(nn.Module):
    """
    Fully vectorized implementation of quantum-geometric embeddings with buffer reuse.
    
    Tensor shapes:
    - Input tokens: [batch_size, seq_len]
    - Output embeddings: [batch_size, seq_len, embedding_dim]
    
    Buffers:
    - positions: [max_seq_len] - Positions for each token
    - thetas: [max_seq_len] - Angular positions
    - rs: [max_seq_len] - Radial positions
    - cos_thetas: [max_seq_len] - Cosine of angular positions
    - sin_thetas: [max_seq_len] - Sine of angular positions
    """
    def __init__(self, embedding_dim=64, use_pretrained=False, vocab_size=None, max_seq_len=512):
        super(VectorizedQuantumGeometricEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        if use_pretrained and vocab_size is not None:
            self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
            self._init_golden_weights()
        else:
            self.token_embedding = None
        
        # Register buffers for position-related tensors
        self.register_buffer('positions', torch.arange(max_seq_len).float())
        self.register_buffer('thetas', torch.zeros(max_seq_len))
        self.register_buffer('rs', torch.zeros(max_seq_len))
        self.register_buffer('cos_thetas', torch.zeros(max_seq_len))
        self.register_buffer('sin_thetas', torch.zeros(max_seq_len))
        
        # Pre-compute position-related tensors
        self._precompute_position_tensors()
    
    def _precompute_position_tensors(self):
        """Pre-compute position-related tensors for reuse"""
        with torch.no_grad():
            # Compute theta values (angular positions) - vectorized
            self.thetas.copy_(2.0 * math.pi * (self.positions / self.max_seq_len))
            
            # Compute r values (radial positions) - vectorized
            self.rs.copy_(0.3 + 0.6 * (self.positions / max(1, self.max_seq_len - 1)))
            
            # Compute cosine and sine of thetas
            self.cos_thetas.copy_(torch.cos(self.thetas))
            self.sin_thetas.copy_(torch.sin(self.thetas))
    
    def _init_golden_weights(self):
        """Initialize weights using golden ratio patterns with vectorized operations"""
        with torch.no_grad():
            # Golden ratio ≈ 1.618
            golden_ratio = (1.0 + math.sqrt(5)) / 2.0
            
            # Get embedding weights
            weights = self.token_embedding.weight  # [vocab_size, embedding_dim]
            
            # Create indices for vectorized operations
            vocab_size, dim = weights.shape
            i_indices = torch.arange(vocab_size).unsqueeze(1).expand(-1, dim)  # [vocab_size, embedding_dim]
            j_indices = torch.arange(dim).unsqueeze(0).expand(vocab_size, -1)  # [vocab_size, embedding_dim]
            
            # Apply golden ratio pattern using vectorized operations
            pattern = torch.cos(2 * math.pi * ((i_indices * j_indices * golden_ratio) % 1.0))
            
            # Set the weights
            self.token_embedding.weight.copy_(pattern)
    
    def forward(self, tokens):
        """
        Forward pass to convert tokens to embeddings with fully vectorized operations.
        Uses precomputed buffers for position-related tensors.
        
        Args:
            tokens: Tensor of token indices [batch_size, seq_len]
            
        Returns:
            Tensor of embeddings [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # Create embeddings tensor
        embeddings = torch.zeros(batch_size, seq_len, self.embedding_dim, device=device)
        
        # Use precomputed position tensors if sequence length is within buffer size
        if seq_len <= self.max_seq_len:
            # Get slices of precomputed tensors
            rs_slice = self.rs[:seq_len].to(device)
            cos_thetas_slice = self.cos_thetas[:seq_len].to(device)
            sin_thetas_slice = self.sin_thetas[:seq_len].to(device)
            
            # Compute rx and ry values (first 2 dimensions) using precomputed values
            # Expand dimensions for broadcasting: [batch_size, seq_len]
            embeddings[:, :, 0] = rs_slice.unsqueeze(0) * cos_thetas_slice.unsqueeze(0)
            embeddings[:, :, 1] = rs_slice.unsqueeze(0) * sin_thetas_slice.unsqueeze(0)
        else:
            # Fall back to computing on the fly for sequences longer than buffer size
            positions = torch.arange(seq_len, device=device).float()
            thetas = 2.0 * math.pi * (positions / seq_len)
            
            if seq_len > 1:
                rs = 0.3 + 0.6 * (positions / (seq_len - 1))
            else:
                rs = torch.tensor([0.3], device=device)
            
            embeddings[:, :, 0] = rs.unsqueeze(0) * torch.cos(thetas).unsqueeze(0)
            embeddings[:, :, 1] = rs.unsqueeze(0) * torch.sin(thetas).unsqueeze(0)
        
        # If using token embeddings, add them to the remaining dimensions
        if self.token_embedding is not None:
            token_embs = self.token_embedding(tokens)  # [batch_size, seq_len, embedding_dim]
            # Use only the remaining dimensions (embedding_dim - 2)
            if self.embedding_dim > 2:
                embeddings[:, :, 2:] = token_embs[:, :, :(self.embedding_dim-2)]
        
        # Normalize embeddings using einsum for efficiency
        # 'bsi,bsi->bs' computes the squared norm for each embedding vector
        norms = torch.sqrt(torch.clamp(
            torch.einsum('bsi,bsi->bs', embeddings, embeddings),
            min=1e-12
        )).unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Normalize with broadcasting
        normalized_embeddings = embeddings / norms
        
        return normalized_embeddings


class VectorizedQuantumGeometricAttention(nn.Module):
    """
    Fully vectorized implementation of quantum-geometric attention with buffer reuse.
    
    Tensor shapes:
    - Input embeddings: [batch_size, seq_len, embedding_dim]
    - Output attention weights: [batch_size, seq_len, seq_len]
    
    Buffers:
    - eye_mask: [1, max_seq_len, max_seq_len] - Mask to zero out diagonal elements
    - seq_len_masks: Dict of masks for different sequence lengths
    - dist_buffers: Dict of distance calculation buffers for different sequence lengths
    """
    def __init__(self, temperature=1.0, max_seq_len=512):
        super(VectorizedQuantumGeometricAttention, self).__init__()
        self.temperature = temperature
        self.max_seq_len = max_seq_len
        
        # Register buffers for reuse
        self.register_buffer('eye_mask', 1.0 - torch.eye(max_seq_len).unsqueeze(0))
        
        # Cache for sequence-specific buffers
        self.seq_len_masks = {}
        self.dist_buffers = {}
    
    def forward(self, embeddings):
        """
        Forward pass to compute attention weights with fully vectorized operations.
        Uses cached buffers for efficiency.
        
        Args:
            embeddings: Tensor of shape [batch_size, seq_len, embedding_dim]
            
        Returns:
            Attention weights of shape [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, embedding_dim = embeddings.shape
        device = embeddings.device
        
        # Get appropriate eye mask for this sequence length
        if seq_len <= self.max_seq_len:
            # Use slice of pre-computed eye mask
            eye_mask = self.eye_mask[:, :seq_len, :seq_len].to(device)
        else:
            # Create new eye mask for this sequence length
            eye_mask = 1.0 - torch.eye(seq_len, device=device).unsqueeze(0)
        
        # Check if we have cached buffers for this sequence length and batch size
        buffer_key = f"{batch_size}_{seq_len}_{device}"
        if buffer_key not in self.dist_buffers:
            # Create and cache buffers for this sequence length and batch size
            self.dist_buffers[buffer_key] = {
                'norms_buffer': torch.zeros(batch_size, seq_len, device=device),
                'dot_products_buffer': torch.zeros(batch_size, seq_len, seq_len, device=device),
                'squared_dists_buffer': torch.zeros(batch_size, seq_len, seq_len, device=device),
                'dist_matrix_buffer': torch.zeros(batch_size, seq_len, seq_len, device=device),
                'neg_dist_buffer': torch.zeros(batch_size, seq_len, seq_len, device=device),
                'scaled_neg_dist_buffer': torch.zeros(batch_size, seq_len, seq_len, device=device)
            }
        
        # Get buffers for this sequence length and batch size
        buffers = self.dist_buffers[buffer_key]
        
        # Compute squared norms using einsum - [batch_size, seq_len]
        # 'bsi,bsi->bs' means sum the product of each element in the embedding with itself
        norms_squared = torch.einsum('bsi,bsi->bs', embeddings, embeddings)
        buffers['norms_buffer'].copy_(norms_squared)
        
        # Compute dot products between all pairs of embeddings - [batch_size, seq_len, seq_len]
        # Using bmm for efficiency
        dot_products = torch.bmm(embeddings, embeddings.transpose(1, 2))
        buffers['dot_products_buffer'].copy_(dot_products)
        
        # Compute squared distances using broadcasting - [batch_size, seq_len, seq_len]
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        squared_dists = buffers['squared_dists_buffer']
        squared_dists.copy_(norms_squared.unsqueeze(2) + norms_squared.unsqueeze(1) - 2 * dot_products)
        
        # Take square root to get actual distances, and handle numerical issues
        dist_matrix = torch.sqrt(torch.clamp(squared_dists, min=1e-12))
        buffers['dist_matrix_buffer'].copy_(dist_matrix)
        
        # Zero out diagonal elements (distance from a token to itself)
        dist_matrix.mul_(eye_mask)
        
        # Calculate negative distance
        neg_dist = torch.neg(dist_matrix)
        buffers['neg_dist_buffer'].copy_(neg_dist)
        
        # Apply temperature scaling
        scaled_neg_dist = torch.div(neg_dist, self.temperature)
        buffers['scaled_neg_dist_buffer'].copy_(scaled_neg_dist)
        
        # Convert to attention weights using softmax
        # This creates a probability distribution over the sequence
        attention_weights = F.softmax(scaled_neg_dist, dim=-1)
        
        return attention_weights


class VectorizedMessagePassing(MessagePassing):
    """
    Fully vectorized implementation of message passing.
    
    Tensor shapes:
    - Node features: [num_nodes, in_channels]
    - Edge index: [2, num_edges]
    - Edge weights: [num_edges]
    - Output node features: [num_nodes, out_channels]
    """
    def __init__(self, in_channels, out_channels):
        super(VectorizedMessagePassing, self).__init__(aggr='add')
        
        # Linear transformation for messages - no bias for efficiency
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        
        # Edge weighting parameter
        self.edge_weight = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        self.edge_weight.data.fill_(0.1)
        nn.init.xavier_uniform_(self.lin.weight)
    
    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass for message passing with vectorized operations.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Transform node features using linear layer
        # This is more efficient than transforming in the message function
        x = self.lin(x)  # [num_nodes, out_channels]
        
        # Scale edge weights with learnable parameter - vectorized
        edge_weight = edge_weight * self.edge_weight  # [num_edges]
        
        # Start propagating messages
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
    
    def message(self, x_j, edge_weight):
        """
        Message function defining how messages are computed.
        
        Args:
            x_j: Source node features [num_edges, out_channels]
            edge_weight: Edge weights [num_edges]
            
        Returns:
            Messages to be aggregated [num_edges, out_channels]
        """
        # Weight messages by edge weights using broadcasting
        return edge_weight.view(-1, 1) * x_j


class VectorizedQuantumFluxGNN(nn.Module):
    """
    Fully vectorized implementation of the Quantum Flux GNN model with buffer reuse.
    
    Tensor shapes:
    - Input tokens: [batch_size, seq_len]
    - Embeddings: [batch_size, seq_len, embedding_dim]
    - Attention weights: [batch_size, seq_len, seq_len]
    - Graph node features: [batch_size * seq_len, embedding_dim]
    - Graph edge index: [2, batch_size * num_edges_per_graph]
    - Graph edge weights: [batch_size * num_edges_per_graph]
    - Output logits: [batch_size, seq_len, vocab_size]
    
    Buffers:
    - diag_masks: Dict of diagonal masks for different sequence lengths
    - graph_buffers: Dict of graph construction buffers for different sequence lengths
    """
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=3, dropout=0.1, sparsity_threshold=0.01, max_seq_len=512):
        super(VectorizedQuantumFluxGNN, self).__init__()
        
        # Model parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Embedding layer
        self.embedding = VectorizedQuantumGeometricEmbedding(
            embedding_dim, 
            use_pretrained=True, 
            vocab_size=vocab_size,
            max_seq_len=max_seq_len
        )
        
        # Attention mechanism
        self.attention = VectorizedQuantumGeometricAttention(
            temperature=0.5,
            max_seq_len=max_seq_len
        )
        
        # Graph neural network layers
        self.gnn_layers = nn.ModuleList()
        
        # First layer takes embedding_dim as input
        self.gnn_layers.append(VectorizedMessagePassing(embedding_dim, hidden_dim))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.gnn_layers.append(VectorizedMessagePassing(hidden_dim, hidden_dim))
        
        # Last layer produces output with hidden_dim dimensions
        self.gnn_layers.append(VectorizedMessagePassing(hidden_dim, hidden_dim))
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Sparsity threshold for graph construction
        self.sparsity_threshold = sparsity_threshold
        
        # Enable gradient checkpointing for memory efficiency
        self.use_checkpointing = True
        
        # Cache for sequence-specific buffers
        self.diag_masks = {}
        self.graph_buffers = {}
        
        # Register a buffer for the diagonal mask of the maximum sequence length
        self.register_buffer(
            'max_diag_mask', 
            ~torch.eye(max_seq_len, dtype=torch.bool).unsqueeze(0)
        )
    
    def _prepare_sparse_graph_fully_vectorized(self, embeddings, attention_weights):
        """
        Convert sequence data to sparse graph data with fully vectorized operations.
        Uses cached buffers for efficiency.
        
        Args:
            embeddings: Token embeddings [batch_size, seq_len, embedding_dim]
            attention_weights: Attention weights [batch_size, seq_len, seq_len]
            
        Returns:
            Batched graph data object
        """
        batch_size, seq_len, _ = embeddings.shape
        device = embeddings.device
        
        # Get or create diagonal mask for this sequence length
        seq_key = f"{seq_len}_{device}"
        if seq_key not in self.diag_masks:
            if seq_len <= self.max_seq_len:
                # Use slice of pre-computed diagonal mask
                diag_mask = self.max_diag_mask[:, :seq_len, :seq_len].to(device)
            else:
                # Create new diagonal mask for this sequence length
                diag_mask = ~torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0)
            
            # Cache the mask for future use
            self.diag_masks[seq_key] = diag_mask
        else:
            # Use cached mask
            diag_mask = self.diag_masks[seq_key]
        
        # Check if we have cached buffers for this sequence length and batch size
        graph_key = f"{batch_size}_{seq_len}_{device}"
        if graph_key not in self.graph_buffers:
            # Create and cache buffers for this sequence length and batch size
            self.graph_buffers[graph_key] = {
                'mask_buffer': torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool, device=device),
                'batch_indices': torch.arange(batch_size, device=device).view(-1, 1).repeat(1, seq_len).view(-1),
                'fallback_src_idx': torch.arange(batch_size * seq_len, device=device),
                'fallback_dst_idx': torch.arange(batch_size * seq_len, device=device),
                'fallback_edge_index': torch.stack([
                    torch.arange(batch_size * seq_len, device=device),
                    torch.arange(batch_size * seq_len, device=device)
                ], dim=0),
                'fallback_edge_weight': torch.ones(batch_size * seq_len, device=device)
            }
        
        # Get buffers for this sequence length and batch size
        buffers = self.graph_buffers[graph_key]
        
        # Apply threshold to attention weights - vectorized
        # This creates a sparse adjacency matrix
        mask = (attention_weights > self.sparsity_threshold)  # [batch_size, seq_len, seq_len]
        
        # Remove self-loops - vectorized
        mask = mask & diag_mask.expand(batch_size, -1, -1)  # [batch_size, seq_len, seq_len]
        
        # Get batch indices tensor for all nodes
        batch_indices = buffers['batch_indices']  # [batch_size * seq_len]
        
        # Flatten embeddings to create node features for the batched graph
        x = embeddings.reshape(-1, embeddings.size(2))  # [batch_size * seq_len, embedding_dim]
        
        # Create edge indices and weights using vectorized operations
        # First, get the indices where mask is True
        edge_indices = mask.nonzero(as_tuple=True)  # tuple of 3 tensors: (batch_idx, src_idx, dst_idx)
        
        if len(edge_indices[0]) > 0:  # Check if there are any edges
            # Extract batch indices, source indices, and destination indices
            batch_idx, src_idx, dst_idx = edge_indices
            
            # Adjust source and destination indices to account for batching
            # Each graph's nodes start at (graph_idx * seq_len)
            src_idx_adjusted = batch_idx * seq_len + src_idx  # [num_edges]
            dst_idx_adjusted = batch_idx * seq_len + dst_idx  # [num_edges]
            
            # Stack to create edge_index tensor
            edge_index = torch.stack([src_idx_adjusted, dst_idx_adjusted], dim=0)  # [2, num_edges]
            
            # Get edge weights from attention weights using the same indices
            edge_weight = attention_weights[batch_idx, src_idx, dst_idx]  # [num_edges]
            
            # Create batched graph
            batched_graph = Data(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                batch=batch_indices,
                num_nodes=batch_size * seq_len
            )
            
            return batched_graph
        else:
            # Handle case with no edges (all attention weights below threshold)
            # Create a minimal graph with self-loops to avoid errors
            # This is a fallback that should rarely happen with proper threshold
            
            # Use pre-allocated buffers for fallback case
            edge_index = buffers['fallback_edge_index']  # [2, batch_size * seq_len]
            edge_weight = buffers['fallback_edge_weight']  # [batch_size * seq_len]
            
            batched_graph = Data(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                batch=batch_indices,
                num_nodes=batch_size * seq_len
            )
            
            return batched_graph
    
    def forward(self, tokens):
        """
        Forward pass through the model with fully vectorized operations.
        
        Args:
            tokens: Input token indices [batch_size, seq_len]
            
        Returns:
            Logits for next token prediction [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # Get token embeddings - vectorized
        embeddings = self.embedding(tokens)  # [batch_size, seq_len, embedding_dim]
        
        # Compute attention weights - vectorized
        if self.use_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            attention_weights = checkpoint(self.attention, embeddings)
        else:
            attention_weights = self.attention(embeddings)  # [batch_size, seq_len, seq_len]
        
        # Convert to sparse graph representation - fully vectorized
        graph = self._prepare_sparse_graph_fully_vectorized(embeddings, attention_weights)
        
        # Process the graph through GNN layers - vectorized
        x = graph.x  # [batch_size * seq_len, embedding_dim]
        
        # Apply GNN layers with residual connections
        for i, gnn_layer in enumerate(self.gnn_layers):
            # Apply message passing
            x_new = gnn_layer(x, graph.edge_index, graph.edge_weight)
            
            # Apply residual connection except for first layer
            if i > 0:
                x = x + x_new
            else:
                x = x_new
            
            # Apply normalization and dropout
            x = self.layer_norm(x)
            x = self.dropout(x)
        
        # Project to vocabulary size
        logits = self.output_projection(x)  # [batch_size * seq_len, vocab_size]
        
        # Reshape logits back to [batch_size, seq_len, vocab_size]
        logits = logits.view(batch_size, seq_len, -1)
        
        return logits
    
    def predict_next_token(self, tokens, temperature=1.0, top_k=5):
        """
        Predict the next token given a sequence of tokens.
        
        Args:
            tokens: Input token indices [batch_size, seq_len]
            temperature: Sampling temperature
            top_k: Number of top predictions to return
            
        Returns:
            Predicted token indices [batch_size]
        """
        # Get logits from forward pass
        logits = self.forward(tokens)  # [batch_size, seq_len, vocab_size]
        
        # Get the last token's logits
        last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        # Apply temperature
        scaled_logits = last_token_logits / temperature
        
        # Get top-k predictions
        top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)
        
        # Convert to probabilities
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # Sample from the distribution
        sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
        
        # Get the actual token indices
        predicted_tokens = torch.gather(top_k_indices, -1, sampled_indices)
        
        return predicted_tokens.squeeze(-1)  # [batch_size]


def create_graph_from_text(text, tokenizer, model):
    """
    Create a graph representation of text using the optimized quantum flux model.
    
    Args:
        text: Input text string
        tokenizer: Tokenizer for encoding text
        model: Trained VectorizedQuantumFluxGNN model
        
    Returns:
        Graph data object and visualization data
    """
    # Tokenize the text
    tokens = tokenizer.encode(text, return_tensors="pt")
    
    # Get embeddings and attention weights
    with torch.no_grad():
        embeddings = model.embedding(tokens)
        attention_weights = model.attention(embeddings)
    
    # Create graph data
    graph = model._prepare_sparse_graph_fully_vectorized(embeddings, attention_weights)
    
    # Extract the first graph's data for visualization
    batch_size, seq_len, _ = embeddings.shape
    
    # Get node indices for the first graph
    node_mask = graph.batch == 0
    first_graph_nodes = torch.arange(graph.num_nodes)[node_mask]
    
    # Get edge indices for the first graph
    edge_mask = torch.isin(graph.edge_index[0], first_graph_nodes)
    first_graph_edges = graph.edge_index[:, edge_mask]
    first_graph_weights = graph.edge_weight[edge_mask]
    
    # Adjust node indices to be relative to the first graph
    first_graph_edges = first_graph_edges - first_graph_nodes[0]
    
    # Prepare visualization data
    vis_data = {
        "tokens": [tokenizer.decode([t.item()]) for t in tokens[0]],
        "embeddings": embeddings[0].cpu().numpy(),
        "attention": attention_weights[0].cpu().numpy(),
        "edge_index": first_graph_edges.cpu().numpy(),
        "edge_weight": first_graph_weights.cpu().numpy()
    }
    
    return graph, vis_data


def main():
    """
    Example usage of the Vectorized Quantum Flux GNN model.
    """
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model with reduced dimensions for efficiency
    vocab_size = tokenizer.vocab_size
    model = VectorizedQuantumFluxGNN(
        vocab_size,
        embedding_dim=32,  # Reduced from 64
        hidden_dim=64,     # Reduced from 128
        num_layers=2,      # Reduced from 3
        dropout=0.1,
        sparsity_threshold=0.01  # Keep only edges with attention > 0.01
    )
    
    # Example text
    texts = [
        "I like quantum mechanics with pizza",
        "quantum mechanics is fascinating and complex",
        "the quick brown fox jumps over the lazy dog"
    ]
    
    # Process each text
    for text in texts:
        print(f"\nProcessing text: {text}")
        
        # Tokenize
        tokens = tokenizer.encode(text, return_tensors="pt")
        print(f"Tokenized: {[tokenizer.decode([t.item()]) for t in tokens[0]]}")
        
        # Create graph
        graph, vis_data = create_graph_from_text(text, tokenizer, model)
        
        # Print graph statistics
        print(f"Graph has {len(vis_data['tokens'])} nodes and {vis_data['edge_index'].shape[1]} edges")
        print(f"Sparsity: {vis_data['edge_index'].shape[1] / (len(vis_data['tokens']) * (len(vis_data['tokens']) - 1)):.2%}")
        
        # Print most important edges (highest attention weights)
        edge_weights = vis_data["edge_weight"]
        edge_indices = vis_data["edge_index"]
        token_names = vis_data["tokens"]
        
        # Get top 5 edges by weight
        if len(edge_weights) > 0:
            top_indices = np.argsort(edge_weights)[-min(5, len(edge_weights)):]
            print("\nTop token relationships (highest attention weights):")
            for idx in top_indices:
                src = edge_indices[0, idx]
                dst = edge_indices[1, idx]
                weight = edge_weights[idx]
                print(f"  {token_names[src]} -> {token_names[dst]}: {weight:.4f}")
        
        # Forward pass
        with torch.no_grad():
            logits = model(tokens)
        
        # Predict next token
        next_token_id = model.predict_next_token(tokens, temperature=0.7)[0].item()
        next_token = tokenizer.decode([next_token_id])
        print(f"\nPredicted next token: '{next_token}'")
        
        print("-" * 70)


if __name__ == "__main__":
    main()

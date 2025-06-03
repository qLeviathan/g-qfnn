#!/usr/bin/env python3
"""
enhanced_multi_sector_model.py - QFNN Epiphany Enhanced Multi-Sector Forecasting Model

This module implements an enhanced version of the multi-sector forecasting model
using the QFNN Epiphany LM architecture, incorporating:

1. Phase space embedding with Fibonacci modular spiral patterns
2. Quantum diffusion with RK2 integration
3. Hebbian learning with zero-gradient updates
4. Physics-informed loss functions
5. Optimized tensor operations for performance
6. Quantum metrics for model monitoring
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
import argparse
import math
import datetime
from tqdm import tqdm
import yfinance as yf
import requests
from pandas_datareader import data as pdr
import warnings
from scipy import stats
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

# Import base sector model components
from sector_model import (
    load_sector_indices,
    load_economic_indicators,
    prepare_multi_sector_dataset,
    validate_model_assumptions,
    validate_feature_importance,
    plot_feature_importance_comparison,
    plot_cross_sector_correlations,
    plot_multi_sector_predictions
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants
PHI = (1 + 5 ** 0.5) / 2  # Golden ratio (φ ≈ 1.618...)
EPSILON = 1e-8

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################################################
# Utility Decorators and Functions                  #
#####################################################

def zero_grad_decorator(fn):
    """Decorator to ensure a function runs with torch.no_grad()"""
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fn(*args, **kwargs)
    return wrapper

def compute_phase_coherence(psi):
    """
    Calculate phase coherence of quantum states
    
    Args:
        psi: Quantum state tensor [batch_size, seq_len, 2]
        
    Returns:
        Phase coherence value [0-1]
    """
    theta = torch.atan2(psi[..., 1], psi[..., 0])
    coherence = torch.abs(torch.mean(torch.exp(1j * theta), dim=1))
    return coherence.mean()

def compute_energy(psi):
    """Calculate the average energy (norm) of quantum states"""
    return torch.norm(psi, dim=-1).mean()

def compute_sparsity(mask):
    """Calculate the sparsity of a binary mask"""
    return mask.mean()

def wave_collapse_metric(psi):
    """
    Calculate a metric for wave function collapse
    based on differences between adjacent states
    """
    delta = torch.norm(psi[:, 1:] - psi[:, :-1], dim=-1)
    return torch.log(1 + delta).mean()

def compute_drift(psi):
    """
    Calculate drift between first and last quantum states
    in the sequence
    """
    return torch.norm(psi[:, -1] - psi[:, 0], dim=-1).mean()

def hamiltonian_eta(psi):
    """
    Calculate dynamic learning rate based on quantum state properties
    
    Args:
        psi: Quantum state tensor
        
    Returns:
        Dynamic learning rate value
    """
    E = compute_energy(psi)
    C = compute_phase_coherence(psi)
    drift = compute_drift(psi)
    eta = (C * E) / (drift + EPSILON)
    return eta.clamp(0.0001, 1.0)

def training_frequency(step, max_steps, mode='fib'):
    """
    Calculate training frequency based on different oscillation patterns
    
    Args:
        step: Current training step
        max_steps: Maximum number of steps
        mode: Oscillation mode ('phi', 'xi', or 'fib')
        
    Returns:
        Frequency value between 0 and 1
    """
    if mode == 'phi':
        return ((PHI * step) % 1.0)
    elif mode == 'xi':
        return ((PHI * step**0.5) % 1.0)
    elif mode == 'fib':
        a, b = 0, 1
        for _ in range(step % 24):
            a, b = b, a + b
        return (a % 13) / 13.0
    else:
        return 1.0

def decoherence_shock(psi, beta=0.03, threshold=0.5):
    """
    Apply random perturbation to quantum states with high entropy
    
    Args:
        psi: Quantum state tensor
        beta: Shock strength
        threshold: Entropy threshold for applying shock
        
    Returns:
        Perturbed quantum state tensor
    """
    entropy = -torch.sum(psi * torch.log(torch.clamp(torch.abs(psi), min=EPSILON)), dim=-1)
    mask = (entropy > threshold).unsqueeze(-1).float()
    shock = beta * torch.randn_like(psi)
    return psi + mask * shock

#####################################################
# Enhanced Core Components                          #
#####################################################

class PhaseHebbianEmbedder(nn.Module):
    """
    Embeds financial features in a Fibonacci modular spiral pattern in phase space.
    This creates a unique 2D representation for each feature based on the golden ratio.
    """
    def __init__(self, feature_dim):
        super().__init__()
        # Create Fibonacci modular spiral embedding
        indices = torch.arange(feature_dim).float()
        r = (PHI * indices) % 1.0  # Fibonacci modular mapping
        theta = 2 * torch.pi * r    # Convert to angle
        x = r * torch.cos(theta)    # x-coordinate
        y = r * torch.sin(theta)    # y-coordinate
        emb = torch.stack([x, y], dim=1)
        self.embedding = nn.Parameter(emb, requires_grad=False)
        
        # Store feature dimension
        self.feature_dim = feature_dim
        
        # Add a learnable projection layer for feature transformation
        self.projection = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, x):
        """
        Maps financial features to their phase space embeddings
        
        Args:
            x: Tensor of financial features [batch_size, seq_len, feature_dim]
            
        Returns:
            Tensor of embeddings [batch_size, seq_len, 2]
        """
        # Apply learnable projection to input features
        x_proj = self.projection(x)
        
        # Project features onto phase space using the embedding
        # This creates a weighted combination of the phase space points
        batch_size, seq_len, _ = x.shape
        
        # Reshape for batch matrix multiplication
        x_flat = x_proj.reshape(-1, self.feature_dim)  # [batch_size * seq_len, feature_dim]
        
        # Matrix multiply with embedding to get phase space coordinates
        phase_coords = torch.matmul(x_flat, self.embedding)  # [batch_size * seq_len, 2]
        
        # Reshape back to original dimensions
        phase_coords = phase_coords.reshape(batch_size, seq_len, 2)
        
        # Normalize to ensure points lie on unit circle
        norm = torch.norm(phase_coords, dim=-1, keepdim=True)
        phase_coords = phase_coords / (norm + EPSILON)
        
        return phase_coords

class HebbianLearner(nn.Module):
    """
    Neural network layer that uses Hebbian learning instead of backpropagation.
    Implements "neurons that fire together, wire together" with cosine modulation.
    """
    def __init__(self, input_dim, output_dim, eta=0.309, gamma=0.618):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.xavier_uniform_(self.weight)
        self.eta = eta        # Learning rate for Hebbian updates
        self.gamma = gamma    # Decay factor for weights

    def forward(self, x):
        """Standard forward pass using matrix multiplication"""
        return torch.matmul(x, self.weight.t())

    @zero_grad_decorator
    def hebbian_update(self, x, y_target):
        """
        Update weights using Hebbian learning rule with cosine modulation
        
        Args:
            x: Input tensor [batch_size, input_dim]
            y_target: Target output tensor [batch_size, output_dim]
        """
        # Calculate distances between inputs and weights
        diff = torch.cdist(x, self.weight, p=2)
        
        # Apply cosine modulation based on distance
        cos_kernel = 0.5 + 0.5 * torch.cos(diff / self.weight.size(-1))
        
        # Calculate Hebbian update term (correlation between input and output)
        hebbian_term = torch.bmm(y_target.unsqueeze(2), x.unsqueeze(1))
        
        # Apply cosine modulation to Hebbian term
        weighted_hebb = hebbian_term.mean(dim=0) * cos_kernel
        
        # Calculate weight update with decay term
        delta_W = self.eta * weighted_hebb - self.gamma * self.weight.data
        
        # Apply update to weights
        self.weight.add_(delta_W)

class RadialDiffusionIntegrator(nn.Module):
    """
    Implements quantum diffusion using a second-order Runge-Kutta (RK2) method
    while preserving the norm of the quantum state vectors.
    """
    def __init__(self, dt=0.1, steps=3):
        super().__init__()
        self.dt = dt        # Time step size
        self.steps = steps  # Number of integration steps

    def forward(self, psi, binary_mask):
        """
        Perform quantum diffusion integration
        
        Args:
            psi: Quantum state tensor [batch_size, seq_len, 2]
            binary_mask: Binary mask for diffusion [batch_size, seq_len, seq_len]
            
        Returns:
            Updated quantum state tensor [batch_size, seq_len, 2]
        """
        # Pre-allocate memory for intermediate results
        k1 = torch.zeros_like(psi)
        k2 = torch.zeros_like(psi)
        psi_star = torch.zeros_like(psi)
        
        # Define force function with operations that support autograd
        def force(state):
            # Use einsum for the matrix multiplication
            result = torch.einsum('bij,bjd->bid', binary_mask, state)
            # Subtract state using standard operation that supports autograd
            return result - state
        
        for _ in range(self.steps):
            # Store original radius (norm) of each state vector
            r = torch.norm(psi, dim=-1, keepdim=True)
            
            # First RK2 step
            k1 = force(psi)
            
            # Compute psi_star
            psi_star = k1 * self.dt + psi
            
            # Normalize intermediate state to preserve radius
            psi_star_norm = torch.norm(psi_star, dim=-1, keepdim=True)
            psi_star.mul_(r / (psi_star_norm + EPSILON))
            
            # Second RK2 step
            k2 = force(psi_star)
            
            # Update psi
            psi = psi + 0.5 * self.dt * (k1 + k2)
            
            # Normalize final state to preserve radius
            psi_norm = torch.norm(psi, dim=-1, keepdim=True)
            psi.mul_(r / (psi_norm + EPSILON))
            
        return psi

class HarmonyLoss(nn.Module):
    """
    Combined loss function that balances MSE with quantum state properties
    """
    def __init__(self, λ_mse=1.0, λ_coherence=0.3, λ_drift=0.2, λ_entropy=0.5):
        super().__init__()
        self.λ_mse = λ_mse                # Weight for MSE loss
        self.λ_coherence = λ_coherence    # Weight for coherence loss
        self.λ_drift = λ_drift            # Weight for drift loss
        self.λ_entropy = λ_entropy        # Weight for entropy loss

    def forward(self, predictions, targets, psi):
        """
        Calculate combined loss
        
        Args:
            predictions: Model output predictions
            targets: Target values
            psi: Quantum state tensor
            
        Returns:
            Combined loss value
        """
        # MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Phase coherence loss
        theta = torch.atan2(psi[..., 1], psi[..., 0])
        coherence = torch.abs(torch.mean(torch.exp(1j * theta), dim=1)).mean()
        coherence_loss = 1 - coherence
        
        # Drift loss
        drift = torch.norm(psi[:, -1] - psi[:, 0], dim=-1).mean()
        
        # Entropy loss
        entropy = -torch.sum(psi * torch.log(torch.clamp(torch.abs(psi), min=EPSILON)), dim=-1).mean()
        
        # Combine losses with weights
        total = (self.λ_mse * mse_loss + 
                 self.λ_coherence * coherence_loss + 
                 self.λ_drift * drift + 
                 self.λ_entropy * entropy)
        
        return total

#####################################################
# Enhanced Financial Quantum Model                  #
#####################################################

class EnhancedFinancialQuantumModel(nn.Module):
    """
    Enhanced Financial Quantum Model with QFNN Epiphany architecture
    
    This model combines phase embedding, quantum diffusion, and Hebbian learning
    for financial time series forecasting with multi-sector awareness.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=3, 
                 hebbian_lr=0.01, decay_rate=0.999, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Phase space embedding
        self.embedder = PhaseHebbianEmbedder(input_dim)
        
        # Quantum diffusion stages
        self.diffusion1 = RadialDiffusionIntegrator(dt=0.1, steps=3)
        self.diffusion2 = RadialDiffusionIntegrator(dt=0.05, steps=4)
        
        # Hebbian learning layer
        self.hebbian = HebbianLearner(
            input_dim=2,  # Phase space dimension
            output_dim=hidden_dim,
            eta=hebbian_lr,
            gamma=decay_rate
        )
        
        # Quantum attention layers
        self.quantum_layers = nn.ModuleList([
            QuantumAttentionBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final projection layers
        self.projection1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.activation = nn.SiLU()  # Sigmoid Linear Unit (SiLU/Swish)
        self.dropout = nn.Dropout(dropout)
        self.projection2 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        
        # Track coherence history
        self.coherence_history = []
        
    def encode_phase(self, x):
        """
        Convert input features to phase space representation
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Phase space tensor [batch_size, seq_len, 2]
        """
        return self.embedder(x)
    
    def forward(self, x, target=None, update=False):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            target: Optional target tensor for Hebbian updates
            update: Whether to perform Hebbian updates
            
        Returns:
            predictions: Output predictions
            A: Attention matrix
            binary_mask: Binary mask used for diffusion
            psi: Final quantum state tensor
        """
        batch_size, seq_len, _ = x.shape
        
        # Convert to phase space representation
        psi = self.encode_phase(x)
        
        # Create binary topology mask for diffusion
        binary_mask = torch.eye(seq_len, device=x.device)
        binary_mask = binary_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Apply first diffusion stage
        psi = self.diffusion1(psi, binary_mask)
        
        # Apply second diffusion stage
        psi = self.diffusion2(psi, binary_mask)
        
        # Compute attention matrix (for visualization and analysis)
        A = torch.bmm(psi.view(batch_size, seq_len, -1), 
                      psi.view(batch_size, seq_len, -1).transpose(1, 2))
        
        # Get final state from last sequence position
        final_psi = psi[:, -1]
        
        # Apply Hebbian layer
        hidden = self.hebbian(final_psi)
        
        # Reshape for quantum attention layers
        hidden = hidden.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Apply quantum attention layers
        for layer in self.quantum_layers:
            hidden = layer(hidden)
        
        # Final projection with residual connection
        hidden = hidden.squeeze(1)  # [batch_size, hidden_dim]
        hidden = self.norm1(hidden)
        
        # First projection
        proj = self.projection1(hidden)
        proj = self.activation(proj)
        proj = self.dropout(proj)
        proj = self.norm2(proj)
        
        # Final projection to output dimension
        predictions = self.projection2(proj)
        
        # Update weights via Hebbian learning if requested
        if update and target is not None:
            self.hebbian.hebbian_update(final_psi, target)
        
        # Track coherence
        coherence = compute_phase_coherence(psi).item()
        self.coherence_history.append(coherence)
        
        return predictions, A, binary_mask, psi

class QuantumAttentionBlock(nn.Module):
    """
    Quantum attention block for financial time series
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass through the attention block
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

#####################################################
# Enhanced Training Functions                       #
#####################################################

def train_enhanced_model(model, train_loader, val_loader, feature_names=None, 
                        num_epochs=100, lr=0.001, patience=20, use_harmony_loss=True):
    """
    Train the enhanced financial quantum model with physics-informed learning
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        feature_names: Names of input features for tracking
        num_epochs: Maximum number of epochs to train
        lr: Learning rate
        patience: Early stopping patience
        use_harmony_loss: Whether to use HarmonyLoss
        
    Returns:
        Trained model, training losses, validation losses, feature weights history
    """
    print(f"Training enhanced financial quantum model on {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Initialize loss function
    if use_harmony_loss:
        criterion = HarmonyLoss()
        print("Using HarmonyLoss for physics-informed training")
    else:
        criterion = nn.MSELoss()
        print("Using MSE loss for training")
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
    )
    
    # Initialize early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    feature_weights_history = []
    epochs_list = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        # Use tqdm for progress tracking
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for X_batch, y_batch in train_iterator:
            # Move tensors to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            predictions, A, binary_mask, psi = model(X_batch, target=y_batch, update=True)
            
            # Compute loss
            if use_harmony_loss:
                loss = criterion(predictions, y_batch, psi)
            else:
                loss = criterion(predictions, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            train_iterator.set_postfix({
                'loss': f"{loss.item():.6f}",
                'coherence': f"{model.coherence_history[-1]:.4f}" if model.coherence_history else "N/A"
            })
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        # Use tqdm for progress tracking
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for X_batch, y_batch in val_iterator:
                # Move tensors to device
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                predictions, _, _, psi = model(X_batch)
                
                # Compute loss
                if use_harmony_loss:
                    loss = criterion(predictions, y_batch, psi)
                else:
                    loss = criterion(predictions, y_batch)
                
                # Track metrics
                val_loss += loss.item()
                val_batch_count += 1
                
                # Update progress bar
                val_iterator.set_postfix({
                    'loss': f"{loss.item():.6f}"
                })
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Track feature weights if feature names are provided
        if feature_names is not None and hasattr(model, 'embedder') and hasattr(model.embedder, 'projection'):
            # Get the weights from the projection layer
            weights = model.embedder.projection.weight.detach().cpu().numpy()
            
            # Create a dictionary mapping feature names to their weights
            feature_weights = {}
            for i, name in enumerate(feature_names):
                if i < weights.shape[1]:  # Ensure we don't go out of bounds
                    feature_weights[name] = np.linalg.norm(weights[:, i])
            
            feature_weights_history.append(feature_weights)
            epochs_list.append(epoch)
        
        # Print epoch summary
        coherence = "N/A"
        if model.coherence_history:
            coherence = f"{sum(model.coherence_history[-batch_count:]) / batch_count:.4f}"
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Coherence: {coherence}")
        
        # Check for improvement and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"New best model! Val Loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation loss: {best_val_loss:.6f}")
    
    return model, train_losses, val_losses, feature_weights_history, epochs_list

def evaluate_enhanced_model(model, test_loader, dates, feature_metadata=None):
    """
    Evaluate the enhanced model on test data
    
    Args:
        model: Trained model
        test_loader: Test data loader
        dates: Dates corresponding to test data
        feature_metadata: Optional metadata about features
        
    Returns:
        Dictionary with evaluation results
    """
    print("Evaluating enhanced financial quantum model...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_coherence = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            # Move tensors to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            predictions, _, _, psi = model(X_batch)
            
            # Track predictions and targets
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
            # Track coherence
            coherence = compute_phase_coherence(psi).item()
            all_coherence.append(coherence)
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions).flatten()
    targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    # Calculate directional accuracy
    correct_direction = np.sum((predictions > 0) == (targets > 0))
    direction_accuracy = correct_direction / len(predictions)
    
    # Print results
    print(f"Test Results:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"Directional Accuracy: {direction_accuracy:.4f}")
    print(f"Average Coherence: {np.mean(all_coherence):.4f}")
    
    # Return results
    return {
        'predictions': predictions,
        'targets': targets,
        'dates': dates,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_accuracy': direction_accuracy,
        'coherence': np.mean(all_coherence)
    }

#####################################################
# Visualization Functions                           #
#####################################################

def plot_enhanced_predictions(results, title=None):
    """
    Plot prediction results with enhanced visualization
    
    Args:
        results: Dictionary with evaluation results
        title: Optional plot title
        
    Returns:
        Matplotlib figure
    """
    predictions = results['predictions']
    targets = results['targets']
    dates = results['dates']
    accuracy = results['direction_accuracy']
    
    if title is None:
        title = f"Enhanced Quantum Financial Forecast (Directional Accuracy: {accuracy:.1%})"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot predictions and targets
    ax.plot(dates, targets, 'b-', label='Actual Returns', linewidth=2, alpha=0.7)
    ax.plot(dates, predictions, 'r-', label='Predicted Returns', linewidth=2, alpha=0.7)
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Fill green when prediction and actual have same sign (correct direction)
    correct_mask = (predictions > 0) == (targets > 0)
    for i in range(len(dates)-1):
        if correct_mask[i]:
            ax.fill_between([dates[i], dates[i+1]], 
                          [min(0, min(predictions[i], targets[i])), min(0, min(predictions[i+1], targets[i+1]))],
                          [max(0, max(predictions[i], targets[i])), max(0, max(predictions[i+1], targets[i+1]))], 
                          color='green', alpha=0.3)
    
    # Fill red when prediction and actual have different signs (incorrect direction)
    for i in range(len(dates)-1):
        if not correct_mask[i]:
            ax.fill_between([dates[i], dates[i+1]], 
                          [min(0, min(predictions[i], targets[i])), min(0, min(predictions[i+1], targets[i+1]))],
                          [max(0, max(predictions[i], targets[i])), max(0, max(predictions[i+1], targets[i+1]))], 
                          color='red', alpha=0.3)
    
    # Customize plot
    ax.set_title(title, size=16)
    ax.set_xlabel("Date", size=12)
    ax.set_ylabel("Standardized Return", size=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Format date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    
    # Add metrics as text
    metrics_text = (
        f"MSE: {results['mse']:.6f}\n"
        f"RMSE: {results['rmse']:.6f}\n"
        f"MAE: {results['mae']:.6f}\n"
        f"R²: {results['r2']:.6f}\n"
        f"Dir. Acc: {results['direction_accuracy']:.2f}\n"
        f"Coherence: {results['coherence']:.4f}"
    )
    
    # Add text box with metrics
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig("enhanced_quantum_financial_predictions.png", dpi=300)
    
    return fig

def plot_phase_space_visualization(model, test_loader, feature_metadata=None, n_samples=5):
    """
    Visualize the phase space representation of financial data
    
    Args:
        model: Trained model
        test_loader: Test data loader
        feature_metadata: Optional metadata about features
        n_samples: Number of samples to visualize
        
    Returns:
        Matplotlib figure
    """
    print("Generating phase space visualization...")
    
    model.eval()
    
    # Get a batch of data
    X_batch, _ = next(iter(test_loader))
    X_batch = X_batch.to(device)
    
    # Limit to n_samples
    X_batch = X_batch[:n_samples]
    
    # Get phase space representation
    with torch.no_grad():
        psi = model.encode_phase(X_batch)
    
    # Convert to numpy
    psi_np = psi.cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(15, 5 * n_samples))
    
    # Create a custom colormap for time progression
    cmap = plt.cm.viridis
    
    for i in range(n_samples):
        # Extract sample
        sample_psi = psi_np[i]
        
        # Create 3D subplot
        ax = fig.add_subplot(n_samples, 1, i+1, projection='3d')
        
        # Extract coordinates
        x = sample_psi[:, 0]
        y = sample_psi[:, 1]
        z = np.arange(len(sample_psi))
        
        # Create colormap based on time
        colors = cmap(z / len(z))
        
        # Plot 3D trajectory
        ax.scatter(x, y, z, c=colors, s=30, alpha=0.8)
        
        # Connect points with lines
        ax.plot(x, y, z, 'k-', alpha=0.3)
        
        # Add a unit circle at each time step
        for t in range(len(z)):
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            ax.plot(circle_x, circle_y, t, 'k-', alpha=0.1)
        
        # Customize plot
        ax.set_title(f"Sample {i+1} Phase Space Trajectory", size=14)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Time Step")
        
        # Set equal aspect ratio for x and y
        ax.set_box_aspect([1, 1, 2])
    
    plt.tight_layout()
    plt.savefig("enhanced_quantum_financial_phase_space.png", dpi=300)
    
    return fig

def plot_feature_evolution(feature_weights_history, feature_names, epochs_list):
    """
    Plot the evolution of feature weights during training
    
    Args:
        feature_weights_history: List of dictionaries with feature weights
        feature_names: List of feature names
        epochs_list: List of epochs
        
    Returns:
        Matplotlib figure
    """
    print("Generating feature weight evolution plot...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get top features based on final weights
    final_weights = feature_weights_history[-1]
    top_features = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)[:10]
    top_feature_names = [f[0] for f in top_features]
    
    # Plot weight evolution for top features
    for feature in top_feature_names:
        weights = [fw[feature] for fw in feature_weights_history]
        ax.plot(epochs_list, weights, '-o', label=feature, alpha=0.7)
    
    # Customize plot
    ax.set_title("Feature Weight Evolution During Training", size=16)
    ax.set_xlabel("Epoch", size=12)
    ax.set_ylabel("Feature Weight", size=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig("enhanced_quantum_financial_feature_evolution.png", dpi=300)
    
    return fig

def plot_coherence_history(model, title="Phase Coherence During Training"):
    """
    Plot the history of phase coherence during training
    
    Args:
        model: Trained model with coherence_history
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    print("Generating coherence history plot...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get coherence history
    coherence = model.coherence_history
    
    # Plot coherence
    ax.plot(coherence, 'b-', alpha=0.7)
    
    # Apply smoothing for trend line
    window_size = min(50, len(coherence) // 10)
    if window_size > 0:
        smoothed = np.convolve(coherence, np.ones(window_size) / window_size, mode='valid')
        ax.plot(np.arange(window_size-1, len(coherence)), smoothed, 'r-', linewidth=2, label='Trend')
    
    # Customize plot
    ax.set_title(title, size=16)
    ax.set_xlabel("Training Step", size=12)
    ax.set_ylabel("Phase Coherence", size=12)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal lines for reference
    ax.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='0.5 Threshold')
    ax.axhline(y=0.8, color='g', linestyle=':', alpha=0.5, label='0.8 Threshold')
    
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("enhanced_quantum_financial_coherence.png", dpi=300)
    
    return fig

#####################################################
# Main Function                                     #
#####################################################

def main(args):
    """Main function to run the enhanced multi-sector financial analysis"""
    print("\n" + "="*60)
    print("QFNN Epiphany Enhanced Multi-Sector Financial Analysis")
    print("="*60 + "\n")
    
    # Step 1: Load Sector Data
    sector_data = load_sector_indices(start_date=args.start_date, end_date=args.end_date)
    
    # Step 2: Load Economic Indicators
    economic_data = load_economic_indicators(start_date=args.start_date, end_date=args.end_date)
    
    # Step 3: Visualize cross-sector correlations
    print("\nVisualizing cross-sector correlations...")
    plot_cross_sector_correlations(sector_data)
    
    # Step 4: Prepare and run models for each sector
    sector_results = {}
    feature_metadata_dict = {}
    
    for target_sector in args.target_sectors:
        if target_sector not in sector_data:
            print(f"Warning: {target_sector} not found in sector data. Skipping.")
            continue
            
        print(f"\n{'-'*60}")
        print(f"Processing target sector: {target_sector}")
        print(f"{'-'*60}")
        
        # Prepare dataset for this sector
        train_dataset, test_dataset, scaler, (dates_train, dates_test), feature_metadata = prepare_multi_sector_dataset(
            sector_data, economic_data, target_sector=target_sector,
            sequence_length=args.seq_length, forecast_horizon=args.forecast_horizon, 
            test_split=args.test_split
        )
        
        feature_metadata_dict[target_sector] = feature_metadata
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Use a portion of training data for validation
        val_size = int(len(train_dataset) * 0.2)
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
        
        # Define and train model
        # Properly extract the number of features by checking the shape
        sample_batch = next(iter(train_loader))[0]
        input_dim = sample_batch.shape[2]  # Number of features
        
        model = EnhancedFinancialQuantumModel(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            num_layers=args.num_layers,
            hebbian_lr=args.hebbian_lr,
            decay_rate=args.decay_rate,
            dropout=args.dropout
        ).to(device)
        
        print(f"\nTraining enhanced model for {target_sector}:")
        print(f"- Input Dimension: {input_dim}")
        print(f"- Hidden Dimension: {args.hidden_dim}")
        print(f"- Number of Quantum Layers: {args.num_layers}")
        
        # Get feature names for weight tracking
        feature_names = feature_metadata['all_features']
        
        # Cross-validation if requested
        if args.run_cv:
            cv_results = run_cross_validation(
                EnhancedFinancialQuantumModel,
                train_dataset,
                n_folds=5,
                hidden_dim=args.hidden_dim,
                output_dim=1,
                num_layers=args.num_layers,
                hebbian_lr=args.hebbian_lr,
                decay_rate=args.decay_rate,
                dropout=args.dropout
            )
        
        # Train the model
        model, train_losses, val_losses, feature_weights_history, epochs_list = train_enhanced_model(
            model, train_loader, val_loader, feature_names,
            num_epochs=args.epochs, 
            lr=args.learning_rate,
            patience=args.patience,
            use_harmony_loss=args.use_harmony_loss
        )
        
        # Evaluate model
        print(f"\nEvaluating enhanced model for {target_sector}...")
        test_results = evaluate_enhanced_model(model, test_loader, dates_test, feature_metadata)
        
        # Validate model assumptions
        validation_results = validate_model_assumptions(
            test_results['predictions'], 
            test_results['targets'], 
            test_results['dates']
        )
        
        # Validate feature importance
        importance_results = validate_feature_importance(
            model, test_dataset, feature_metadata, n_samples=10
        )
        
        # Plot feature importance comparison
        plot_feature_importance_comparison(
            importance_results, feature_metadata['feature_groups']
        )
        
        # Store results for this sector
        sector_results[target_sector] = test_results
        
        # Generate enhanced visualizations
        print(f"\nGenerating enhanced visualizations for {target_sector}...")
        
        # Plot training and validation losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'{target_sector} Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"enhanced_quantum_financial_{target_sector}_training_loss.png", dpi=300)
        plt.close()
        
        # Plot enhanced predictions
        plot_enhanced_predictions(test_results, title=f"{target_sector} Enhanced Quantum Forecast")
        
        # Plot phase space visualization
        plot_phase_space_visualization(model, test_loader, feature_metadata)
        
        # Plot feature evolution
        plot_feature_evolution(feature_weights_history, feature_names, epochs_list)
        
        # Plot coherence history
        plot_coherence_history(model, title=f"{target_sector} Phase Coherence During Training")
    
    # Plot multi-sector predictions
    if len(sector_results) > 1:
        plot_multi_sector_predictions(sector_results, title="Enhanced Multi-Sector Quantum Model Predictions")
    
    print("\nEnhanced multi-sector financial analysis completed successfully!")
    print(f"All visualizations have been saved to the current directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QFNN Epiphany Enhanced Multi-Sector Financial Analysis")
    
    # Data parameters
    parser.add_argument('--start-date', type=str, default='2000-01-01', help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date for data (YYYY-MM-DD)')
    parser.add_argument('--seq-length', type=int, default=12, help='Sequence length (months)')
    parser.add_argument('--forecast-horizon', type=int, default=1, help='Forecast horizon (months ahead)')
    parser.add_argument('--test-split', type=float, default=0.2, help='Test set proportion')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of quantum attention layers')
    parser.add_argument('--hebbian-lr', type=float, default=0.01, help='Hebbian learning rate')
    parser.add_argument('--decay-rate', type=float, default=0.999, help='Hebbian weight decay rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--use-harmony-loss', action='store_true', help='Use HarmonyLoss')
    
    # Analysis parameters
    parser.add_argument('--target-sectors', nargs='+', 
                        default=['S&P500', 'Technology', 'Healthcare', 'Financials', 'Energy'],
                        help='Target sectors to analyze')
    parser.add_argument('--run-cv', action='store_true', help='Run cross-validation')
    
    args = parser.parse_args()
    
    main(args)

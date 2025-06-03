#!/usr/bin/env python3
"""
financial_quantum_forecaster.py - Xi/Psi Quantum Field Neural Network for Financial Time Series

This script demonstrates the Xi/Psi framework applied to financial forecasting:
- Uses S&P 500 monthly returns and economic indicators
- Applies phase-space encoding to financial data
- Implements quantum field neural network with causal structure
- Tests both in-sample and out-of-sample performance
- Visualizes quantum field properties over financial time series
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse
import math
import datetime
from tqdm import tqdm
import yfinance as yf
import requests
from pandas_datareader import data as pdr

# FRED API key for economic data
FRED_API_KEY = "a3d0da246d51ae579c0186a16dc29075"  # Set your FRED API key here if available

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################################################
# Xi/Psi Quantum Field Neural Network Architecture  #
#####################################################

class HebbianLinear(nn.Module):
    def __init__(self, in_features, out_features, hebbian_lr=0.01, decay_rate=0.999):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hebbian_lr = hebbian_lr
        self.decay_rate = decay_rate

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def hebbian_update(self, x, y):
        with torch.no_grad():
            x_flat = x.view(-1, self.in_features)
            y_flat = y.view(-1, self.out_features)

            update = torch.bmm(y_flat.unsqueeze(2), x_flat.unsqueeze(1)).mean(0)
            self.weight.mul_(self.decay_rate).add_(self.hebbian_lr * update)
            self.bias.add_(self.hebbian_lr * (y_flat.mean(0) - self.forward(x_flat).mean(0)))

class PhaseAwareBinaryAttention(nn.Module):
    def __init__(self, threshold_factor=1.25):
        super().__init__()
        self.threshold_factor = threshold_factor
    
    def forward(self, r_embed, causal=True):
        # Get dimensions
        batch_size, seq_len, _ = r_embed.shape
        
        # Calculate pairwise differences
        diff = r_embed.unsqueeze(2) - r_embed.unsqueeze(1)
        
        # Compute squared distances
        dist_sq = torch.sum(diff ** 2, dim=-1)
        
        # Calculate phase differences for directionality
        phase_diff = torch.atan2(diff[..., 1], diff[..., 0])
        
        # Forward bias based on phase alignment (strengthens causal connections)
        forward_bias = 0.5 * (1 + torch.cos(phase_diff))
        
        # Calculate adaptive threshold
        mean_dist = dist_sq.mean(dim=(-1, -2), keepdim=True)
        std_dist = dist_sq.std(dim=(-1, -2), keepdim=True)
        threshold = mean_dist + self.threshold_factor * std_dist
        
        # Apply causal masking with phase awareness
        if causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=r_embed.device))
            valid_mask = (dist_sq <= threshold) & causal_mask.unsqueeze(0).bool()
            attention = valid_mask.float() * forward_bias
        else:
            valid_mask = (dist_sq <= threshold)
            attention = valid_mask.float()
        
        return attention

class QuantumAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, hebbian_lr=0.01, decay_rate=0.999):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Hebbian projections for Q, K, V
        self.query_proj = HebbianLinear(embedding_dim, embedding_dim, hebbian_lr, decay_rate)
        self.key_proj = HebbianLinear(embedding_dim, embedding_dim, hebbian_lr, decay_rate)
        self.value_proj = HebbianLinear(embedding_dim, embedding_dim, hebbian_lr, decay_rate)
        self.output_proj = HebbianLinear(embedding_dim, embedding_dim, hebbian_lr, decay_rate)
        
        # Quantum projections
        self.to_quantum = nn.Linear(embedding_dim, 2)
        self.from_quantum = HebbianLinear(2, embedding_dim, hebbian_lr, decay_rate)
        
        # Phase-aware binary attention
        self.attn = PhaseAwareBinaryAttention()
    
    def forward(self, hidden_states):
        # Project to query, key, value spaces
        q = self.query_proj(hidden_states)
        k = self.key_proj(hidden_states)
        v = self.value_proj(hidden_states)
        
        # Efficient scaled dot-product attention with einsum
        attention_scores = torch.einsum('bid,bjd->bij', q, k) / math.sqrt(self.embedding_dim)
        
        # Project to quantum phase space for binary topological attention
        quantum_proj = self.to_quantum(hidden_states)
        
        # Get binary attention mask with phase awareness
        binary_attn = self.attn(quantum_proj)
        
        # Apply binary mask to attention scores
        masked_scores = attention_scores.masked_fill(binary_attn == 0, float('-inf'))
        weights = F.softmax(masked_scores, dim=-1)
        
        # Apply attention weights to values with einsum
        context = torch.einsum('bij,bjd->bid', weights, v)
        
        # Project to output space
        output = self.output_proj(context)
        
        # Normalize quantum projection for stable phase representation
        r_embed = F.normalize(quantum_proj, p=2, dim=-1)
        
        # Project back from quantum space
        quantum_output = self.from_quantum(r_embed)
        
        # Combine through residual connection
        return hidden_states + output + 0.1 * quantum_output

class FinancialQuantumModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, hebbian_lr=0.01, decay_rate=0.999):
        super().__init__()
        
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Quantum attention layers
        self.qfnn_layers = nn.ModuleList([
            QuantumAttentionBlock(hidden_dim, hebbian_lr, decay_rate) 
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
        
        # Phase coherence tracker
        self.coherence_history = []
        
        # Save dimensions for phase encoding
        self.hidden_dim = hidden_dim
        self.phi = (1 + 5**0.5) / 2  # Golden ratio
        
    def encode_phase(self, x):
        """Convert financial values to phase space representation"""
        # Map inputs to [0, 1] range
        normalized = (x - x.min(dim=1, keepdim=True)[0]) / (
            x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0] + 1e-8
        )
        
        # Convert to phase using golden ratio distribution
        phase = 2 * math.pi * normalized * self.phi
        
        # Convert to points on unit circle
        return torch.stack([
            torch.cos(phase),
            torch.sin(phase)
        ], dim=-1)
    
    def forward(self, x):
        # Encode financial data in phase space
        batch_size, seq_len, features = x.shape
        
        # Initial embedding
        h = self.embedding(x)
        
        # Track phase coherence
        encoded = self.encode_phase(x.reshape(-1, features)).reshape(batch_size, seq_len, features, 2)
        phase = torch.atan2(encoded[..., 1], encoded[..., 0])
        sin_mean = torch.mean(torch.sin(phase), dim=(1, 2))
        cos_mean = torch.mean(torch.cos(phase), dim=(1, 2))
        coherence = torch.sqrt(sin_mean**2 + cos_mean**2).mean().item()
        self.coherence_history.append(coherence)
        
        # Process through quantum attention layers
        for layer in self.qfnn_layers:
            h = layer(h)
        
        # Output projection
        return self.output(h)

#####################################################
# Data Loading and Processing Functions             #
#####################################################

def load_sp500_data(start_date='1990-01-01', end_date=None):
    """Load S&P 500 data from Yahoo Finance"""
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    print(f"Loading S&P 500 data from {start_date} to {end_date}...")
    
    # Use yfinance to get S&P 500 data
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, interval='1mo')
    
    # Debug: Print available columns
    print(f"Available columns in SP500 data: {sp500.columns.tolist()}")
    
    # Handle MultiIndex columns if present
    if isinstance(sp500.columns, pd.MultiIndex):
        # Flatten the MultiIndex to use just the first level (e.g., 'Close', 'High', etc.)
        sp500.columns = sp500.columns.get_level_values(0)
        print(f"Flattened columns: {sp500.columns.tolist()}")
    
    # Check if 'Adj Close' exists, otherwise use 'Close'
    price_col = 'Close'
    if 'Adj Close' in sp500.columns:
        price_col = 'Adj Close'
        
    # Calculate monthly returns
    sp500['Return'] = sp500[price_col].pct_change()
    
    # Add more financial indicators
    sp500['Volatility'] = sp500['Return'].rolling(window=12).std()
    sp500['Moving_Avg_12'] = sp500[price_col].rolling(window=12).mean() / sp500[price_col] - 1
    sp500['Price_to_SMA'] = sp500[price_col] / sp500['Moving_Avg_12'].shift(1)
    
    # Remove rows with NaN values
    sp500 = sp500.dropna()
    
    print(f"Loaded {len(sp500)} months of S&P 500 data")
    return sp500

def get_fred_data(series_ids, start_date='1990-01-01', end_date=None):
    """Load economic data from FRED"""
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    print(f"Loading economic data from FRED...")
    
    economic_data = {}
    
    try:
        # Try using pandas_datareader
        for series_id in series_ids:
            data = pdr.get_data_fred(series_id, start_date, end_date)
            economic_data[series_id] = data
        
        # Combine all series
        combined_data = pd.concat(economic_data.values(), axis=1)
        combined_data.columns = series_ids
        
    except Exception as e:
        print(f"Error loading FRED data: {e}")
        print("Using random synthetic data instead for demonstration")
        
        # Generate synthetic data for demonstration
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        synthetic_data = {}
        
        for series_id in series_ids:
            # Generate smooth random data with realistic trends
            values = np.cumsum(np.random.randn(len(date_range)) * 0.1)
            
            # Add seasonality for realism
            seasonality = 0.2 * np.sin(np.arange(len(date_range)) * 2 * np.pi / 12)
            values += seasonality
            
            synthetic_data[series_id] = pd.Series(values, index=date_range)
        
        combined_data = pd.DataFrame(synthetic_data)
    
    print(f"Loaded {len(combined_data)} months of economic data")
    return combined_data.dropna()

def prepare_financial_dataset(sp500_data, economic_data, sequence_length=12, forecast_horizon=1, test_split=0.2):
    """Prepare dataset for financial forecasting with time series structure"""
    
    # Merge S&P 500 and economic data on date index
    merged_data = sp500_data[['Return', 'Volatility']].join(economic_data)
    merged_data = merged_data.dropna()
    
    # Normalize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(merged_data.values)
    scaled_df = pd.DataFrame(scaled_data, index=merged_data.index, columns=merged_data.columns)
    
    # Create sequences for time-series prediction
    X, y = [], []
    dates = []
    
    for i in range(len(scaled_df) - sequence_length - forecast_horizon + 1):
        # Input sequence
        X.append(scaled_df.iloc[i:i+sequence_length].values)
        
        # Target: S&P 500 return forecast_horizon months ahead
        y.append(scaled_df.iloc[i+sequence_length+forecast_horizon-1]['Return'])
        
        # Keep track of the date for each sequence's end point
        dates.append(scaled_df.index[i+sequence_length-1])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets based on time
    split_idx = int(len(X) * (1 - test_split))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    print(f"Dataset prepared: {X_train_tensor.shape[0]} training sequences, {X_test_tensor.shape[0]} testing sequences")
    print(f"Each sequence has {X_train_tensor.shape[1]} time steps and {X_train_tensor.shape[2]} features")
    
    # Create dataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return train_dataset, test_dataset, scaler, (dates_train, dates_test), merged_data

#####################################################
# Training and Evaluation Functions                 #
#####################################################

def train_financial_quantum_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, patience=10):
    """Train the financial quantum model with phase-coherent learning"""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler to reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Track metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"Training on {device}...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass with causal prediction
            predictions = model(X_batch)[:, -1, :]  # Get prediction for last time step
            
            # Compute loss
            loss = criterion(predictions, y_batch)
            
            # Apply phase coherence regularization
            phase_coherence = sum(model.coherence_history[-10:]) / min(10, len(model.coherence_history))
            coherence_penalty = 0.01 * (1 - phase_coherence)  # Penalize low coherence
            regularized_loss = loss + coherence_penalty
            
            # Backward pass and optimization
            optimizer.zero_grad()
            regularized_loss.backward()
            
            # Apply Hebbian updates to qualifying layers
            # Pass the embedded features through the embedding layer first
            with torch.no_grad():
                # Get embedded features (which will have the right dimensions)
                embedded_features = model.embedding(X_batch)
                
                # Iterate through each quantum layer
                for layer in model.qfnn_layers:
                    if hasattr(layer.query_proj, 'hebbian_update'):
                        # For correct dimensionality, get activations from the already embedded features
                        batch_size, seq_len, hidden_dim = embedded_features.shape
                        flat_embedded = embedded_features.reshape(-1, hidden_dim)
                        
                        # Get intermediate activations
                        q = layer.query_proj(flat_embedded)
                        k = layer.key_proj(flat_embedded)
                        
                        # Apply Hebbian updates with properly shaped inputs
                        layer.query_proj.hebbian_update(flat_embedded, q)
                        layer.key_proj.hebbian_update(flat_embedded, k)
            
            # Standard optimizer step
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                predictions = model(X_batch)[:, -1, :]
                
                # Compute loss
                loss = criterion(predictions, y_batch)
                
                total_val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = total_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Coherence: {phase_coherence:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader, dates, orig_data):
    """Evaluate model on test set and calculate metrics"""
    model.eval()
    criterion = nn.MSELoss()
    
    # Lists to store predictions and targets
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            pred = model(X_batch)[:, -1, :]
            
            # Store results
            predictions.extend(pred.cpu().numpy())
            targets.extend(y_batch.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # Directional accuracy
    correct_direction = np.sum((predictions > 0) == (targets > 0))
    direction_accuracy = correct_direction / len(predictions)
    
    print(f"Test MSE: {mse:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Directional Accuracy: {direction_accuracy:.4f}")
    
    return {
        'predictions': predictions,
        'targets': targets,
        'dates': dates,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'direction_accuracy': direction_accuracy
    }

#####################################################
# Visualization Functions                           #
#####################################################

def plot_phase_space_visualization(model, test_data, timestep=-1, sample_idx=0):
    """Visualize the phase space representation of financial data"""
    # Get a sample from test data
    X_sample = test_data[sample_idx][0].unsqueeze(0).to(device)
    
    # Get model's phase encoding
    with torch.no_grad():
        batch_size, seq_len, features = X_sample.shape
        encoded = model.encode_phase(X_sample.reshape(-1, features)).reshape(
            batch_size, seq_len, features, 2)
    
    # Extract phase angles
    phase = torch.atan2(encoded[..., 1], encoded[..., 0])
    
    # Convert to numpy for plotting
    phase_np = phase[0, timestep].cpu().numpy()
    x_coords = np.cos(phase_np)
    y_coords = np.sin(phase_np)
    
    # Get feature names for the sample
    feature_names = ["Feature " + str(i) for i in range(features)]
    
    # Create polar plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # Plot points on unit circle
    for i in range(len(phase_np)):
        ax.plot([0, phase_np[i]], [0, 1], marker='o', label=feature_names[i])
    
    # Add unity circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(theta, np.ones_like(theta), 'k--', alpha=0.3)
    
    # Set title and legend
    ax.set_title("Phase Space Representation of Financial Features", size=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    plt.savefig("quantum_financial_phase_space.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_coherence_history(model):
    """Plot the phase coherence history during training"""
    coherence = model.coherence_history
    
    plt.figure(figsize=(10, 6))
    plt.plot(coherence, color='blue')
    plt.title("Phase Coherence During Training", size=14)
    plt.xlabel("Training Step")
    plt.ylabel("Phase Coherence")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("quantum_financial_coherence.png", dpi=300)
    plt.show()
    
    return plt.gcf()

def plot_prediction_results(results, title="S&P 500 Return Prediction"):
    """Plot the prediction results against actual values"""
    predictions = results['predictions']
    targets = results['targets']
    dates = results['dates']
    
    plt.figure(figsize=(12, 6))
    
    # Plot predictions and targets
    plt.plot(dates, targets, 'b-', label='Actual Returns', alpha=0.7)
    plt.plot(dates, predictions, 'r-', label='Predicted Returns', alpha=0.7)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Fill green when prediction and actual have same sign (correct direction)
    correct_mask = (predictions > 0) == (targets > 0)
    for i in range(len(dates)-1):
        if correct_mask[i]:
            plt.fill_between([dates[i], dates[i+1]], 
                            [min(0, min(predictions[i], targets[i])), min(0, min(predictions[i+1], targets[i+1]))],
                            [max(0, max(predictions[i], targets[i])), max(0, max(predictions[i+1], targets[i+1]))], 
                            color='green', alpha=0.3)
    
    # Customize plot
    plt.title(title, size=16)
    plt.xlabel("Date")
    plt.ylabel("Standardized Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig("quantum_financial_predictions.png", dpi=300)
    plt.show()
    
    return plt.gcf()

def plot_quantum_attention_heatmap(model, test_loader, sample_idx=0):
    """Visualize the quantum attention weights for a sample"""
    model.eval()
    
    # Get a sample batch
    X_sample = None
    for i, (X, _) in enumerate(test_loader):
        if i == sample_idx:
            X_sample = X.to(device)
            break
    
    if X_sample is None:
        print("Sample index out of range. Using first batch instead.")
        X_sample, _ = next(iter(test_loader))
        X_sample = X_sample.to(device)
    
    # Forward pass to extract attention weights
    attention_maps = []
    
    def hook_fn(module, input, output):
        # Extract attention weights from the module
        if hasattr(module, 'attn'):
            with torch.no_grad():
                quantum_proj = module.to_quantum(input[0])
                attention = module.attn(quantum_proj)
                attention_maps.append(attention.detach().cpu().numpy())
    
    # Register hooks to capture attention
    hooks = []
    for layer in model.qfnn_layers:
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(X_sample)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Plot attention heatmaps
    if attention_maps:
        n_layers = len(attention_maps)
        fig, axes = plt.subplots(1, n_layers, figsize=(n_layers*5, 5))
        
        if n_layers == 1:
            axes = [axes]
        
        for i, attn_map in enumerate(attention_maps):
            # Choose first sample and first head
            att = attn_map[0]
            
            # Plot heatmap
            im = axes[i].imshow(att, cmap='viridis')
            axes[i].set_title(f"Layer {i+1} Attention")
            axes[i].set_xlabel("Sequence Position (Time)")
            axes[i].set_ylabel("Sequence Position (Time)")
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig("quantum_financial_attention.png", dpi=300)
        plt.show()
        
        return fig
    
    return None

def compute_cumulative_returns(predictions, targets, dates, initial_value=1.0):
    """Compute cumulative return performance for model vs. actual"""
    # Convert standardized returns back to percentage returns
    # These are already standardized, so we'll treat them as excess returns above mean
    
    # Start with the same initial value
    pred_cum = [initial_value]
    target_cum = [initial_value]
    
    # Compute cumulative returns
    for i in range(len(predictions)):
        pred_cum.append(pred_cum[-1] * (1 + predictions[i] * 0.01))  # Scale factor for realistic returns
        target_cum.append(target_cum[-1] * (1 + targets[i] * 0.01))
    
    # Prepend the start date
    all_dates = [dates[0] - datetime.timedelta(days=30)] + list(dates)
    
    return all_dates, pred_cum, target_cum

def plot_cumulative_returns(predictions, targets, dates):
    """Plot cumulative return performance"""
    all_dates, pred_cum, target_cum = compute_cumulative_returns(predictions, targets, dates)
    
    plt.figure(figsize=(12, 6))
    
    # Plot cumulative returns
    plt.plot(all_dates, target_cum, 'b-', label='Actual Returns', linewidth=2)
    plt.plot(all_dates, pred_cum, 'r-', label='Strategy Returns', linewidth=2)
    
    # Customize plot
    plt.title("Cumulative Return Performance", size=16)
    plt.xlabel("Date")
    plt.ylabel("Value of $1 Investment")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()
    
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig("quantum_financial_cumulative_returns.png", dpi=300)
    plt.show()
    
    return plt.gcf()

#####################################################
# Main Function                                     #
#####################################################

def main(args):
    """Main function to run the financial quantum experiment"""
    print("\n" + "="*50)
    print("Xi/Psi Quantum Financial Time Series Analysis")
    print("="*50 + "\n")
    
    # Step 1: Load Data
    sp500_data = load_sp500_data(start_date=args.start_date, end_date=args.end_date)
    
    # Define economic indicators to use
    fred_series = [
        'UNRATE',      # Unemployment Rate
        'CPIAUCSL',    # Consumer Price Index
        'FEDFUNDS',    # Federal Funds Rate
        'INDPRO',      # Industrial Production Index
        'M2SL',        # M2 Money Stock
        'GS10',        # 10-Year Treasury Rate
        'HOUST',       # Housing Starts
        'DCOILWTICO'   # Crude Oil Price
    ]
    
    economic_data = get_fred_data(fred_series, start_date=args.start_date, end_date=args.end_date)
    
    # Step 2: Prepare Dataset
    train_dataset, test_dataset, scaler, (dates_train, dates_test), merged_data = prepare_financial_dataset(
        sp500_data, economic_data, sequence_length=args.seq_length, forecast_horizon=args.forecast_horizon, 
        test_split=args.test_split
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Use a portion of training data for validation
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    
    # Step 3: Define and Train Model
    input_dim = train_dataset[0][0].shape[1]  # Number of features
    
    model = FinancialQuantumModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_layers=args.num_layers,
        hebbian_lr=args.hebbian_lr,
        decay_rate=args.decay_rate
    ).to(device)
    
    print(f"\nModel Architecture:")
    print(f"- Input Dimension: {input_dim}")
    print(f"- Hidden Dimension: {args.hidden_dim}")
    print(f"- Number of Quantum Layers: {args.num_layers}")
    print(f"- Hebbian Learning Rate: {args.hebbian_lr}")
    print(f"- Decay Rate: {args.decay_rate}")
    
    # Train the model
    model, train_losses, val_losses = train_financial_quantum_model(
        model, train_loader, val_loader, 
        num_epochs=args.epochs, 
        lr=args.learning_rate,
        patience=args.patience
    )
    
    # Step 4: Evaluate Model
    print("\nEvaluating model on test set...")
    test_results = evaluate_model(model, test_loader, dates_test, merged_data)
    
    # Step 5: Visualize Results
    print("\nGenerating visualizations...")
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("quantum_financial_training_loss.png", dpi=300)
    plt.show()
    
    # Plot phase space visualization
    plot_phase_space_visualization(model, test_dataset)
    
    # Plot coherence history
    plot_coherence_history(model)
    
    # Plot predictions vs actual
    plot_prediction_results(test_results)
    
    # Plot cumulative returns
    plot_cumulative_returns(test_results['predictions'], test_results['targets'], test_results['dates'])
    
    # Plot quantum attention heatmap
    plot_quantum_attention_heatmap(model, test_loader)
    
    print("\nExperiment completed successfully!")
    print(f"All visualizations have been saved to the current directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xi/Psi Quantum Financial Time Series Analysis")
    
    # Data parameters
    parser.add_argument('--start-date', type=str, default='2000-01-01', help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date for data (YYYY-MM-DD)')
    parser.add_argument('--seq-length', type=int, default=12, help='Sequence length (months)')
    parser.add_argument('--forecast-horizon', type=int, default=1, help='Forecast horizon (months ahead)')
    parser.add_argument('--test-split', type=float, default=0.2, help='Test set proportion')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of quantum attention layers')
    parser.add_argument('--hebbian-lr', type=float, default=0.01, help='Hebbian learning rate')
    parser.add_argument('--decay-rate', type=float, default=0.999, help='Hebbian weight decay rate')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    
    main(args)

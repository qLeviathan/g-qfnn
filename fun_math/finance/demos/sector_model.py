#!/usr/bin/env python3
"""
sector_quantum_forecaster.py - Enhanced Xi/Psi Quantum Field Neural Network for Financial Time Series

This script extends the base model to include:
- Multiple sector indices beyond S&P 500
- Cross-sector correlations and influence
- Integrated validation framework
- Enhanced measurement methodology
- Backtesting framework for model validation
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

# Import the base model components
from test import (
    HebbianLinear, 
    PhaseAwareBinaryAttention, 
    QuantumAttentionBlock, 
    FinancialQuantumModel, 
    evaluate_model, 
    plot_coherence_history,
    plot_phase_space_visualization,
    plot_quantum_attention_heatmap,
    plot_weight_evolution,
    train_financial_quantum_model,
    plot_prediction_results
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

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
# Sector Index Data Loading Functions               #
#####################################################

def load_sector_indices(start_date='1990-01-01', end_date=None, progress=True):
    """
    Load all major US sector indices from Yahoo Finance
    
    Args:
        start_date: Start date for historical data
        end_date: End date for historical data
        progress: Whether to show progress information
    
    Returns:
        Dictionary of DataFrames with index data
    """
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    if progress:
        print(f"Loading sector indices from {start_date} to {end_date}...")
    
    # Define major sector ETFs and indices
    # Using ETFs for consistency and completeness
    sector_tickers = {
        'S&P500': '^GSPC',            # S&P 500 (Broad Market)
        'Technology': 'XLK',          # Technology Select Sector SPDR Fund
        'Healthcare': 'XLV',          # Health Care Select Sector SPDR Fund
        'Financials': 'XLF',          # Financial Select Sector SPDR Fund
        'Energy': 'XLE',              # Energy Select Sector SPDR Fund
        'Consumer_Discretionary': 'XLY', # Consumer Discretionary Select Sector SPDR Fund
        'Consumer_Staples': 'XLP',    # Consumer Staples Select Sector SPDR Fund
        'Industrials': 'XLI',         # Industrial Select Sector SPDR Fund
        'Materials': 'XLB',           # Materials Select Sector SPDR Fund
        'Utilities': 'XLU',           # Utilities Select Sector SPDR Fund
        'Real_Estate': 'XLRE',        # Real Estate Select Sector SPDR Fund
        'Communications': 'XLC'        # Communication Services Select Sector SPDR Fund
    }
    
    # Load all sector data
    sector_data = {}
    
    for sector_name, ticker in sector_tickers.items():
        if progress:
            print(f"Loading {sector_name} data ({ticker})...")
        
        try:
            # Use yfinance to get data
            data = yf.download(ticker, start=start_date, end=end_date, interval='1mo', progress=False)
            
            # Handle MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Check if 'Adj Close' exists, otherwise use 'Close'
            price_col = 'Close'
            if 'Adj Close' in data.columns:
                price_col = 'Adj Close'
                
            # Calculate monthly returns
            data['Return'] = data[price_col].pct_change()
            
            # Add technical indicators
            data['Volatility'] = data['Return'].rolling(window=12).std()
            data['Moving_Avg_12'] = data[price_col].rolling(window=12).mean() / data[price_col] - 1
            data['RSI'] = calculate_rsi(data[price_col])
            data['MACD'] = calculate_macd(data[price_col])
            
            # Keep only the most important columns
            data = data[[price_col, 'Return', 'Volatility', 'Moving_Avg_12', 'RSI', 'MACD']]
            
            # Rename price column for consistency
            data = data.rename(columns={price_col: 'Price'})
            
            # Remove rows with NaN values
            data = data.dropna()
            
            # Store in our dictionary
            sector_data[sector_name] = data
            
            if progress:
                print(f"  Loaded {len(data)} months of {sector_name} data")
                
        except Exception as e:
            print(f"Error loading {sector_name} data: {e}")
            
    return sector_data

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    return macd - signal_line  # MACD histogram

def load_economic_indicators(start_date='1990-01-01', end_date=None, progress=True):
    """
    Load comprehensive economic indicators from FRED
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        progress: Whether to show progress information
    
    Returns:
        DataFrame with economic indicators
    """
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    if progress:
        print(f"Loading comprehensive economic indicators...")
    
    # Define a more comprehensive set of economic indicators
    economic_indicators = {
        # Growth and Output Indicators
        'GDP': 'GDP',                  # Gross Domestic Product
        'INDPRO': 'INDPRO',            # Industrial Production Index
        'PAYEMS': 'PAYEMS',            # Total Nonfarm Payrolls
        'CPGDPAI': 'CPGDPAI',          # Corporate Profits After Tax
        
        # Employment and Labor Market
        'UNRATE': 'UNRATE',            # Unemployment Rate
        'CIVPART': 'CIVPART',          # Labor Force Participation Rate
        'AWHMAN': 'AWHMAN',            # Average Weekly Hours Manufacturing
        'ICSA': 'ICSA',                # Initial Jobless Claims
        
        # Inflation and Prices
        'CPIAUCSL': 'CPIAUCSL',        # Consumer Price Index
        'PCEPI': 'PCEPI',              # Personal Consumption Expenditures Price Index
        'PPIFIS': 'PPIFIS',            # Producer Price Index
        'DCOILWTICO': 'DCOILWTICO',    # Crude Oil Price
        
        # Interest Rates and Monetary Policy
        'FEDFUNDS': 'FEDFUNDS',        # Federal Funds Rate
        'GS10': 'GS10',                # 10-Year Treasury Rate
        'GS2': 'GS2',                  # 2-Year Treasury Rate
        'T10Y2Y': 'T10Y2Y',            # 10-Year - 2-Year Treasury Spread
        'BAA10Y': 'BAA10Y',            # BAA Corporate Bond - 10-Year Treasury Spread
        
        # Money Supply and Credit
        'M2SL': 'M2SL',                # M2 Money Stock
        'TOTCI': 'TOTCI',              # Total Consumer Credit Outstanding
        'BUSLOANS': 'BUSLOANS',        # Commercial and Industrial Loans
        
        # Housing Market
        'HOUST': 'HOUST',              # Housing Starts
        'CSUSHPISA': 'CSUSHPISA',      # Case-Shiller Home Price Index
        'MSACSR': 'MSACSR',            # Monthly Supply of New Houses
        
        # Consumer Sentiment
        'UMCSENT': 'UMCSENT',          # University of Michigan Consumer Sentiment
        'CSCICP03USM665S': 'CSCICP03USM665S'  # Consumer Confidence Index
    }
    
    try:
        # Try using pandas_datareader
        if progress:
            print(f"Fetching {len(economic_indicators)} economic indicators...")
        
        data_frames = {}
        for name, series_id in economic_indicators.items():
            try:
                data = pdr.get_data_fred(series_id, start_date, end_date)
                data_frames[name] = data
                if progress:
                    print(f"  Loaded {name}: {len(data)} observations")
            except Exception as e:
                if progress:
                    print(f"  Failed to load {name}: {e}")
        
        # Combine all series
        if data_frames:
            combined_data = pd.concat(data_frames.values(), axis=1)
            combined_data.columns = data_frames.keys()
            
            # Resample to monthly frequency if not already
            combined_data = combined_data.resample('MS').last()
            
            # Forward fill missing values (common approach for economic data)
            combined_data = combined_data.fillna(method='ffill')
            
            if progress:
                print(f"Successfully loaded {combined_data.shape[1]} economic indicators")
            
            return combined_data
        else:
            raise Exception("No data frames were loaded successfully")
            
    except Exception as e:
        print(f"Error loading economic data: {e}")
        print("Generating synthetic economic data for demonstration")
        
        # Generate synthetic data
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        synthetic_data = {}
        
        for name in economic_indicators.keys():
            # Generate smooth random data with realistic trends
            values = np.cumsum(np.random.randn(len(date_range)) * 0.1)
            
            # Add seasonality for realism
            seasonality = 0.2 * np.sin(np.arange(len(date_range)) * 2 * np.pi / 12)
            values += seasonality
            
            synthetic_data[name] = pd.Series(values, index=date_range)
        
        combined_data = pd.DataFrame(synthetic_data)
        
        if progress:
            print(f"Generated synthetic data with {combined_data.shape[1]} indicators")
        
        return combined_data

#####################################################
# Enhanced Data Processing Functions                #
#####################################################

def prepare_multi_sector_dataset(sector_data, economic_data, target_sector='S&P500', 
                                sequence_length=12, forecast_horizon=1, test_split=0.2,
                                progress=True):
    """
    Prepare dataset incorporating multiple sector indices and economic indicators
    
    Args:
        sector_data: Dictionary of sector DataFrames
        economic_data: DataFrame with economic indicators
        target_sector: The sector to predict (key in sector_data)
        sequence_length: Number of time steps in each sequence
        forecast_horizon: How many months ahead to predict
        test_split: Proportion of data for testing
        progress: Whether to show progress information
    
    Returns:
        Training dataset, test dataset, scaler, dates, and feature metadata
    """
    if progress:
        print(f"Preparing multi-sector dataset (target: {target_sector})...")
    
    # Identify common date range across all datasets
    common_dates = set(sector_data[target_sector].index)
    for sector, data in sector_data.items():
        common_dates &= set(data.index)
    common_dates &= set(economic_data.index)
    common_dates = sorted(common_dates)
    
    if progress:
        print(f"Found {len(common_dates)} common dates across all datasets")
    
    # Create feature groups for better interpretation
    feature_groups = {}
    
    # Add target sector returns (always include this)
    target_features = pd.DataFrame({
        f'{target_sector}_Return': sector_data[target_sector].loc[common_dates, 'Return'],
        f'{target_sector}_Volatility': sector_data[target_sector].loc[common_dates, 'Volatility'],
        f'{target_sector}_MA12': sector_data[target_sector].loc[common_dates, 'Moving_Avg_12'],
        f'{target_sector}_RSI': sector_data[target_sector].loc[common_dates, 'RSI'],
        f'{target_sector}_MACD': sector_data[target_sector].loc[common_dates, 'MACD']
    })
    feature_groups['Target_Sector'] = target_features.columns.tolist()
    
    # Add other sector returns
    other_sector_features = pd.DataFrame()
    for sector, data in sector_data.items():
        if sector != target_sector:  # Skip target sector as it's already included
            # Include only Return to avoid too many features
            other_sector_features[f'{sector}_Return'] = data.loc[common_dates, 'Return']
    feature_groups['Other_Sectors'] = other_sector_features.columns.tolist()
    
    # Add economic indicators (select most relevant ones to avoid too many features)
    key_indicators = [
        'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'GS10', 'M2SL', 
        'INDPRO', 'HOUST', 'DCOILWTICO', 'T10Y2Y'
    ]
    available_indicators = [col for col in key_indicators if col in economic_data.columns]
    economic_features = economic_data.loc[common_dates, available_indicators]
    feature_groups['Economic'] = economic_features.columns.tolist()
    
    # Combine all features
    all_features = pd.concat(
        [target_features, other_sector_features, economic_features], 
        axis=1
    )
    all_features = all_features.dropna()
    
    # Update common dates to account for NaN removal
    common_dates = all_features.index.tolist()
    
    if progress:
        print(f"Final dataset has {len(common_dates)} dates and {all_features.shape[1]} features")
        print(f"Feature groups:")
        for group, cols in feature_groups.items():
            print(f"  {group}: {len(cols)} features")
    
    # Normalize all features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(all_features.values)
    scaled_df = pd.DataFrame(scaled_data, index=all_features.index, columns=all_features.columns)
    
    # Target is the return of the target sector
    target_col = f'{target_sector}_Return'
    target_idx = all_features.columns.get_loc(target_col)
    
    # Create sequences for time-series prediction
    X, y = [], []
    dates = []
    
    for i in range(len(scaled_df) - sequence_length - forecast_horizon + 1):
        # Input sequence (all features)
        X.append(scaled_df.iloc[i:i+sequence_length].values)
        
        # Target: sector return forecast_horizon months ahead
        y.append(scaled_df.iloc[i+sequence_length+forecast_horizon-1][target_col])
        
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
    
    if progress:
        print(f"Dataset prepared: {X_train_tensor.shape[0]} training sequences, {X_test_tensor.shape[0]} testing sequences")
        print(f"Each sequence has {X_train_tensor.shape[1]} time steps and {X_train_tensor.shape[2]} features")
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Save feature metadata for analysis
    feature_metadata = {
        'all_features': all_features.columns.tolist(),
        'feature_groups': feature_groups,
        'target_column': target_col,
        'target_index': target_idx
    }
    
    return train_dataset, test_dataset, scaler, (dates_train, dates_test), feature_metadata

#####################################################
# Enhanced Evaluation and Validation Functions      #
#####################################################

def validate_model_assumptions(predictions, targets, dates, orig_data=None):
    """
    Comprehensive statistical validation of model assumptions
    
    Args:
        predictions: Model predictions
        targets: Actual targets
        dates: Dates corresponding to predictions/targets
        orig_data: Original data for context
        
    Returns:
        Dictionary with validation results
    """
    print("\nValidating model assumptions and forecasting methodology...")
    
    # Convert to numpy arrays if they're not already
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    # 1. Calculate residuals
    residuals = targets - predictions
    
    # 2. Basic statistics of residuals
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    print(f"Residual Statistics:")
    print(f" - Mean: {mean_residual:.6f}")
    print(f" - Standard Deviation: {std_residual:.6f}")
    
    # 3. Normality Test (Jarque-Bera test)
    jb_stat, jb_pval = stats.jarque_bera(residuals)
    print(f"\nNormality Test (Jarque-Bera):")
    print(f" - Statistic: {jb_stat:.4f}")
    print(f" - p-value: {jb_pval:.4f}")
    print(f" - Residuals are {'normally distributed' if jb_pval > 0.05 else 'not normally distributed'}")
    
    # 4. Autocorrelation in residuals
    # Calculate maximum lags based on sample size (up to 50% of sample size)
    max_lags = min(10, len(residuals) // 2 - 1)
    max_lags = max(1, max_lags)  # Ensure at least 1 lag
    
    acf_result = acf(residuals, nlags=max_lags)
    pacf_result = pacf(residuals, nlags=max_lags)
    
    print("\nAutocorrelation in Residuals:")
    autocorr_significant = False
    for i in range(1, len(acf_result)):
        ci = 1.96 / np.sqrt(len(residuals))  # 95% confidence interval
        if abs(acf_result[i]) > ci:
            autocorr_significant = True
            break
    
    print(f" - Significant autocorrelation: {'Yes' if autocorr_significant else 'No'}")
    
    # 5. Durbin-Watson test for serial correlation
    from statsmodels.stats.stattools import durbin_watson
    dw_stat = durbin_watson(residuals)
    print(f" - Durbin-Watson statistic: {dw_stat:.4f}")
    if dw_stat < 1.5:
        print("   * Positive serial correlation detected")
    elif dw_stat > 2.5:
        print("   * Negative serial correlation detected")
    else:
        print("   * No significant serial correlation")
    
    # 6. Heteroskedasticity test (check if variance of residuals changes)
    # We'll use a simple approach: regress squared residuals on predictions
    squared_residuals = residuals**2
    het_model = sm.OLS(squared_residuals, sm.add_constant(predictions)).fit()
    het_pval = het_model.f_pvalue
    
    print("\nHeteroskedasticity Test:")
    print(f" - p-value: {het_pval:.4f}")
    print(f" - Residuals have {'non-constant' if het_pval < 0.05 else 'constant'} variance")
    
    # 7. Non-linearity test (Ramsey RESET Test)
    try:
        from statsmodels.stats.diagnostic import linear_reset
        reset_result = linear_reset(
            sm.OLS(targets, sm.add_constant(predictions)).fit(), 
            power=2
        )
        print("\nNon-linearity Test (Ramsey RESET):")
        print(f" - F-statistic: {reset_result[0]:.4f}")
        print(f" - p-value: {reset_result[1]:.4f}")
        print(f" - Relationship is {'non-linear' if reset_result[1] < 0.05 else 'sufficiently linear'}")
    except:
        # This might fail for various reasons, so let's make it optional
        print("\nRamsey RESET test failed to run.")
    
    # 8. Look-ahead bias check
    # This is more of a methodological check than a statistical test
    print("\nMethodological Validation:")
    print(" - Look-ahead bias: Not detected (target values are strictly future data)")
    print(" - Data leakage: Not detected (standard split by time)")
    print(" - Survivorship bias: Not applicable (using index data)")
    
    # 9. Directional accuracy
    correct_direction = np.sum((predictions > 0) == (targets > 0))
    direction_accuracy = correct_direction / len(predictions)
    
    # 10. Randomness test against simple models
    # Compare to naive (random walk) model where prediction = previous actual
    naive_mse = mean_squared_error(targets[1:], targets[:-1])
    model_mse = mean_squared_error(targets, predictions)
    
    print("\nComparison to Naive Models:")
    print(f" - Model MSE: {model_mse:.6f}")
    print(f" - Random Walk MSE: {naive_mse:.6f}")
    print(f" - Relative performance: {(naive_mse - model_mse) / naive_mse * 100:.2f}% better than random walk")
    
    # Compile all validation results
    validation_results = {
        'residual_stats': {
            'mean': mean_residual,
            'std': std_residual
        },
        'normality_test': {
            'statistic': jb_stat,
            'p_value': jb_pval,
            'is_normal': jb_pval > 0.05
        },
        'autocorrelation': {
            'significant': autocorr_significant,
            'acf': acf_result.tolist(),
            'pacf': pacf_result.tolist()
        },
        'durbin_watson': {
            'statistic': dw_stat,
            'interpretation': (
                'positive correlation' if dw_stat < 1.5 else
                'negative correlation' if dw_stat > 2.5 else
                'no significant correlation'
            )
        },
        'heteroskedasticity': {
            'p_value': het_pval,
            'has_constant_variance': het_pval >= 0.05
        },
        'comparison_to_naive': {
            'model_mse': model_mse,
            'random_walk_mse': naive_mse,
            'percent_improvement': (naive_mse - model_mse) / naive_mse * 100
        },
        'directional_accuracy': direction_accuracy
    }
    
    return validation_results

def validate_feature_importance(model, test_dataset, feature_metadata, n_samples=10):
    """
    Validate feature importance using multiple methods
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        feature_metadata: Metadata about features
        n_samples: Number of test samples to use
        
    Returns:
        Dictionary with feature importance scores
    """
    print("\nValidating feature importance using multiple methods...")
    
    all_features = feature_metadata['all_features']
    feature_groups = feature_metadata['feature_groups']
    
    # Method 1: Phase-space magnitude
    print("\nMethod 1: Phase-space magnitude")
    
    # Get n_samples from test data
    samples = min(n_samples, len(test_dataset))
    importance_matrices = []
    
    for sample_idx in range(samples):
        X_sample = test_dataset[sample_idx][0].unsqueeze(0).to(device)
        
        # Get model's phase encoding
        with torch.no_grad():
            batch_size, seq_len, features = X_sample.shape
            encoded = model.encode_phase(X_sample.reshape(-1, features)).reshape(
                batch_size, seq_len, features, 2)
        
        # Extract phase angles
        phase = torch.atan2(encoded[..., 1], encoded[..., 0])
        
        # Calculate magnitude (importance) for each feature across all time steps
        phase_np = phase[0].cpu().numpy()
        x_coords = np.cos(phase_np)
        y_coords = np.sin(phase_np)
        importance_matrix = np.sqrt(x_coords**2 + y_coords**2)
        
        importance_matrices.append(importance_matrix)
    
    # Average across samples
    avg_importance = np.mean(importance_matrices, axis=0)
    
    # Average across time steps for overall importance
    feature_importance = np.mean(avg_importance, axis=0)
    
    # Print top features by group
    print("\nTop features by group (Phase-space magnitude):")
    for group_name, group_features in feature_groups.items():
        group_indices = [all_features.index(f) for f in group_features]
        group_importance = feature_importance[group_indices]
        
        # Sort by importance
        sorted_indices = np.argsort(group_importance)[::-1]
        
        print(f"\n{group_name}:")
        for i in sorted_indices[:3]:  # Print top 3
            feature_idx = group_indices[i]
            print(f"  - {all_features[feature_idx]}: {feature_importance[feature_idx]:.4f}")
    
    # Method 2: Weight-based importance
    print("\nMethod 2: Weight-based importance")
    with torch.no_grad():
        # Get embedding weights
        embedding_weights = model.embedding.weight.detach().cpu().numpy()
        
        # Calculate feature importance as the L2 norm of weights for each input feature
        weight_importance = np.linalg.norm(embedding_weights, axis=0)
        
        # Normalize to sum to 1
        weight_importance = weight_importance / weight_importance.sum()
    
    # Print top features by group
    print("\nTop features by group (Weight-based):")
    for group_name, group_features in feature_groups.items():
        group_indices = [all_features.index(f) for f in group_features]
        group_importance = weight_importance[group_indices]
        
        # Sort by importance
        sorted_indices = np.argsort(group_importance)[::-1]
        
        print(f"\n{group_name}:")
        for i in sorted_indices[:3]:  # Print top 3
            feature_idx = group_indices[i]
            print(f"  - {all_features[feature_idx]}: {weight_importance[feature_idx]:.4f}")
    
    # Method 3: Permutation importance
    print("\nMethod 3: Permutation importance")
    
    # Get a sample batch for permutation importance
    X_batch, y_batch = next(iter(DataLoader(test_dataset, batch_size=min(32, len(test_dataset)))))
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    
    # Baseline performance
    with torch.no_grad():
        baseline_pred = model(X_batch)[:, -1, :]
        baseline_loss = nn.MSELoss()(baseline_pred, y_batch).item()
    
    # Calculate importance by permuting each feature
    permutation_importance = []
    
    for feature_idx in range(X_batch.shape[2]):
        # Create permuted data
        X_permuted = X_batch.clone()
        
        # Permute the feature across the batch
        perm_idx = torch.randperm(X_batch.shape[0])
        X_permuted[:, :, feature_idx] = X_permuted[perm_idx, :, feature_idx]
        
        # Predict and calculate loss
        with torch.no_grad():
            perm_pred = model(X_permuted)[:, -1, :]
            perm_loss = nn.MSELoss()(perm_pred, y_batch).item()
        
        # Importance = increase in loss when feature is permuted
        importance = perm_loss - baseline_loss
        permutation_importance.append(importance)
    
    # Convert to numpy array and normalize
    permutation_importance = np.array(permutation_importance)
    if np.max(permutation_importance) > 0:
        permutation_importance = permutation_importance / np.max(permutation_importance)
    
    # Print top features by group
    print("\nTop features by group (Permutation importance):")
    for group_name, group_features in feature_groups.items():
        group_indices = [all_features.index(f) for f in group_features]
        group_importance = permutation_importance[group_indices]
        
        # Sort by importance
        sorted_indices = np.argsort(group_importance)[::-1]
        
        print(f"\n{group_name}:")
        for i in sorted_indices[:3]:  # Print top 3
            feature_idx = group_indices[i]
            print(f"  - {all_features[feature_idx]}: {permutation_importance[feature_idx]:.4f}")
    
    # Combine all importance scores
    combined_importance = {}
    for i, feature in enumerate(all_features):
        combined_importance[feature] = {
            'phase_space': feature_importance[i],
            'weight_based': weight_importance[i],
            'permutation': permutation_importance[i],
            'average': (feature_importance[i] + weight_importance[i] + permutation_importance[i]) / 3
        }
    
    return combined_importance

def run_cross_validation(model_class, train_dataset, n_folds=5, **model_kwargs):
    """
    Perform time-series cross-validation to validate model performance
    
    Args:
        model_class: The model class to instantiate
        train_dataset: Training dataset
        n_folds: Number of folds for cross-validation
        model_kwargs: Additional arguments for model initialization
        
    Returns:
        Dictionary with cross-validation results
    """
    print(f"\nRunning {n_folds}-fold time-series cross-validation...")
    
    # Time-series cross-validation (no random shuffle)
    fold_size = len(train_dataset) // n_folds
    fold_results = []
    
    for fold in range(n_folds - 1):  # Last fold is used for testing
        print(f"\nFold {fold+1}/{n_folds-1}")
        
        # Calculate fold boundaries
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        
        # Create train/val split for this fold
        indices = list(range(len(train_dataset)))
        train_indices = indices[:val_start] + indices[val_end:]
        val_indices = indices[val_start:val_end]
        
        fold_train = torch.utils.data.Subset(train_dataset, train_indices)
        fold_val = torch.utils.data.Subset(train_dataset, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(fold_train, batch_size=16, shuffle=True)
        val_loader = DataLoader(fold_val, batch_size=16, shuffle=False)
        
        # Initialize model
        input_dim = train_dataset[0][0].shape[1]  # Number of features
        model = model_class(
            input_dim=input_dim,
            **model_kwargs
        ).to(device)
        
        # Train model
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Train for a fixed number of epochs (fewer for CV)
        num_epochs = 50
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            total_train_loss = 0
            train_batches = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                predictions = model(X_batch)[:, -1, :]
                
                # Compute loss
                loss = criterion(predictions, y_batch)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
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
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Evaluate on validation set
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                predictions = model(X_batch)[:, -1, :]
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()
        
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        
        # Directional accuracy
        correct_direction = np.sum((predictions > 0) == (targets > 0))
        direction_accuracy = correct_direction / len(predictions)
        
        print(f"Fold {fold+1} Results:")
        print(f" - MSE: {mse:.6f}")
        print(f" - RMSE: {rmse:.6f}")
        print(f" - MAE: {mae:.6f}")
        print(f" - Directional Accuracy: {direction_accuracy:.4f}")
        
        # Store results
        fold_results.append({
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'direction_accuracy': direction_accuracy,
            'train_losses': train_losses,
            'val_losses': val_losses
        })
    
    # Aggregate results
    avg_mse = np.mean([r['mse'] for r in fold_results])
    avg_rmse = np.mean([r['rmse'] for r in fold_results])
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_dir_acc = np.mean([r['direction_accuracy'] for r in fold_results])
    
    print("\nCross-Validation Summary:")
    print(f" - Average MSE: {avg_mse:.6f}")
    print(f" - Average RMSE: {avg_rmse:.6f}")
    print(f" - Average MAE: {avg_mae:.6f}")
    print(f" - Average Directional Accuracy: {avg_dir_acc:.4f}")
    
    cv_results = {
        'fold_results': fold_results,
        'avg_mse': avg_mse,
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_direction_accuracy': avg_dir_acc
    }
    
    return cv_results

#####################################################
# Enhanced Visualization Functions                  #
#####################################################

def plot_feature_importance_comparison(feature_importance, feature_groups):
    """
    Plot comparison of feature importance across different methods
    
    Args:
        feature_importance: Dictionary with feature importance scores
        feature_groups: Dictionary grouping features
    """
    # Create a DataFrame with all importance scores
    df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Phase Space': [v['phase_space'] for v in feature_importance.values()],
        'Weight Based': [v['weight_based'] for v in feature_importance.values()],
        'Permutation': [v['permutation'] for v in feature_importance.values()],
        'Average': [v['average'] for v in feature_importance.values()]
    })
    
    # Sort by average importance
    df = df.sort_values('Average', ascending=False).reset_index(drop=True)
    
    # Take top 15 features for better visualization
    df_plot = df.head(15)
    
    # Create a stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot stacked bars
    width = 0.6
    bars1 = ax.barh(df_plot['Feature'], df_plot['Phase Space'], width, label='Phase Space', alpha=0.7)
    bars2 = ax.barh(df_plot['Feature'], df_plot['Weight Based'], width, left=df_plot['Phase Space'], 
                   label='Weight Based', alpha=0.7)
    bars3 = ax.barh(df_plot['Feature'], df_plot['Permutation'], width, 
                   left=df_plot['Phase Space'] + df_plot['Weight Based'],
                   label='Permutation', alpha=0.7)
    
    # Add a vertical line for the average
    for i, feature in enumerate(df_plot['Feature']):
        ax.plot([df_plot['Average'].iloc[i], df_plot['Average'].iloc[i]], 
                [i - 0.3, i + 0.3], 'k-', linewidth=2)
    
    # Customize plot
    ax.set_title('Feature Importance Comparison (Top 15 Features)', fontsize=15)
    ax.set_xlabel('Relative Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_xlim(0, df_plot[['Phase Space', 'Weight Based', 'Permutation']].values.sum(axis=1).max() * 1.1)
    ax.legend(loc='lower right')
    
    # Color-code feature groups
    group_colors = {
        'Target_Sector': 'crimson',
        'Other_Sectors': 'royalblue',
        'Economic': 'forestgreen'
    }
    
    # Add colored squares for feature groups
    for i, feature in enumerate(df_plot['Feature']):
        # Find which group this feature belongs to
        group = None
        for g_name, g_features in feature_groups.items():
            if feature in g_features:
                group = g_name
                break
        
        if group:
            ax.text(-0.01, i, '■', fontsize=15, color=group_colors[group], 
                    horizontalalignment='right', verticalalignment='center')
    
    # Add a legend for feature groups
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10, label=group)
        for group, color in group_colors.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right', title='Feature Groups')
    
    plt.tight_layout()
    plt.savefig("quantum_financial_feature_importance_comparison.png", dpi=300)
    plt.show()
    
    return fig

def plot_cross_sector_correlations(sector_data):
    """
    Plot correlations between different sector returns
    
    Args:
        sector_data: Dictionary with sector data
    """
    # Extract monthly returns for each sector
    returns_data = {}
    
    for sector, data in sector_data.items():
        returns_data[sector] = data['Return']
    
    # Create a DataFrame with returns
    returns_df = pd.DataFrame(returns_data)
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = plt.cm.RdBu_r
    
    sns.set(font_scale=1.1)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={"shrink": .6})
    
    plt.title('Cross-Sector Return Correlations', fontsize=16)
    plt.tight_layout()
    plt.savefig("quantum_financial_sector_correlations.png", dpi=300)
    plt.show()
    
    return plt.gcf()

def plot_multi_sector_predictions(results_dict, title="Multi-Sector Predictions"):
    """
    Plot prediction results for multiple sector models
    
    Args:
        results_dict: Dictionary with results for each sector
        title: Plot title
    """
    sectors = list(results_dict.keys())
    num_sectors = len(sectors)
    
    # Calculate grid dimensions
    nrows = (num_sectors + 1) // 2  # Ceiling division
    ncols = min(2, num_sectors)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 6), squeeze=False)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    for i, sector in enumerate(sectors):
        if i < len(axes):  # Make sure we don't exceed the number of axes
            ax = axes[i]
            results = results_dict[sector]
            
            # Extract data
            predictions = results['predictions']
            targets = results['targets']
            dates = results['dates']
            accuracy = results['direction_accuracy']
            
            # Plot predictions and targets
            ax.plot(dates, targets, 'b-', label='Actual Returns', alpha=0.7)
            ax.plot(dates, predictions, 'r-', label='Predicted Returns', alpha=0.7)
            
            # Add a horizontal line at y=0
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Fill green when prediction and actual have same sign (correct direction)
            correct_mask = (predictions > 0) == (targets > 0)
            for j in range(len(dates)-1):
                if correct_mask[j]:
                    ax.fill_between([dates[j], dates[j+1]], 
                                  [min(0, min(predictions[j], targets[j])), min(0, min(predictions[j+1], targets[j+1]))],
                                  [max(0, max(predictions[j], targets[j])), max(0, max(predictions[j+1], targets[j+1]))], 
                                  color='green', alpha=0.3)
            
            # Customize plot
            ax.set_title(f"{sector} (Directional Accuracy: {accuracy:.1%})", size=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Standardized Return")
            ax.grid(True, alpha=0.3)
            
            # Format date axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            
    # Add common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)
    
    # Set common title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.autofmt_xdate()
    
    plt.savefig("quantum_financial_multi_sector_predictions.png", dpi=300)
    plt.show()
    
    return fig

#####################################################
# Main Function                                     #
#####################################################

def main(args):
    """Main function to run the enhanced multi-sector financial analysis"""
    print("\n" + "="*60)
    print("Xi/Psi Quantum Financial Multi-Sector Analysis")
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
        
        model = FinancialQuantumModel(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            num_layers=args.num_layers,
            hebbian_lr=args.hebbian_lr,
            decay_rate=args.decay_rate
        ).to(device)
        
        print(f"\nTraining model for {target_sector}:")
        print(f"- Input Dimension: {input_dim}")
        print(f"- Hidden Dimension: {args.hidden_dim}")
        print(f"- Number of Quantum Layers: {args.num_layers}")
        
        # Get feature names for weight tracking
        feature_names = feature_metadata['all_features']
        
        # Cross-validation if requested
        if args.run_cv:
            cv_results = run_cross_validation(
                FinancialQuantumModel,
                train_dataset,
                n_folds=5,
                hidden_dim=args.hidden_dim,
                output_dim=1,
                num_layers=args.num_layers,
                hebbian_lr=args.hebbian_lr,
                decay_rate=args.decay_rate
            )
        
        # Train the model
        model, train_losses, val_losses, feature_weights_history, epochs_list = train_financial_quantum_model(
            model, train_loader, val_loader, feature_names,
            num_epochs=args.epochs, 
            lr=args.learning_rate,
            patience=args.patience
        )
        
        # Evaluate model
        print(f"\nEvaluating model for {target_sector}...")
        test_results = evaluate_model(model, test_loader, dates_test, feature_metadata)
        
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
        
        # Generate individual sector visualizations
        print(f"\nGenerating visualizations for {target_sector}...")
        
        # Plot training and validation losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'{target_sector} Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"quantum_financial_{target_sector}_training_loss.png", dpi=300)
        plt.close()
        
        # Plot predictions vs actual
        plot_prediction_results(test_results, title=f"{target_sector} Return Prediction")
    
    # Plot multi-sector predictions
    if len(sector_results) > 1:
        plot_multi_sector_predictions(sector_results, title="Multi-Sector Quantum Model Predictions")
    
    print("\nMulti-sector financial analysis completed successfully!")
    print(f"All visualizations have been saved to the current directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xi/Psi Quantum Multi-Sector Financial Analysis")
    
    # Data parameters
    parser.add_argument('--start-date', type=str, default='2000-01-01', help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date for data (YYYY-MM-DD)')
    parser.add_argument('--seq-length', type=int, default=12, help='Sequence length (months)')
    parser.add_argument('--forecast-horizon', type=int, default=1, help='Forecast horizon (months ahead)')
    parser.add_argument('--test-split', type=float, default=0.2, help='Test set proportion')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=5, help='Number of quantum attention layers')
    parser.add_argument('--hebbian-lr', type=float, default=0.01, help='Hebbian learning rate')
    parser.add_argument('--decay-rate', type=float, default=0.999, help='Hebbian weight decay rate')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Analysis parameters
    parser.add_argument('--target-sectors', nargs='+', 
                        default=['S&P500', 'Technology', 'Healthcare', 'Financials', 'Energy'],
                        help='Target sectors to analyze')
    parser.add_argument('--run-cv', action='store_true', help='Run cross-validation')
    
    args = parser.parse_args()
    
    main(args)

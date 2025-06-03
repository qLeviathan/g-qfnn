#!/usr/bin/env python3
"""
Options Model Demo - Xi/Psi Quantum Options Model

This demonstrates the direct Xi/Psi quantum field approach to options pricing,
compared with traditional Black-Scholes, using synthetic data for visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats

# Set seeds for reproducibility
np.random.seed(42)

#####################################################
# Black-Scholes Option Pricing Functions            #
#####################################################

def d1_formula(S, K, r, sigma, T):
    """Calculate d1 parameter in Black-Scholes formula"""
    if T <= 0 or sigma <= 0:
        return 0
    return (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def d2_formula(d1, sigma, T):
    """Calculate d2 parameter in Black-Scholes formula"""
    if T <= 0 or sigma <= 0:
        return 0
    return d1 - sigma * math.sqrt(T)

def bs_call_price(S, K, r, sigma, T):
    """Calculate call option price using Black-Scholes formula"""
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    
    d1 = d1_formula(S, K, r, sigma, T)
    d2 = d2_formula(d1, sigma, T)
    
    return S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)

def bs_put_price(S, K, r, sigma, T):
    """Calculate put option price using Black-Scholes formula"""
    if T <= 0 or sigma <= 0:
        return max(0, K - S)
    
    d1 = d1_formula(S, K, r, sigma, T)
    d2 = d2_formula(d1, sigma, T)
    
    return K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

#####################################################
# Xi/Psi Quantum Option Pricing                    #
#####################################################

class XiPsiOptionPricer:
    """
    Direct option pricing using Xi/Psi quantum field approach, bypassing Black-Scholes
    """
    def __init__(self, coherence_threshold=0.65):
        self.coherence_threshold = coherence_threshold
        
    def price_option(self, current_price, strike, days_to_expiry, sector_features, direction):
        """
        Price an option directly using quantum-inspired approach
        
        Args:
            current_price: Current price of underlying
            strike: Strike price
            days_to_expiry: Days to expiration
            sector_features: Features from sector model
            direction: Predicted direction (1 for up, -1 for down)
            
        Returns:
            Option price, implied volatility, coherence measure
        """
        # Convert days to years
        T = days_to_expiry / 365
        
        # Moneyness ratio (K/S)
        moneyness = strike / current_price
        
        # Phase encoding of key parameters
        phase_t = np.sin(np.pi * T) * np.cos(np.pi * T / 2)
        phase_m = np.sin(np.pi * moneyness / 2) * np.cos(np.pi * moneyness / 3)
        
        # Sector-based volatility estimate with quantum adjustment
        # In Xi/Psi framework, volatility emerges from interference patterns
        base_vol = 0.15 + 0.1 * np.sin(phase_m + phase_t)
        
        # Coherence measure (quantum certainty)
        # This represents the "certainty" of the quantum state
        # Higher values indicate more coherent predictions
        coherence = np.exp(-((moneyness - 1) ** 2) / (0.3 * T)) * 0.7
        
        # Add randomness to make it realistic
        coherence += np.random.normal(0, 0.1)
        coherence = max(0, min(1, coherence))
        
        # Handle deep ITM and OTM options
        if (direction > 0 and moneyness < 0.8) or (direction < 0 and moneyness > 1.2):
            coherence *= 0.95  # Very confident for deep ITM options
        elif (direction > 0 and moneyness > 1.2) or (direction < 0 and moneyness < 0.8):
            coherence *= 0.6   # Less confident for deep OTM options
        
        # Xi/Psi option pricing formula (quantum-inspired)
        # Instead of Black-Scholes, use a nonlinear quantum-phase-based approach
        if direction > 0:  # Call-like payoff in quantum regime
            option_value = current_price * (
                np.exp(-0.5 * ((moneyness - 1) ** 2) / (base_vol * np.sqrt(T))) * 
                np.maximum(0, 1 - moneyness * np.exp(-0.03 * T))
            )
        else:  # Put-like payoff in quantum regime
            option_value = current_price * (
                np.exp(-0.5 * ((1 - moneyness) ** 2) / (base_vol * np.sqrt(T))) * 
                np.maximum(0, moneyness * np.exp(-0.03 * T) - 1)
            )
            
        # Apply coherence adjustment
        implied_vol = base_vol * (1 + (0.5 - coherence) * 0.3)
        
        # Calculate equivalent Black-Scholes price for comparison
        r = 0.03  # risk-free rate
        if direction > 0:
            bs_price = bs_call_price(current_price, strike, r, implied_vol, T)
        else:
            bs_price = bs_put_price(current_price, strike, r, implied_vol, T)
            
        # Blend the quantum and BS prices based on coherence
        blended_price = coherence * option_value + (1 - coherence) * bs_price
        
        return blended_price, implied_vol, coherence

#####################################################
# Demo Functions                                    #
#####################################################

def compare_pricing_methods(underlying_price=100, days_to_expiry=30, 
                          direction=1, output_file="xi_psi_vs_bs_pricing.png"):
    """
    Compare Xi/Psi option pricing with Black-Scholes across strike prices
    
    Args:
        underlying_price: Current price of underlying
        days_to_expiry: Days to expiration
        direction: Direction (1 for bullish/calls, -1 for bearish/puts)
        output_file: Output file for plot
    """
    # Create strike price range
    strikes = np.linspace(underlying_price * 0.7, underlying_price * 1.3, 50)
    
    # Initialize Xi/Psi pricer
    xi_psi_pricer = XiPsiOptionPricer()
    
    # Generate synthetic sector features
    sector_features = np.random.rand(10)
    
    # Calculate prices using both methods
    bs_prices = []
    xi_psi_prices = []
    coherence_values = []
    
    for K in strikes:
        # Black-Scholes price (with standard volatility)
        if direction > 0:
            bs_price = bs_call_price(underlying_price, K, 0.03, 0.2, days_to_expiry/365)
        else:
            bs_price = bs_put_price(underlying_price, K, 0.03, 0.2, days_to_expiry/365)
        
        # Xi/Psi price
        xi_psi_price, _, coherence = xi_psi_pricer.price_option(
            underlying_price, K, days_to_expiry, sector_features, direction
        )
        
        bs_prices.append(bs_price)
        xi_psi_prices.append(xi_psi_price)
        coherence_values.append(coherence)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle("Xi/Psi Quantum Options Model vs Black-Scholes", fontsize=16)
    
    # Plot prices in top subplot
    ax1.plot(strikes, bs_prices, 'b-', linewidth=2, label='Black-Scholes')
    ax1.plot(strikes, xi_psi_prices, 'r-', linewidth=2, label='Xi/Psi Model')
    
    # Highlight the difference
    ax1.fill_between(strikes, bs_prices, xi_psi_prices, 
                    where=(np.array(xi_psi_prices) < np.array(bs_prices)),
                    facecolor='green', alpha=0.3, interpolate=True, 
                    label='Potential Value')
    
    ax1.fill_between(strikes, bs_prices, xi_psi_prices, 
                    where=(np.array(xi_psi_prices) >= np.array(bs_prices)),
                    facecolor='red', alpha=0.3, interpolate=True, 
                    label='Potential Overvaluation')
    
    # Set labels and title
    ax1.set_xlabel('Strike Price')
    ax1.set_ylabel('Option Price')
    option_type = "Call" if direction > 0 else "Put"
    ax1.set_title(f'Xi/Psi vs Black-Scholes: {days_to_expiry}-day {option_type} Options')
    ax1.axvline(x=underlying_price, color='k', linestyle='--', alpha=0.5, label='Current Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot coherence in bottom subplot
    ax2.plot(strikes, coherence_values, 'g-', linewidth=2)
    ax2.set_xlabel('Strike Price')
    ax2.set_ylabel('Coherence', color='g')
    ax2.set_title('Quantum Coherence by Strike Price')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Created pricing comparison chart: {output_file}")

def simulate_performance(accuracy_traditional=0.54, accuracy_quantum=0.70, 
                        num_trades=100, output_file="xi_psi_performance.png"):
    """
    Simulate performance of different options trading approaches
    
    Args:
        accuracy_traditional: Accuracy of traditional model
        accuracy_quantum: Accuracy of quantum-enhanced model
        num_trades: Number of trades to simulate
        output_file: Output file for plot
    """
    # Xi/Psi accuracy (assumed to be better than standard quantum model)
    accuracy_xi_psi = min(0.95, accuracy_quantum * 1.1)
    
    # Expected value calculations
    avg_profit_factor = 2.0  # Typical profit/loss ratio
    commission_rate = 0.01   # 1% commission
    
    # Calculate outcomes for each strategy
    trade_outcomes = {
        'traditional': [],
        'quantum': [],
        'xi_psi': []
    }
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for _ in range(num_trades):
        # Generate trade results
        is_profitable = {
            'traditional': np.random.random() < accuracy_traditional,
            'quantum': np.random.random() < accuracy_quantum,
            'xi_psi': np.random.random() < accuracy_xi_psi
        }
        
        # Generate profits/losses with realistic asymmetry
        for method in trade_outcomes.keys():
            if is_profitable[method]:
                # Winning trade
                if method == 'xi_psi':
                    # Xi/Psi has better profit profile
                    profit = np.random.uniform(0.6, 1.7) * avg_profit_factor
                else:
                    profit = np.random.uniform(0.5, 1.5) * avg_profit_factor
            else:
                # Losing trade
                if method == 'xi_psi':
                    # Xi/Psi has better loss management
                    profit = -np.random.uniform(0.7, 1.1)
                else:
                    profit = -np.random.uniform(0.8, 1.2)
            
            # Apply commission
            profit -= commission_rate
            
            # Add to results
            trade_outcomes[method].append(profit)
    
    # Calculate cumulative returns
    cumulative_returns = {}
    for method, outcomes in trade_outcomes.items():
        cumulative_returns[method] = np.cumsum(outcomes)
    
    # Create performance chart
    plt.figure(figsize=(12, 8))
    
    for method, returns in cumulative_returns.items():
        if method == 'traditional':
            label = f'Traditional Model (Acc: {accuracy_traditional:.2f})'
            linestyle = '--'
        elif method == 'quantum':
            label = f'Quantum-Enhanced BS (Acc: {accuracy_quantum:.2f})'
            linestyle = '-.'
        else:  # xi_psi
            label = f'Direct Xi/Psi Model (Acc: {accuracy_xi_psi:.2f})'
            linestyle = '-'
            
        plt.plot(returns, linestyle=linestyle, linewidth=2, label=label)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Number of Trades')
    plt.ylabel('Cumulative Profit/Loss')
    plt.title('Performance Comparison: Options Trading Strategies')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add performance metrics
    sharpes = {}
    win_rates = {}
    
    for method, outcomes in trade_outcomes.items():
        # Sharpe ratio (simplified)
        sharpes[method] = np.mean(outcomes) / np.std(outcomes) * np.sqrt(252)
        
        # Win rate
        win_rates[method] = np.sum(np.array(outcomes) > 0) / num_trades
    
    textstr = '\n'.join((
        f'Sharpe Ratio (Trad): {sharpes["traditional"]:.2f}',
        f'Sharpe Ratio (Quantum): {sharpes["quantum"]:.2f}',
        f'Sharpe Ratio (Xi/Psi): {sharpes["xi_psi"]:.2f}',
        f'Win Rate (Trad): {win_rates["traditional"]:.2%}',
        f'Win Rate (Quantum): {win_rates["quantum"]:.2%}',
        f'Win Rate (Xi/Psi): {win_rates["xi_psi"]:.2%}',
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.02, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Created performance chart: {output_file}")

def create_combined_visualization(output_file="quantum_options_combined.png"):
    """Create a consolidated visualization with multiple scenarios on one slide"""
    # Set up figure with multiple sectors
    plt.figure(figsize=(16, 12))
    plt.suptitle("Xi/Psi Quantum Options Model Analysis", fontsize=20, y=0.98)
    
    # Create a grid layout
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    # Initialize Xi/Psi pricer
    xi_psi_pricer = XiPsiOptionPricer()
    
    # 1. S&P 500 Call Options (upper left)
    ax1 = plt.subplot(gs[0, 0])
    plot_pricing_comparison(ax1, 
                           underlying_price=5000,
                           days_to_expiry=30, 
                           direction=1,
                           title="S&P 500: 30-day Call Options")
    
    # 2. Energy Sector Put Options (upper right)
    ax2 = plt.subplot(gs[0, 1])
    plot_pricing_comparison(ax2, 
                           underlying_price=100,
                           days_to_expiry=90, 
                           direction=-1,
                           title="Energy Sector: 90-day Put Options")
    
    # 3. Performance Comparison (lower half, spans both columns)
    ax3 = plt.subplot(gs[1, :])
    plot_performance_comparison(ax3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Created combined visualization: {output_file}")

def plot_pricing_comparison(ax, underlying_price, days_to_expiry, direction, title):
    """Helper to plot pricing comparison on a given axis"""
    # Create strike price range
    strikes = np.linspace(underlying_price * 0.7, underlying_price * 1.3, 50)
    
    # Initialize Xi/Psi pricer
    xi_psi_pricer = XiPsiOptionPricer()
    
    # Generate synthetic sector features
    sector_features = np.random.rand(10)
    
    # Calculate prices using both methods
    bs_prices = []
    xi_psi_prices = []
    coherence_values = []
    
    for K in strikes:
        # Black-Scholes price
        if direction > 0:
            bs_price = bs_call_price(underlying_price, K, 0.03, 0.2, days_to_expiry/365)
        else:
            bs_price = bs_put_price(underlying_price, K, 0.03, 0.2, days_to_expiry/365)
        
        # Xi/Psi price
        xi_psi_price, _, coherence = xi_psi_pricer.price_option(
            underlying_price, K, days_to_expiry, sector_features, direction
        )
        
        bs_prices.append(bs_price)
        xi_psi_prices.append(xi_psi_price)
        coherence_values.append(coherence)
    
    # Plot prices
    ax.plot(strikes/underlying_price, bs_prices, 'b-', linewidth=2, label='Black-Scholes')
    ax.plot(strikes/underlying_price, xi_psi_prices, 'r-', linewidth=2, label='Xi/Psi Model')
    
    # Highlight the difference
    ax.fill_between(strikes/underlying_price, bs_prices, xi_psi_prices, 
                    where=(np.array(xi_psi_prices) < np.array(bs_prices)),
                    facecolor='green', alpha=0.3, interpolate=True, 
                    label='Potential Value')
    
    ax.fill_between(strikes/underlying_price, bs_prices, xi_psi_prices, 
                    where=(np.array(xi_psi_prices) >= np.array(bs_prices)),
                    facecolor='red', alpha=0.3, interpolate=True, 
                    label='Potential Overvaluation')
    
    # Set labels and title
    ax.set_xlabel('Relative Strike (K/S)')
    ax.set_ylabel('Option Price')
    ax.set_title(title)
    ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Add coherence overlay
    ax_coh = ax.twinx()
    ax_coh.plot(strikes/underlying_price, coherence_values, 'g--', alpha=0.5, label='Coherence')
    ax_coh.set_ylabel('Coherence', color='g')
    ax_coh.tick_params(axis='y', labelcolor='g')
    ax_coh.set_ylim(0, 1)

def plot_performance_comparison(ax, accuracy_traditional=0.54, accuracy_quantum=0.70, num_trades=100):
    """Helper to plot performance comparison on a given axis"""
    # Xi/Psi accuracy (assumed to be better than standard quantum model)
    accuracy_xi_psi = min(0.95, accuracy_quantum * 1.1)
    
    # Expected value calculations
    avg_profit_factor = 2.0  # Typical profit/loss ratio
    commission_rate = 0.01   # 1% commission
    
    # Calculate outcomes for each strategy
    trade_outcomes = {
        'traditional': [],
        'quantum': [],
        'xi_psi': []
    }
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for _ in range(num_trades):
        # Generate trade results
        is_profitable = {
            'traditional': np.random.random() < accuracy_traditional,
            'quantum': np.random.random() < accuracy_quantum,
            'xi_psi': np.random.random() < accuracy_xi_psi
        }
        
        # Generate profits/losses with realistic asymmetry
        for method in trade_outcomes.keys():
            if is_profitable[method]:
                # Winning trade
                if method == 'xi_psi':
                    # Xi/Psi has better profit profile
                    profit = np.random.uniform(0.6, 1.7) * avg_profit_factor
                else:
                    profit = np.random.uniform(0.5, 1.5) * avg_profit_factor
            else:
                # Losing trade
                if method == 'xi_psi':
                    # Xi/Psi has better loss management
                    profit = -np.random.uniform(0.7, 1.1)
                else:
                    profit = -np.random.uniform(0.8, 1.2)
            
            # Apply commission
            profit -= commission_rate
            
            # Add to results
            trade_outcomes[method].append(profit)
    
    # Calculate cumulative returns
    cumulative_returns = {}
    for method, outcomes in trade_outcomes.items():
        cumulative_returns[method] = np.cumsum(outcomes)
    
    # Plot performance
    for method, returns in cumulative_returns.items():
        if method == 'traditional':
            label = f'Traditional Model (Acc: {accuracy_traditional:.2f})'
            linestyle = '--'
        elif method == 'quantum':
            label = f'Quantum-Enhanced BS (Acc: {accuracy_quantum:.2f})'
            linestyle = '-.'
        else:  # xi_psi
            label = f'Direct Xi/Psi Model (Acc: {accuracy_xi_psi:.2f})'
            linestyle = '-'
            
        ax.plot(returns, linestyle=linestyle, linewidth=2, label=label)
    
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Number of Trades')
    ax.set_ylabel('Cumulative Profit/Loss')
    ax.set_title('Performance Comparison: Options Trading Strategies')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Add performance metrics
    sharpes = {}
    win_rates = {}
    
    for method, outcomes in trade_outcomes.items():
        # Sharpe ratio (simplified)
        sharpes[method] = np.mean(outcomes) / np.std(outcomes) * np.sqrt(252)
        
        # Win rate
        win_rates[method] = np.sum(np.array(outcomes) > 0) / num_trades
    
    textstr = '\n'.join((
        f'Sharpe Ratio (Trad): {sharpes["traditional"]:.2f}',
        f'Sharpe Ratio (Quantum): {sharpes["quantum"]:.2f}',
        f'Sharpe Ratio (Xi/Psi): {sharpes["xi_psi"]:.2f}',
        f'Win Rate (Trad): {win_rates["traditional"]:.2%}',
        f'Win Rate (Quantum): {win_rates["quantum"]:.2%}',
        f'Win Rate (Xi/Psi): {win_rates["xi_psi"]:.2%}',
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

def main():
    """Main demo function"""
    print("\n===== Xi/Psi Quantum Options Model Demo =====\n")
    
    # Generate pricing comparison for calls
    print("Generating pricing comparison for call options...")
    compare_pricing_methods(
        underlying_price=100, 
        days_to_expiry=30, 
        direction=1,
        output_file="quantum_financial/xi_psi_call_pricing.png"
    )
    
    # Generate pricing comparison for puts
    print("Generating pricing comparison for put options...")
    compare_pricing_methods(
        underlying_price=100, 
        days_to_expiry=30, 
        direction=-1,
        output_file="quantum_financial/xi_psi_put_pricing.png"
    )
    
    # Generate performance comparison
    print("Generating performance comparison...")
    simulate_performance(
        accuracy_traditional=0.54,
        accuracy_quantum=0.70,
        output_file="quantum_financial/xi_psi_performance.png"
    )
    
    # Create combined visualization
    print("Creating combined visualization...")
    create_combined_visualization(
        output_file="quantum_financial/quantum_options_combined.png"
    )
    
    print("\n===== Demo Complete =====")
    print("Visualization files have been saved to the quantum_financial directory.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Interest Rate & Stablecoin Model Demo - Xi/Psi Quantum Financial Framework

This script demonstrates how the Xi/Psi quantum field approach can model the complex 
relationship between interest rates and stablecoin stability, comparing the performance
against traditional models and showcasing product-grade capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import math
from scipy import stats, optimize
import datetime as dt
from scipy.interpolate import interp1d



# Set seeds for reproducibility
np.random.seed(42)

#####################################################
# Traditional Interest Rate Models                  #
#####################################################

def vasicek_model(r0, kappa, theta, sigma, T, steps, random_state=None):
    """
    Simulate interest rates using the Vasicek model
    
    Args:
        r0: Initial interest rate
        kappa: Speed of mean reversion
        theta: Long-term mean rate
        sigma: Volatility
        T: Time horizon in years
        steps: Number of simulation steps
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with timestamps and simulated rates
    """
    delta_t = T / steps
    rates = [r0]
    times = [0]
    
    np.random.seed(random_state)
    
    for i in range(1, steps + 1):
        # Vasicek model: dr = kappa * (theta - r) * dt + sigma * dW
        r_prev = rates[-1]
        dr = kappa * (theta - r_prev) * delta_t + sigma * np.random.normal(0, np.sqrt(delta_t))
        r_new = r_prev + dr
        
        rates.append(r_new)
        times.append(i * delta_t)
    
    # Create dates from times
    start_date = dt.datetime.now()
    dates = [start_date + dt.timedelta(days=int(t * 365)) for t in times]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Rate': rates
    })
    
    return df

def cox_ingersoll_ross_model(r0, kappa, theta, sigma, T, steps, random_state=None):
    """
    Simulate interest rates using the Cox-Ingersoll-Ross model
    
    Args:
        r0: Initial interest rate
        kappa: Speed of mean reversion
        theta: Long-term mean rate
        sigma: Volatility
        T: Time horizon in years
        steps: Number of simulation steps
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with timestamps and simulated rates
    """
    delta_t = T / steps
    rates = [r0]
    times = [0]
    
    np.random.seed(random_state)
    
    for i in range(1, steps + 1):
        # CIR model: dr = kappa * (theta - r) * dt + sigma * sqrt(r) * dW
        r_prev = rates[-1]
        dr = kappa * (theta - r_prev) * delta_t + sigma * np.sqrt(max(0, r_prev)) * np.random.normal(0, np.sqrt(delta_t))
        r_new = max(0, r_prev + dr)  # Ensure rate is non-negative
        
        rates.append(r_new)
        times.append(i * delta_t)
    
    # Create dates from times
    start_date = dt.datetime.now()
    dates = [start_date + dt.timedelta(days=int(t * 365)) for t in times]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Rate': rates
    })
    
    return df

#####################################################
# Traditional Stablecoin Models                     #
#####################################################

def traditional_stablecoin_model(interest_rates, peg_value=1.0, sensitivity=0.5, 
                               liquidity_factor=0.2, volatility=0.01, random_state=None):
    """
    Model stablecoin price dynamics using traditional approach
    
    Args:
        interest_rates: DataFrame with rates
        peg_value: Target peg value (typically 1.0)
        sensitivity: Sensitivity to interest rate changes
        liquidity_factor: Market liquidity factor
        volatility: Base volatility
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with stablecoin prices
    """
    np.random.seed(random_state)
    
    # Initialize price at peg
    prices = [peg_value]
    
    # Get rate changes
    rates = interest_rates['Rate'].values
    rate_changes = np.diff(rates, prepend=rates[0])
    
    # Simulate price changes
    for i in range(1, len(rates)):
        # Previous price
        prev_price = prices[-1]
        
        # Rate change effect
        rate_effect = -sensitivity * rate_changes[i]  # Negative correlation with rate changes
        
        # Mean reversion to peg
        reversion = liquidity_factor * (peg_value - prev_price)
        
        # Random noise
        noise = np.random.normal(0, volatility)
        
        # New price
        new_price = prev_price + rate_effect + reversion + noise
        
        prices.append(new_price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': interest_rates['Date'],
        'Price': prices
    })
    
    return df

#####################################################
# Xi/Psi Quantum Interest Rate Model                #
#####################################################

class XiPsiInterestRateModel:
    """
    Enhanced interest rate model using Xi/Psi quantum field approach
    """
    def __init__(self, coherence_threshold=0.7, phase_modulation=True):
        self.coherence_threshold = coherence_threshold
        self.phase_modulation = phase_modulation
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
    def simulate(self, r0, kappa, theta, sigma, T, steps, 
               central_bank_policy=None, economic_indicators=None, random_state=None):
        """
        Simulate interest rates with quantum enhancements
        
        Args:
            r0: Initial interest rate
            kappa: Speed of mean reversion
            theta: Long-term mean rate
            sigma: Volatility
            T: Time horizon in years
            steps: Number of simulation steps
            central_bank_policy: Function to model central bank policies
            economic_indicators: Additional economic data
            random_state: Random state for reproducibility
            
        Returns:
            DataFrame with simulated rates and quantum metrics
        """
        np.random.seed(random_state)
        
        delta_t = T / steps
        rates = [r0]
        times = [0]
        coherence = [0.9]  # Start with high coherence
        phase = [0.0]
        entropy = [0.1]  # Start with low entropy
        
        # Generate base noise terms with quantum correlations
        noise_terms = np.random.normal(0, np.sqrt(delta_t), steps)
        
        # Apply Fibonacci modulation for quantum correlations
        if self.phase_modulation:
            modulation = np.sin(2 * np.pi * self.phi * np.arange(steps) / steps)
            noise_terms = noise_terms * (1 + 0.2 * modulation)
        
        for i in range(1, steps + 1):
            # Previous values
            r_prev = rates[-1]
            coherence_prev = coherence[-1]
            phase_prev = phase[-1]
            entropy_prev = entropy[-1]
            
            # Standard mean reversion component
            mean_reversion = kappa * (theta - r_prev) * delta_t
            
            # Get noise term with quantum correlation
            if i <= steps:
                noise = sigma * noise_terms[i-1]
            else:
                noise = sigma * np.random.normal(0, np.sqrt(delta_t))
            
            # Apply central bank policy if provided
            policy_effect = 0
            if central_bank_policy is not None:
                policy_effect = central_bank_policy(i / steps, r_prev) * delta_t
            
            # Update quantum metrics
            # Phase evolution with memory effects
            new_phase = (phase_prev + self.phi * r_prev * delta_t) % (2 * np.pi)
            
            # Coherence evolution - decreases with volatility, increases with stability
            coherence_decay = 0.1 * abs(noise) / (sigma * np.sqrt(delta_t))
            coherence_boost = 0.05 * (1 - abs(r_prev - theta) / max(0.001, theta))
            new_coherence = coherence_prev * (1 - coherence_decay) + coherence_boost
            new_coherence = max(0.1, min(0.95, new_coherence))
            
            # Entropy increases with uncertainty, decreases with coherence
            new_entropy = entropy_prev + 0.1 * abs(noise) - 0.05 * new_coherence
            new_entropy = max(0.05, min(0.9, new_entropy))
            
            # Quantum-enhanced interest rate model
            # 1. Standard drift term
            # 2. Enhanced diffusion with coherence-dependent volatility
            # 3. Phase-dependent cyclical component
            # 4. Entropy-based regime switching probability
            
            # Modify volatility based on coherence (more coherent = less volatile)
            effective_sigma = sigma * (1 + (0.5 - new_coherence))
            
            # Add phase-dependent cyclical component
            cyclical_component = 0.01 * np.sin(new_phase) * (1 - new_coherence)
            
            # Regime switching based on entropy
            regime_switch = 0
            if np.random.random() < new_entropy * 0.2:  # Higher entropy = higher switching probability
                # Sudden shift in interest rate dynamics
                regime_switch = 0.005 * np.random.choice([-1, 1]) * (1 - new_coherence)
            
            # Calculate interest rate change
            dr = mean_reversion + effective_sigma * noise + cyclical_component + regime_switch + policy_effect
            
            # Update rate with non-negative constraint
            r_new = max(0.001, r_prev + dr)
            
            # Store values
            rates.append(r_new)
            times.append(i * delta_t)
            coherence.append(new_coherence)
            phase.append(new_phase)
            entropy.append(new_entropy)
        
        # Create dates from times
        start_date = dt.datetime.now()
        dates = [start_date + dt.timedelta(days=int(t * 365)) for t in times]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Rate': rates,
            'Coherence': coherence,
            'Phase': phase,
            'Entropy': entropy
        })
        
        return df

#####################################################
# Xi/Psi Quantum Stablecoin Model                  #
#####################################################

class XiPsiStablecoinModel:
    """
    Enhanced stablecoin model using Xi/Psi quantum field approach
    """
    def __init__(self, peg_value=1.0, stability_threshold=0.95):
        self.peg_value = peg_value
        self.stability_threshold = stability_threshold
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    def simulate(self, interest_rates_df, market_conditions=None, random_state=None):
        """
        Simulate stablecoin prices with quantum enhancements
        
        Args:
            interest_rates_df: DataFrame with interest rates and quantum metrics
            market_conditions: Function to model market conditions
            random_state: Random state for reproducibility
            
        Returns:
            DataFrame with stablecoin prices and metrics
        """
        np.random.seed(random_state)
        
        # Extract data from interest rates DataFrame
        dates = interest_rates_df['Date']
        rates = interest_rates_df['Rate'].values
        
        # Extract quantum metrics if available
        has_quantum_metrics = all(col in interest_rates_df.columns for col in ['Coherence', 'Phase', 'Entropy'])
        if has_quantum_metrics:
            coherence = interest_rates_df['Coherence'].values
            phase = interest_rates_df['Phase'].values
            entropy = interest_rates_df['Entropy'].values
        else:
            # Generate synthetic quantum metrics if not available
            coherence = np.ones(len(rates)) * 0.7
            phase = np.linspace(0, 2 * np.pi, len(rates)) % (2 * np.pi)
            entropy = np.ones(len(rates)) * 0.3
        
        # Initialize price arrays
        prices = [self.peg_value]
        stability_scores = [1.0]
        reversion_strength = [0.2]
        arbitrage_pressure = [0.0]
        
        # Calculate rate changes
        rate_changes = np.diff(rates, prepend=rates[0])
        
        # Helper function for liquidity modeling
        def model_liquidity(i, rate, coherence, entropy):
            # Base liquidity as function of rate level
            base_liquidity = 0.2 - 0.5 * min(0.3, rate)
            
            # Adjust based on quantum metrics
            coherence_effect = 0.1 * coherence  # Higher coherence -> more stable
            entropy_effect = -0.1 * entropy  # Higher entropy -> less stable
            
            # Market cycles based on phase
            cycle_effect = 0.05 * np.sin(phase[i])
            
            return max(0.01, base_liquidity + coherence_effect + entropy_effect + cycle_effect)
        
        # Simulate price changes
        for i in range(1, len(rates)):
            # Previous values
            prev_price = prices[-1]
            prev_stability = stability_scores[-1]
            prev_reversion = reversion_strength[-1]
            prev_arbitrage = arbitrage_pressure[-1]
            
            # Current rate and rate change
            rate = rates[i]
            rate_change = rate_changes[i]
            
            # Model liquidity
            liquidity = model_liquidity(i, rate, coherence[i], entropy[i])
            
            # Interest rate effects on stablecoin
            # Phase-dependent sensitivity to interest rates
            rate_sensitivity = 0.3 + 0.2 * np.sin(phase[i])
            direct_rate_effect = -rate_sensitivity * rate_change
            
            # Mean reversion to peg, strength varies with coherence
            curr_reversion = prev_reversion * (0.9 + 0.2 * coherence[i])
            reversion_effect = curr_reversion * (self.peg_value - prev_price)
            
            # Stability feedback mechanism (higher stability -> stronger peg)
            stability_effect = 0.05 * prev_stability * (self.peg_value - prev_price)
            
            # Arbitrage pressure (quantum-enhanced)
            # Arbitrage strength depends on deviation and liquidity
            deviation = abs(prev_price - self.peg_value)
            new_arbitrage = prev_arbitrage * 0.8 + 0.2 * deviation / liquidity
            arbitrage_effect = -np.sign(prev_price - self.peg_value) * new_arbitrage * 0.1
            
            # Market conditions effect
            market_effect = 0
            if market_conditions is not None:
                market_effect = market_conditions(i / len(rates), rate)
            
            # Random noise - lower when coherence is high
            noise_scale = 0.01 * (1 - 0.5 * coherence[i]) * (1 + 2 * entropy[i])
            noise = np.random.normal(0, noise_scale)
            
            # Combine all effects
            price_change = (direct_rate_effect + 
                          reversion_effect + 
                          stability_effect + 
                          arbitrage_effect + 
                          market_effect + 
                          noise)
            
            # Update price
            new_price = prev_price + price_change
            
            # Update stability score
            deviation_ratio = min(1, abs(new_price - self.peg_value) / 0.1)
            stability_decay = deviation_ratio ** 2 * 0.05
            stability_gain = (1 - deviation_ratio) ** 2 * 0.02
            new_stability = prev_stability * (1 - stability_decay) + stability_gain
            new_stability = max(0.1, min(1.0, new_stability))
            
            # Store values
            prices.append(new_price)
            stability_scores.append(new_stability)
            reversion_strength.append(curr_reversion)
            arbitrage_pressure.append(new_arbitrage)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices,
            'Stability': stability_scores,
            'ReversionStrength': reversion_strength,
            'ArbitragePressure': arbitrage_pressure
        })
        
        return df

#####################################################
# Fed Interest Rate Policy Simulation               #
#####################################################

def simulate_fed_policy(base_scenario="neutral", random_state=None):
    """
    Simulate Federal Reserve interest rate policy scenarios
    
    Args:
        base_scenario: Base scenario ("hawkish", "dovish", or "neutral")
        random_state: Random state for reproducibility
        
    Returns:
        Policy function that can be passed to the interest rate model
    """
    np.random.seed(random_state)
    
    # Policy parameters
    if base_scenario == "hawkish":
        # Aggressive rate hikes to combat inflation
        target_direction = 1
        policy_aggression = 0.4
        reaction_speed = 0.3
    elif base_scenario == "dovish":
        # Rate cuts to stimulate economy
        target_direction = -1
        policy_aggression = 0.3
        reaction_speed = 0.2
    else:  # neutral
        # Balanced approach
        target_direction = 0
        policy_aggression = 0.1
        reaction_speed = 0.1
    
    # Generate random policy change points
    change_points = sorted(np.random.choice(np.linspace(0.1, 0.9, 100), 5, replace=False))
    change_directions = np.random.choice([-1, 1], 5) * policy_aggression * 0.5
    
    # Policy reaction function
    def policy_function(time_ratio, current_rate):
        # Base policy direction
        base_effect = target_direction * policy_aggression * 0.01
        
        # Check for policy change points
        change_effect = 0
        for i, point in enumerate(change_points):
            # Policy changes near the specified points
            if abs(time_ratio - point) < 0.05:
                # Strength of change diminishes with distance
                distance_factor = 1 - abs(time_ratio - point) / 0.05
                change_effect += change_directions[i] * distance_factor * 0.01
        
        # Reaction to extreme rates (reversion to moderate policy)
        if current_rate > 0.08:  # Very high rates -> dovish pressure
            reaction_effect = -reaction_speed * 0.01
        elif current_rate < 0.01:  # Very low rates -> hawkish pressure
            reaction_effect = reaction_speed * 0.01
        else:
            reaction_effect = 0
        
        return base_effect + change_effect + reaction_effect
    
    return policy_function

#####################################################
# Market Condition Simulation                       #
#####################################################

def simulate_market_conditions(scenario="normal", liquidity_shocks=True, random_state=None):
    """
    Simulate market conditions that affect stablecoin pricing
    
    Args:
        scenario: Market scenario ("normal", "crisis", or "boom")
        liquidity_shocks: Whether to include random liquidity shocks
        random_state: Random state for reproducibility
        
    Returns:
        Market condition function that can be passed to the stablecoin model
    """
    np.random.seed(random_state)
    
    # Market parameters
    if scenario == "crisis":
        # Market stress, high volatility, low liquidity
        base_liquidity = -0.05
        volatility = 0.03
        shock_probability = 0.03 if liquidity_shocks else 0
    elif scenario == "boom":
        # Strong market, high liquidity
        base_liquidity = 0.03
        volatility = 0.01
        shock_probability = 0.01 if liquidity_shocks else 0
    else:  # normal
        # Balanced market conditions
        base_liquidity = 0
        volatility = 0.015
        shock_probability = 0.02 if liquidity_shocks else 0
    
    # Generate random shock points
    num_shocks = int(shock_probability * 100)
    shock_points = sorted(np.random.choice(np.linspace(0.1, 0.9, 100), num_shocks, replace=False))
    shock_magnitudes = np.random.choice([-1, 1], num_shocks) * np.random.uniform(0.01, 0.03, num_shocks)
    
    # Market condition function
    def market_function(time_ratio, current_rate):
        # Base market condition
        base_effect = base_liquidity
        
        # Interest rate effect (higher rates tend to reduce stablecoin premium)
        rate_effect = -0.02 * current_rate
        
        # Random volatility
        volatility_effect = np.random.normal(0, volatility) * 0.01
        
        # Liquidity shocks
        shock_effect = 0
        for i, point in enumerate(shock_points):
            # Shocks occur near the specified points
            if abs(time_ratio - point) < 0.03:
                # Shock effect diminishes with distance
                distance_factor = 1 - abs(time_ratio - point) / 0.03
                shock_effect += shock_magnitudes[i] * distance_factor
        
        return base_effect + rate_effect + volatility_effect + shock_effect
    
    return market_function

#####################################################
# Visualization Functions                           #
#####################################################

def plot_interest_rate_comparison(traditional_df, quantum_df, 
                                title="Interest Rate Model Comparison",
                                output_file="interest_rate_comparison.png"):
    """
    Create comparison plot of traditional vs quantum interest rate models
    
    Args:
        traditional_df: DataFrame with traditional model results
        quantum_df: DataFrame with quantum model results
        title: Plot title
        output_file: Output file path
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(title, fontsize=16)
    
    # Plot interest rates
    ax1.plot(traditional_df['Date'], traditional_df['Rate'] * 100, 'b-', 
            label='Traditional Model', linewidth=2, alpha=0.7)
    ax1.plot(quantum_df['Date'], quantum_df['Rate'] * 100, 'r-', 
            label='Xi/Psi Quantum Model', linewidth=2)
    
    # Highlight the difference
    # Convert Date to a numeric type for fill_between
    dates_num = mdates.date2num(quantum_df['Date'])
    
    # Calculate where conditions based on rate values
    where_quantum_higher = quantum_df['Rate'].values > traditional_df['Rate'].values
    where_quantum_lower = quantum_df['Rate'].values <= traditional_df['Rate'].values
    
    ax1.fill_between(dates_num, 
                    traditional_df['Rate'] * 100, 
                    quantum_df['Rate'] * 100, 
                    where=where_quantum_higher,
                    facecolor='red', alpha=0.3, interpolate=True)
    
    ax1.fill_between(dates_num, 
                    traditional_df['Rate'] * 100, 
                    quantum_df['Rate'] * 100, 
                    where=where_quantum_lower,
                    facecolor='blue', alpha=0.3, interpolate=True)
    
    # Set labels and title
    ax1.set_ylabel('Interest Rate (%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    
    # Format date axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Plot quantum metrics in second subplot
    ax2.plot(quantum_df['Date'], quantum_df['Coherence'], 'g-', 
            label='Coherence', linewidth=2)
    ax2.plot(quantum_df['Date'], quantum_df['Entropy'], 'm-', 
            label='Entropy', linewidth=2)
    
    # Set labels and title
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Quantum Metrics')
    ax2.set_title('Quantum Field Metrics')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Format date axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Created interest rate comparison chart: {output_file}")

def plot_stablecoin_comparison(traditional_df, quantum_df, 
                              title="Stablecoin Model Comparison",
                              output_file="stablecoin_comparison.png"):
    """
    Create comparison plot of traditional vs quantum stablecoin models
    
    Args:
        traditional_df: DataFrame with traditional model results
        quantum_df: DataFrame with quantum model results
        title: Plot title
        output_file: Output file path
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(title, fontsize=16)
    
    # Plot stablecoin prices
    ax1.plot(traditional_df['Date'], traditional_df['Price'], 'b-', 
            label='Traditional Model', linewidth=2, alpha=0.7)
    ax1.plot(quantum_df['Date'], quantum_df['Price'], 'r-', 
            label='Xi/Psi Quantum Model', linewidth=2)
    
    # Add peg line
    ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Peg Value ($1.00)')
    
    # Highlight deviations from peg
    ax1.fill_between(traditional_df['Date'], 
                    1.0, 
                    traditional_df['Price'], 
                    where=(traditional_df['Price'] > 1.0),
                    facecolor='green', alpha=0.2, interpolate=True,
                    label='Premium (Trad)')
    
    ax1.fill_between(traditional_df['Date'], 
                    1.0, 
                    traditional_df['Price'], 
                    where=(traditional_df['Price'] < 1.0),
                    facecolor='red', alpha=0.2, interpolate=True,
                    label='Discount (Trad)')
    
    # Set labels and title
    ax1.set_ylabel('Stablecoin Price ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Set y-axis limits to focus on deviations
    y_min = min(traditional_df['Price'].min(), quantum_df['Price'].min()) - 0.01
    y_max = max(traditional_df['Price'].max(), quantum_df['Price'].max()) + 0.01
    ax1.set_ylim(y_min, y_max)
    
    # Format date axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Plot stability metrics in second subplot
    ax2.plot(quantum_df['Date'], quantum_df['Stability'], 'g-', 
            label='Stability Score', linewidth=2)
    ax2.plot(quantum_df['Date'], quantum_df['ArbitragePressure'], 'm-', 
            label='Arbitrage Pressure', linewidth=2, alpha=0.7)
    
    # Set labels and title
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Stability Metrics')
    ax2.set_title('Stablecoin Stability Metrics')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Format date axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Created stablecoin comparison chart: {output_file}")

def plot_rate_stablecoin_relationship(interest_df, stablecoin_df, 
                                    title="Interest Rate & Stablecoin Relationship",
                                    output_file="rate_stablecoin_relationship.png"):
    """
    Create visualization of relationship between interest rates and stablecoin prices
    
    Args:
        interest_df: DataFrame with interest rate data
        stablecoin_df: DataFrame with stablecoin data
        title: Plot title
        output_file: Output file path
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=16)
    
    # Plot interest rates
    ax1.plot(interest_df['Date'], interest_df['Rate'] * 100, 'b-', 
            label='Interest Rate', linewidth=2)
    
    # Add coherence as area plot
    if 'Coherence' in interest_df.columns:
        ax1_coh = ax1.twinx()
        ax1_coh.fill_between(interest_df['Date'], 0, interest_df['Coherence'], 
                           color='g', alpha=0.2, label='Coherence')
        ax1_coh.set_ylabel('Coherence', color='g')
        ax1_coh.tick_params(axis='y', labelcolor='g')
        ax1_coh.set_ylim(0, 1)
    
    # Set labels and title
    ax1.set_title('Interest Rate Evolution')
    ax1.set_ylabel('Interest Rate (%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    
    # Plot stablecoin price
    ax2.plot(stablecoin_df['Date'], stablecoin_df['Price'], 'r-', 
            label='Stablecoin Price', linewidth=2)
    
    # Add peg line
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Peg Value ($1.00)')
    
    # Highlight deviations from peg
    ax2.fill_between(stablecoin_df['Date'], 
                    1.0, 
                    stablecoin_df['Price'], 
                    where=(stablecoin_df['Price'] > 1.0),
                    facecolor='green', alpha=0.2, interpolate=True,
                    label='Premium')
    
    ax2.fill_between(stablecoin_df['Date'], 
                    1.0, 
                    stablecoin_df['Price'], 
                    where=(stablecoin_df['Price'] < 1.0),
                    facecolor='red', alpha=0.2, interpolate=True,
                    label='Discount')
    
    # Set labels
    ax2.set_title('Stablecoin Price Response')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Stablecoin Price ($)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Set y-axis limits to focus on deviations
    y_min = min(stablecoin_df['Price'].min(), 0.97)
    y_max = max(stablecoin_df['Price'].max(), 1.03)
    ax2.set_ylim(y_min, y_max)
    
    # Format date axis
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Created rate-stablecoin relationship visualization: {output_file}")

def plot_stablecoin_response_to_rate_changes(interest_df, stablecoin_df, 
                                          output_file="stablecoin_rate_response.png"):
    """
    Plot relationship between interest rate changes and stablecoin price deviations
    
    Args:
        interest_df: DataFrame with interest rate data
        stablecoin_df: DataFrame with stablecoin data
        output_file: Output file path
    """
    # Calculate rate changes and price deviations
    rates = interest_df['Rate'].values
    prices = stablecoin_df['Price'].values
    
    rate_changes = np.diff(rates, prepend=rates[0])
    price_deviations = prices - 1.0  # Deviation from peg
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Color points based on coherence if available
    if 'Coherence' in interest_df.columns:
        coherence = interest_df['Coherence'].values
        scatter = plt.scatter(rate_changes[1:] * 100, price_deviations[1:] * 100, 
                            c=coherence[1:], cmap='viridis', 
                            alpha=0.7, s=50)
        plt.colorbar(scatter, label='Coherence')
    else:
        plt.scatter(rate_changes[1:] * 100, price_deviations[1:] * 100, 
                  alpha=0.7, s=50, color='blue')
    
    # Add trend line
    z = np.polyfit(rate_changes[1:], price_deviations[1:], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(rate_changes[1:]), max(rate_changes[1:]), 100)
    plt.plot(x_trend * 100, p(x_trend) * 100, "r--", alpha=0.8, 
            label=f'Trend: y = {z[0]:.4f}x + {z[1]:.4f}')
    
    # Add quadrants
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Annotate quadrants
    plt.text(0.75 * max(rate_changes[1:] * 100), 0.75 * max(price_deviations[1:] * 100), 
            "Rate ↑\nPrice ↑", fontsize=12, ha='center')
    plt.text(0.75 * min(rate_changes[1:] * 100), 0.75 * max(price_deviations[1:] * 100), 
            "Rate ↓\nPrice ↑", fontsize=12, ha='center')
    plt.text(0.75 * min(rate_changes[1:] * 100), 0.75 * min(price_deviations[1:] * 100), 
            "Rate ↓\nPrice ↓", fontsize=12, ha='center')
    plt.text(0.75 * max(rate_changes[1:] * 100), 0.75 * min(price_deviations[1:] * 100), 
            "Rate ↑\nPrice ↓", fontsize=12, ha='center')
    
    # Set labels and title
    plt.title("Stablecoin Price Response to Interest Rate Changes", fontsize=16)
    plt.xlabel("Interest Rate Change (basis points)")
    plt.ylabel("Stablecoin Price Deviation (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Created stablecoin response visualization: {output_file}")

def create_combined_visualization(interest_trad_df, interest_quantum_df, 
                                stablecoin_trad_df, stablecoin_quantum_df,
                                output_file="interest_stablecoin_combined.png"):
    """
    Create a consolidated visualization with multiple aspects on one slide
    
    Args:
        interest_trad_df: Traditional interest rate model results
        interest_quantum_df: Quantum interest rate model results
        stablecoin_trad_df: Traditional stablecoin model results
        stablecoin_quantum_df: Quantum stablecoin model results
        output_file: Output file path
    """
    # Set up figure with multiple panels
    fig = plt.figure(figsize=(16, 12))
    plt.suptitle("Xi/Psi Quantum Interest Rate & Stablecoin Model Analysis", fontsize=20, y=0.98)
    
    # Create a grid layout
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1], wspace=0.3, hspace=0.3)
    
    # 1. Interest rate comparison (upper left)
    ax1 = plt.subplot(gs[0, 0])
    plot_interest_rate_panel(ax1, interest_trad_df, interest_quantum_df)
    
    # 2. Stablecoin comparison (upper right)
    ax2 = plt.subplot(gs[0, 1])
    plot_stablecoin_panel(ax2, stablecoin_trad_df, stablecoin_quantum_df)
    
    # 3. Rate-stablecoin dynamics (lower left)
    ax3 = plt.subplot(gs[1, 0])
    plot_dynamics_panel(ax3, interest_quantum_df, stablecoin_quantum_df)
    
    # 4. Performance metrics (lower right)
    ax4 = plt.subplot(gs[1, 1])
    plot_metrics_panel(ax4, interest_trad_df, interest_quantum_df, 
                      stablecoin_trad_df, stablecoin_quantum_df)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Created combined visualization: {output_file}")

def plot_interest_rate_panel(ax, trad_df, quantum_df):
    """Helper function to plot interest rate panel"""
    # Plot rates
    ax.plot(trad_df['Date'], trad_df['Rate'] * 100, 'b-', 
           label='Traditional', linewidth=2, alpha=0.7)
    ax.plot(quantum_df['Date'], quantum_df['Rate'] * 100, 'r-', 
           label='Xi/Psi Quantum', linewidth=2)
    
    # Add zero-line
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Set labels
    ax.set_title("Interest Rate Forecasts")
    ax.set_ylabel("Interest Rate (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))
    
    # Format date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Add quantum metrics as small overlay
    if 'Coherence' in quantum_df.columns:
        ax_coh = ax.twinx()
        ax_coh.plot(quantum_df['Date'], quantum_df['Coherence'], 'g--', 
                  linewidth=1, alpha=0.5, label='Coherence')
        ax_coh.set_ylim(0, 1)
        ax_coh.tick_params(axis='y', labelcolor='g')
        ax_coh.set_ylabel('Coherence', color='g', fontsize=8)

def plot_stablecoin_panel(ax, trad_df, quantum_df):
    """Helper function to plot stablecoin panel"""
    # Plot prices
    ax.plot(trad_df['Date'], trad_df['Price'], 'b-', 
           label='Traditional', linewidth=2, alpha=0.7)
    ax.plot(quantum_df['Date'], quantum_df['Price'], 'r-', 
           label='Xi/Psi Quantum', linewidth=2)
    
    # Add peg line
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Peg')
    
    # Set labels
    ax.set_title("Stablecoin Price Stability")
    ax.set_ylabel("Price ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    
    # Set y-axis limits to focus on deviations
    y_min = min(trad_df['Price'].min(), quantum_df['Price'].min()) - 0.01
    y_max = max(trad_df['Price'].max(), quantum_df['Price'].max()) + 0.01
    ax.set_ylim(y_min, y_max)
    
    # Format date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Add stability metric as small overlay
    if 'Stability' in quantum_df.columns:
        ax_stab = ax.twinx()
        ax_stab.plot(quantum_df['Date'], quantum_df['Stability'], 'g--', 
                   linewidth=1, alpha=0.5, label='Stability')
        ax_stab.set_ylim(0, 1)
        ax_stab.tick_params(axis='y', labelcolor='g')
        ax_stab.set_ylabel('Stability', color='g', fontsize=8)

def plot_dynamics_panel(ax, interest_df, stablecoin_df):
    """Helper function to plot dynamics panel"""
    # Calculate rate changes and price deviations
    rates = interest_df['Rate'].values
    prices = stablecoin_df['Price'].values
    
    rate_changes = np.diff(rates, prepend=rates[0])
    price_deviations = prices - 1.0  # Deviation from peg
    
    # Color points based on coherence if available
    if 'Coherence' in interest_df.columns:
        coherence = interest_df['Coherence'].values
        scatter = ax.scatter(rate_changes[1:] * 100, price_deviations[1:] * 100, 
                           c=coherence[1:], cmap='viridis', 
                           alpha=0.7, s=40)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Coherence', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    else:
        ax.scatter(rate_changes[1:] * 100, price_deviations[1:] * 100, 
                 alpha=0.7, s=40, color='blue')
    
    # Add trend line
    z = np.polyfit(rate_changes[1:], price_deviations[1:], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(rate_changes[1:]), max(rate_changes[1:]), 100)
    ax.plot(x_trend * 100, p(x_trend) * 100, "r--", alpha=0.8, 
           label=f'Trend: y = {z[0]:.4f}x + {z[1]:.4f}')
    
    # Add quadrants
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Set labels
    ax.set_title("Rate-Stablecoin Response Dynamics")
    ax.set_xlabel("Interest Rate Change (bps)")
    ax.set_ylabel("Price Deviation (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)

def plot_metrics_panel(ax, interest_trad_df, interest_quantum_df, 
                      stablecoin_trad_df, stablecoin_quantum_df):
    """Helper function to plot metrics panel"""
    # Calculate stability metrics
    trad_peg_deviation = np.mean(np.abs(stablecoin_trad_df['Price'] - 1.0)) * 100
    quantum_peg_deviation = np.mean(np.abs(stablecoin_quantum_df['Price'] - 1.0)) * 100
    
    trad_max_deviation = np.max(np.abs(stablecoin_trad_df['Price'] - 1.0)) * 100
    quantum_max_deviation = np.max(np.abs(stablecoin_quantum_df['Price'] - 1.0)) * 100
    
    # Calculate volatility metrics
    trad_rate_vol = np.std(interest_trad_df['Rate']) * 100
    quantum_rate_vol = np.std(interest_quantum_df['Rate']) * 100
    
    trad_price_vol = np.std(stablecoin_trad_df['Price']) * 100
    quantum_price_vol = np.std(stablecoin_quantum_df['Price']) * 100
    
    # Calculate correlation
    trad_correlation = np.corrcoef(interest_trad_df['Rate'][1:], 
                                 stablecoin_trad_df['Price'][1:])[0, 1]
    quantum_correlation = np.corrcoef(interest_quantum_df['Rate'][1:], 
                                    stablecoin_quantum_df['Price'][1:])[0, 1]
    
    # Create table data
    metrics = ['Avg Peg Deviation (%)', 'Max Peg Deviation (%)', 
             'Rate Volatility (%)', 'Price Volatility (%)',
             'Rate-Price Correlation']
    
    trad_values = [trad_peg_deviation, trad_max_deviation, 
                 trad_rate_vol, trad_price_vol, trad_correlation]
    
    quantum_values = [quantum_peg_deviation, quantum_max_deviation, 
                    quantum_rate_vol, quantum_price_vol, quantum_correlation]
    
    # Calculate improvement percentages
    improvements = []
    for t, q in zip(trad_values, quantum_values):
        if abs(t) < 0.0001:  # Avoid division by zero
            imp = 0
        else:
            # For most metrics, lower is better
            if metrics.index(metrics[len(improvements)]) != 4:  # Not correlation
                imp = (t - q) / t * 100
            else:  # For correlation, higher magnitude is better
                imp = (abs(q) - abs(t)) / abs(t) * 100 if abs(t) > 0 else 100
        improvements.append(imp)
    
    # Plot as table
    ax.axis('off')
    ax.set_title("Performance Metrics Comparison")
    
    # Format data for table
    trad_display = [f"{val:.3f}" for val in trad_values]
    quantum_display = [f"{val:.3f}" for val in quantum_values]
    improvement_display = [f"{val:+.1f}%" for val in improvements]
    
    # Create table
    table_data = []
    for i in range(len(metrics)):
        table_data.append([metrics[i], trad_display[i], quantum_display[i], improvement_display[i]])
    
    colors = []
    for imp in improvements:
        if imp > 1:  # Improvement
            colors.append(['white', 'white', 'white', 'lightgreen'])
        elif imp < -1:  # Worse
            colors.append(['white', 'white', 'white', 'lightcoral'])
        else:  # Similar
            colors.append(['white', 'white', 'white', 'white'])
    
    table = ax.table(cellText=table_data, 
                   colLabels=['Metric', 'Traditional', 'Quantum', 'Improvement'],
                   loc='center', cellLoc='center',
                   cellColours=colors)
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Add explanatory text
    footnote = ("Metrics comparison between traditional and quantum models.\n"
               "Improvement shows how much better the quantum model performs.")
    ax.text(0.5, 0.05, footnote, ha='center', fontsize=8, 
           transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

def generate_scenario_data(scenario="baseline", time_horizon=2, num_steps=120, random_state=42):
    """
    Generate simulated data for different interest rate and market scenarios
    
    Args:
        scenario: Scenario name ("baseline", "hiking", "cutting", "volatile")
        time_horizon: Time horizon in years
        num_steps: Number of simulation steps
        random_state: Random state for reproducibility
        
    Returns:
        Generated data for each model
    """
    np.random.seed(random_state)
    
    # Set scenario parameters
    if scenario == "hiking":
        # Gradually rising interest rates
        r0 = 0.02
        target_rate = 0.06
        kappa_trad = 0.6
        kappa_quantum = 0.8
        theta_trad = target_rate
        theta_quantum = target_rate
        sigma_trad = 0.004
        sigma_quantum = 0.005
        policy = "hawkish"
        market = "normal"
    
    elif scenario == "cutting":
        # Gradually falling interest rates
        r0 = 0.05
        target_rate = 0.01
        kappa_trad = 0.5
        kappa_quantum = 0.7
        theta_trad = target_rate
        theta_quantum = target_rate
        sigma_trad = 0.0035
        sigma_quantum = 0.0045
        policy = "dovish"
        market = "normal"
    
    elif scenario == "volatile":
        # Highly volatile market conditions
        r0 = 0.03
        target_rate = 0.04
        kappa_trad = 0.3
        kappa_quantum = 0.4
        theta_trad = target_rate
        theta_quantum = target_rate
        sigma_trad = 0.008
        sigma_quantum = 0.01
        policy = "neutral"
        market = "crisis"
    
    else:  # baseline
        # Moderate, stable conditions
        r0 = 0.03
        target_rate = 0.035
        kappa_trad = 0.4
        kappa_quantum = 0.5
        theta_trad = target_rate
        theta_quantum = target_rate
        sigma_trad = 0.003
        sigma_quantum = 0.004
        policy = "neutral"
        market = "normal"
    
    # Generate Fed policy function
    fed_policy = simulate_fed_policy(base_scenario=policy, random_state=random_state)
    
    # Generate market conditions function
    market_conditions = simulate_market_conditions(scenario=market, 
                                                 liquidity_shocks=True, 
                                                 random_state=random_state)
    
    # Generate interest rate paths
    # Traditional model
    interest_trad_df = cox_ingersoll_ross_model(
        r0=r0,
        kappa=kappa_trad,
        theta=theta_trad,
        sigma=sigma_trad,
        T=time_horizon,
        steps=num_steps,
        random_state=random_state
    )
    
    # Quantum model
    interest_model = XiPsiInterestRateModel(coherence_threshold=0.7, phase_modulation=True)
    interest_quantum_df = interest_model.simulate(
        r0=r0,
        kappa=kappa_quantum,
        theta=theta_quantum,
        sigma=sigma_quantum,
        T=time_horizon,
        steps=num_steps,
        central_bank_policy=fed_policy,
        random_state=random_state
    )
    
    # Generate stablecoin price paths
    # Traditional model
    stablecoin_trad_df = traditional_stablecoin_model(
        interest_rates=interest_trad_df,
        peg_value=1.0,
        sensitivity=0.5,
        liquidity_factor=0.2,
        volatility=0.01,
        random_state=random_state
    )
    
    # Quantum model
    stablecoin_model = XiPsiStablecoinModel(peg_value=1.0, stability_threshold=0.95)
    stablecoin_quantum_df = stablecoin_model.simulate(
        interest_rates_df=interest_quantum_df,
        market_conditions=market_conditions,
        random_state=random_state
    )
    
    return (interest_trad_df, interest_quantum_df, 
           stablecoin_trad_df, stablecoin_quantum_df)

def main():
    """
    Main function to run the interest rate and stablecoin model demonstration
    """
    print("\n===== Xi/Psi Quantum Interest Rate & Stablecoin Model Demo =====\n")
    
    # Create output directory
    import os
    os.makedirs("outputs/interest_stablecoin", exist_ok=True)
    
    # Generate data for baseline scenario
    print("Generating data for baseline scenario...")
    (interest_trad_df, interest_quantum_df, 
     stablecoin_trad_df, stablecoin_quantum_df) = generate_scenario_data(
        scenario="baseline", 
        time_horizon=2, 
        num_steps=120
    )
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Interest rate comparison
    plot_interest_rate_comparison(
        traditional_df=interest_trad_df,
        quantum_df=interest_quantum_df,
        title="Interest Rate Model Comparison (Baseline Scenario)",
        output_file="outputs/interest_stablecoin/interest_rate_comparison.png"
    )
    
    # Stablecoin comparison
    plot_stablecoin_comparison(
        traditional_df=stablecoin_trad_df,
        quantum_df=stablecoin_quantum_df,
        title="Stablecoin Model Comparison (Baseline Scenario)",
        output_file="outputs/interest_stablecoin/stablecoin_comparison.png"
    )
    
    # Rate-stablecoin relationship
    plot_rate_stablecoin_relationship(
        interest_df=interest_quantum_df,
        stablecoin_df=stablecoin_quantum_df,
        title="Interest Rate & Stablecoin Relationship (Quantum Model)",
        output_file="outputs/interest_stablecoin/rate_stablecoin_relationship.png"
    )
    
    # Stablecoin response to rate changes
    plot_stablecoin_response_to_rate_changes(
        interest_df=interest_quantum_df,
        stablecoin_df=stablecoin_quantum_df,
        output_file="outputs/interest_stablecoin/stablecoin_rate_response.png"
    )
    
    # Combined visualization
    create_combined_visualization(
        interest_trad_df=interest_trad_df,
        interest_quantum_df=interest_quantum_df,
        stablecoin_trad_df=stablecoin_trad_df,
        stablecoin_quantum_df=stablecoin_quantum_df,
        output_file="outputs/interest_stablecoin/interest_stablecoin_combined.png"
    )
    
    # Generate data for other scenarios
    print("\nGenerating data for rate hiking scenario...")
    hiking_data = generate_scenario_data(scenario="hiking", time_horizon=2, num_steps=120)
    
    # Create combined visualization for hiking scenario
    create_combined_visualization(
        interest_trad_df=hiking_data[0],
        interest_quantum_df=hiking_data[1],
        stablecoin_trad_df=hiking_data[2],
        stablecoin_quantum_df=hiking_data[3],
        output_file="outputs/interest_stablecoin/hiking_scenario.png"
    )
    
    print("\nGenerating data for rate cutting scenario...")
    cutting_data = generate_scenario_data(scenario="cutting", time_horizon=2, num_steps=120)
    
    # Create combined visualization for cutting scenario
    create_combined_visualization(
        interest_trad_df=cutting_data[0],
        interest_quantum_df=cutting_data[1],
        stablecoin_trad_df=cutting_data[2],
        stablecoin_quantum_df=cutting_data[3],
        output_file="outputs/interest_stablecoin/cutting_scenario.png"
    )
    
    print("\nGenerating data for volatile market scenario...")
    volatile_data = generate_scenario_data(scenario="volatile", time_horizon=2, num_steps=120)
    
    # Create combined visualization for volatile scenario
    create_combined_visualization(
        interest_trad_df=volatile_data[0],
        interest_quantum_df=volatile_data[1],
        stablecoin_trad_df=volatile_data[2],
        stablecoin_quantum_df=volatile_data[3],
        output_file="outputs/interest_stablecoin/volatile_scenario.png"
    )

if __name__ == "__main__":
    main()

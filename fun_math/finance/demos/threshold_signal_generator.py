#!/usr/bin/env python3
"""
Threshold-Based Signal Generator for Xi/Psi Quantum Financial Model

This module implements the threshold-based trading signal generation system described
in the quantum financial model summary. It generates real-time buy and sell triggers
based on quantum coherence and phase alignment thresholds instead of using fixed-time
trading signals (e.g., end-of-day).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

class ThresholdSignalGenerator:
    """
    Threshold-based trading signal generator for the Xi/Psi Quantum Financial Model.
    Generates signals based on multiple quantum metrics crossing predefined thresholds.
    """
    
    def __init__(self, 
                 coherence_threshold=0.75, 
                 phase_buy_threshold=0.82, 
                 phase_sell_threshold=-0.78,
                 exit_coherence_threshold=0.60,
                 volatility_scaling=True,
                 momentum_confirmation=True):
        """
        Initialize the threshold-based signal generator
        
        Args:
            coherence_threshold: Minimum coherence required for signal generation
            phase_buy_threshold: Phase alignment threshold for buy signals
            phase_sell_threshold: Phase alignment threshold for sell signals
            exit_coherence_threshold: Coherence threshold for exiting positions
            volatility_scaling: Whether to dynamically adjust thresholds based on volatility
            momentum_confirmation: Whether to require momentum confirmation for signals
        """
        self.coherence_threshold = coherence_threshold
        self.phase_buy_threshold = phase_buy_threshold
        self.phase_sell_threshold = phase_sell_threshold
        self.exit_coherence_threshold = exit_coherence_threshold
        self.volatility_scaling = volatility_scaling
        self.momentum_confirmation = momentum_confirmation
        
        # State tracking
        self.current_position = 0  # -1=short, 0=neutral, 1=long
        self.position_entry_price = 0
        self.position_entry_time = None
        self.signals_generated = []
        
    def generate_signal(self, 
                        time, 
                        price, 
                        coherence, 
                        phase_alignment, 
                        momentum=None, 
                        volatility=None):
        """
        Generate trading signals based on quantum coherence and phase alignment
        
        Args:
            time: Timestamp for the signal evaluation
            price: Current price of the asset
            coherence: Quantum coherence measure (0-1)
            phase_alignment: Phase alignment between Xi and Psi (-1 to 1)
            momentum: Momentum indicator (optional)
            volatility: Market volatility measure (optional)
            
        Returns:
            Trading signal: 1=Buy, -1=Sell, 0=No action
        """
        # Adjust thresholds based on volatility if enabled
        coh_threshold = self.coherence_threshold
        buy_threshold = self.phase_buy_threshold
        sell_threshold = self.phase_sell_threshold
        
        if self.volatility_scaling and volatility is not None:
            # Scale thresholds based on current volatility relative to historical average
            # Higher volatility = higher thresholds to reduce false signals
            vol_scale = min(1.5, max(0.8, volatility / volatility.mean()))
            coh_threshold *= vol_scale
            buy_threshold *= vol_scale
            sell_threshold *= vol_scale
            
        # Determine base signal from coherence and phase alignment
        signal = 0
        
        # Generate entry signals
        if coherence >= coh_threshold:  # Only generate signals when coherence is high
            if self.current_position <= 0 and phase_alignment >= buy_threshold:
                # Buy signal
                if not self.momentum_confirmation or (momentum is not None and momentum > 0):
                    signal = 1
            elif self.current_position >= 0 and phase_alignment <= sell_threshold:
                # Sell signal
                if not self.momentum_confirmation or (momentum is not None and momentum < 0):
                    signal = -1
                    
        # Generate exit signals when currently in a position
        if self.current_position != 0:
            exit_signal = False
            
            # Exit on coherence breakdown
            if coherence < self.exit_coherence_threshold:
                exit_signal = True
                
            # Exit on phase alignment crossing zero (direction change)
            if (self.current_position > 0 and phase_alignment < 0) or \
               (self.current_position < 0 and phase_alignment > 0):
                exit_signal = True
                
            if exit_signal:
                signal = 0  # Signal to exit position
        
        # Update position tracking if signal would change position
        if (signal == 1 and self.current_position <= 0) or \
           (signal == -1 and self.current_position >= 0) or \
           (signal == 0 and self.current_position != 0):
            
            # Record the signal
            self.signals_generated.append({
                'time': time,
                'price': price,
                'signal': signal,
                'previous_position': self.current_position,
                'coherence': coherence,
                'phase_alignment': phase_alignment,
                'momentum': momentum,
                'volatility': volatility
            })
            
            # Update position tracking
            if signal != 0:
                self.current_position = signal
                self.position_entry_price = price
                self.position_entry_time = time
            else:
                self.current_position = 0
                self.position_entry_price = 0
                self.position_entry_time = None
                
        return signal
    
    def backtest(self, data, price_col='Close', time_col='Date', 
                coherence_col='Coherence', phase_col='PhaseAlignment', 
                momentum_col=None, volatility_col=None):
        """
        Backtest the threshold signal generator on historical data
        
        Args:
            data: DataFrame with historical price and quantum metric data
            price_col: Column name for price data
            time_col: Column name for timestamp data
            coherence_col: Column name for coherence data
            phase_col: Column name for phase alignment data
            momentum_col: Column name for momentum data (optional)
            volatility_col: Column name for volatility data (optional)
            
        Returns:
            DataFrame with signal and performance information
        """
        # Reset state
        self.current_position = 0
        self.position_entry_price = 0
        self.position_entry_time = None
        self.signals_generated = []
        
        # Generate signals for each time point
        for i, row in data.iterrows():
            # Extract relevant data
            time_val = row[time_col]
            price = row[price_col]
            coherence = row[coherence_col]
            phase = row[phase_col]
            
            # Get optional data if available
            momentum = row[momentum_col] if momentum_col is not None else None
            volatility = row[volatility_col] if volatility_col is not None else None
            
            # Generate signal
            self.generate_signal(time_val, price, coherence, phase, momentum, volatility)
            
        # Convert signals to DataFrame
        signals_df = pd.DataFrame(self.signals_generated)
        
        if not signals_df.empty:
            # Calculate returns for each completed trade
            returns = []
            entry_signals = []
            exit_signals = []
            
            for i in range(len(signals_df) - 1):
                current = signals_df.iloc[i]
                next_sig = signals_df.iloc[i + 1]
                
                # Check if this is an entry signal followed by an exit
                if current['signal'] != 0 and next_sig['signal'] == 0:
                    # Calculate return
                    entry_price = current['price']
                    exit_price = next_sig['price']
                    direction = current['signal']  # 1 for long, -1 for short
                    
                    # Calculate return (positive if price went up for long, down for short)
                    trade_return = direction * (exit_price - entry_price) / entry_price
                    
                    returns.append({
                        'entry_time': current['time'],
                        'exit_time': next_sig['time'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'Long' if direction > 0 else 'Short',
                        'return': trade_return,
                        'holding_period': (next_sig['time'] - current['time']).total_seconds() / (60 * 60 * 24)  # days
                    })
                    
                    entry_signals.append(i)
                    exit_signals.append(i + 1)
                    
            returns_df = pd.DataFrame(returns)
            
            # Calculate performance metrics
            if not returns_df.empty:
                total_return = (1 + returns_df['return']).prod() - 1
                avg_return = returns_df['return'].mean()
                win_rate = (returns_df['return'] > 0).mean()
                avg_holding_period = returns_df['holding_period'].mean()
                
                performance = {
                    'total_return': total_return,
                    'avg_return_per_trade': avg_return,
                    'win_rate': win_rate,
                    'avg_holding_period': avg_holding_period,
                    'number_of_trades': len(returns_df)
                }
                
                # Calculate drawdown
                if len(returns_df) > 1:
                    # Compound returns
                    cumulative = (1 + returns_df['return']).cumprod()
                    running_max = cumulative.cummax()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = drawdown.min()
                    performance['max_drawdown'] = max_drawdown
                    
                # Calculate annualized metrics if we have dates
                if 'entry_time' in returns_df.columns:
                    first_date = returns_df['entry_time'].min()
                    last_date = returns_df['exit_time'].max()
                    days = (last_date - first_date).total_seconds() / (60 * 60 * 24)
                    
                    if days > 0:
                        years = days / 365.25
                        annual_return = (1 + total_return) ** (1 / years) - 1
                        performance['annualized_return'] = annual_return
                        
                        # Calculate Sharpe ratio if we have more than one trade
                        if len(returns_df) > 1:
                            std_dev = returns_df['return'].std()
                            if std_dev > 0:
                                sharpe = (avg_return / std_dev) * np.sqrt(252 / avg_holding_period)
                                performance['sharpe_ratio'] = sharpe
                
                return signals_df, returns_df, performance
            
        # Return empty DataFrames if no signals or no completed trades
        return signals_df, pd.DataFrame(), {}
        
    def plot_signals(self, data, signals_df, price_col='Close', time_col='Date',
                    coherence_col='Coherence', phase_col='PhaseAlignment',
                    title="Threshold-Based Trading Signals", output_file=None):
        """
        Plot the price chart with trading signals and quantum metrics
        
        Args:
            data: DataFrame with historical price and quantum metric data
            signals_df: DataFrame with signal information from backtest
            price_col: Column name for price data
            time_col: Column name for timestamp data
            coherence_col: Column name for coherence data
            phase_col: Column name for phase alignment data
            title: Plot title
            output_file: File path to save the plot (if None, plot is displayed)
        """
        if data.empty or signals_df.empty:
            print("No data or signals to plot")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), 
                                           gridspec_kw={'height_ratios': [3, 1, 1]},
                                           sharex=True)
        
        # Plot price data
        ax1.plot(data[time_col], data[price_col], 'k-', label='Price')
        
        # Plot buy signals
        buy_signals = signals_df[signals_df['signal'] == 1]
        if not buy_signals.empty:
            ax1.scatter(buy_signals['time'], buy_signals['price'], 
                       marker='^', color='green', s=100, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = signals_df[signals_df['signal'] == -1]
        if not sell_signals.empty:
            ax1.scatter(sell_signals['time'], sell_signals['price'], 
                       marker='v', color='red', s=100, label='Sell Signal')
        
        # Plot exit signals
        exit_signals = signals_df[(signals_df['signal'] == 0) & 
                                (signals_df['previous_position'] != 0)]
        if not exit_signals.empty:
            ax1.scatter(exit_signals['time'], exit_signals['price'], 
                       marker='o', color='blue', s=80, label='Exit Signal')
        
        # Set up price plot
        ax1.set_ylabel('Price')
        ax1.set_title(title)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot coherence
        ax2.plot(data[time_col], data[coherence_col], 'g-', label='Coherence')
        ax2.axhline(y=self.coherence_threshold, color='g', linestyle='--', 
                   alpha=0.5, label=f'Coherence Threshold ({self.coherence_threshold})')
        ax2.axhline(y=self.exit_coherence_threshold, color='g', linestyle=':', 
                   alpha=0.5, label=f'Exit Threshold ({self.exit_coherence_threshold})')
        ax2.set_ylabel('Coherence')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Plot phase alignment
        ax3.plot(data[time_col], data[phase_col], 'b-', label='Phase Alignment')
        ax3.axhline(y=self.phase_buy_threshold, color='g', linestyle='--', 
                   alpha=0.5, label=f'Buy Threshold ({self.phase_buy_threshold})')
        ax3.axhline(y=self.phase_sell_threshold, color='r', linestyle='--', 
                   alpha=0.5, label=f'Sell Threshold ({self.phase_sell_threshold})')
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_ylabel('Phase Alignment')
        ax3.set_ylim(-1, 1)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Set up x-axis
        ax3.set_xlabel('Date')
        
        plt.tight_layout()
        
        # Save or display the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        else:
            plt.show()
        
        plt.close(fig)
        
    def compare_with_eod(self, data, threshold_results, eod_results, 
                        time_col='Date', price_col='Close',
                        title="Threshold vs End-of-Day Trading Performance",
                        output_file=None):
        """
        Compare performance between threshold-based and end-of-day trading
        
        Args:
            data: DataFrame with historical price data
            threshold_results: (signals_df, returns_df, performance) from threshold backtest
            eod_results: (signals_df, returns_df, performance) from EOD backtest
            time_col: Column name for timestamp data
            price_col: Column name for price data
            title: Plot title
            output_file: File path to save the plot (if None, plot is displayed)
        """
        # Unpack results
        _, threshold_returns, threshold_perf = threshold_results
        _, eod_returns, eod_perf = eod_results
        
        if threshold_returns.empty or eod_returns.empty:
            print("Insufficient return data for comparison")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                      gridspec_kw={'height_ratios': [3, 1]},
                                      sharex=True)
        
        # Plot price data
        ax1.plot(data[time_col], data[price_col], 'k-', alpha=0.3, label='Price')
        
        # Calculate cumulative returns
        threshold_equity = (1 + threshold_returns['return']).cumprod()
        eod_equity = (1 + eod_returns['return']).cumprod()
        
        # Create date index for equity curves
        threshold_dates = threshold_returns['exit_time'].tolist()
        threshold_equity_series = pd.Series(threshold_equity.values, index=threshold_dates)
        
        eod_dates = eod_returns['exit_time'].tolist()
        eod_equity_series = pd.Series(eod_equity.values, index=eod_dates)
        
        # Plot equity curves
        ax2.plot(threshold_equity_series.index, threshold_equity_series, 
                'g-', linewidth=2, label='Threshold-Based')
        ax2.plot(eod_equity_series.index, eod_equity_series, 
                'b-', linewidth=2, label='End-of-Day')
        
        # Plot trades on price chart
        # Threshold buys
        threshold_buys = threshold_returns[threshold_returns['direction'] == 'Long']
        if not threshold_buys.empty:
            ax1.scatter(threshold_buys['entry_time'], threshold_buys['entry_price'], 
                       marker='^', color='green', s=80, label='Threshold Buy')
            ax1.scatter(threshold_buys['exit_time'], threshold_buys['exit_price'], 
                       marker='o', color='green', s=60)
            
        # Threshold sells
        threshold_sells = threshold_returns[threshold_returns['direction'] == 'Short']
        if not threshold_sells.empty:
            ax1.scatter(threshold_sells['entry_time'], threshold_sells['entry_price'], 
                       marker='v', color='red', s=80, label='Threshold Sell')
            ax1.scatter(threshold_sells['exit_time'], threshold_sells['exit_price'], 
                       marker='o', color='red', s=60)
            
        # EOD buys
        eod_buys = eod_returns[eod_returns['direction'] == 'Long']
        if not eod_buys.empty:
            ax1.scatter(eod_buys['entry_time'], eod_buys['entry_price'], 
                       marker='^', color='blue', s=80, alpha=0.5, label='EOD Buy')
            ax1.scatter(eod_buys['exit_time'], eod_buys['exit_price'], 
                       marker='o', color='blue', s=60, alpha=0.5)
            
        # EOD sells
        eod_sells = eod_returns[eod_returns['direction'] == 'Short']
        if not eod_sells.empty:
            ax1.scatter(eod_sells['entry_time'], eod_sells['entry_price'], 
                       marker='v', color='purple', s=80, alpha=0.5, label='EOD Sell')
            ax1.scatter(eod_sells['exit_time'], eod_sells['exit_price'], 
                       marker='o', color='purple', s=60, alpha=0.5)
        
        # Set up price plot
        ax1.set_ylabel('Price')
        ax1.set_title(title)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Set up equity plot
        ax2.set_ylabel('Equity Curve (Multiple of Initial)')
        ax2.set_xlabel('Date')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Add performance comparison as text
        textstr = '\n'.join((
            f"Threshold: Return={threshold_perf.get('annualized_return', 0):.2%}, Sharpe={threshold_perf.get('sharpe_ratio', 0):.2f}, "
            f"Win Rate={threshold_perf.get('win_rate', 0):.2%}, Drawdown={threshold_perf.get('max_drawdown', 0):.2%}",
            f"EOD: Return={eod_perf.get('annualized_return', 0):.2%}, Sharpe={eod_perf.get('sharpe_ratio', 0):.2f}, "
            f"Win Rate={eod_perf.get('win_rate', 0):.2%}, Drawdown={eod_perf.get('max_drawdown', 0):.2%}",
            f"Improvement: Return=+{threshold_perf.get('annualized_return', 0) - eod_perf.get('annualized_return', 0):.2%}, "
            f"Sharpe=+{threshold_perf.get('sharpe_ratio', 0) - eod_perf.get('sharpe_ratio', 0):.2f}, "
            f"Win Rate=+{threshold_perf.get('win_rate', 0) - eod_perf.get('win_rate', 0):.2%}"
        ))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=props)
        
        plt.tight_layout()
        
        # Save or display the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {output_file}")
        else:
            plt.show()
            
        plt.close(fig)

def generate_sample_data(days=252, seed=42):
    """
    Generate sample data for demonstration purposes
    
    Args:
        days: Number of trading days to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with sample price, coherence, and phase alignment data
    """
    np.random.seed(seed)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, periods=days)
    
    # Generate price with random walk and trend
    returns = np.random.normal(0.0005, 0.01, days)
    price = 100 * (1 + returns).cumprod()
    
    # Add a trend to make it more realistic
    trend = np.linspace(0, 0.3, days)
    price = price * (1 + trend)
    
    # Generate coherence (cyclical pattern with noise)
    t = np.linspace(0, 8*np.pi, days)
    coherence_base = 0.5 + 0.3 * np.sin(t/3)
    coherence = np.clip(coherence_base + np.random.normal(0, 0.1, days), 0, 1)
    
    # Generate phase alignment (cyclical pattern with noise)
    phase_base = 0.7 * np.sin(t/2)
    phase = np.clip(phase_base + np.random.normal(0, 0.15, days), -1, 1)
    
    # Calculate momentum (direction of price change)
    momentum = np.zeros(days)
    momentum[1:] = np.diff(price)
    
    # Calculate volatility (rolling standard deviation of returns)
    volatility = pd.Series(returns).rolling(20, min_periods=1).std().values
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Close': price,
        'Coherence': coherence,
        'PhaseAlignment': phase,
        'Momentum': momentum,
        'Volatility': volatility
    })
    
    return df

def run_demo():
    """Run a demonstration of the threshold signal generator"""
    print("\n===== Threshold-Based Signal Generator Demo =====\n")
    
    # Generate sample data
    print("Generating sample market data...")
    data = generate_sample_data(days=252)
    
    # Create threshold signal generator
    threshold_gen = ThresholdSignalGenerator(
        coherence_threshold=0.75,
        phase_buy_threshold=0.82,
        phase_sell_threshold=-0.78,
        exit_coherence_threshold=0.60,
        volatility_scaling=True,
        momentum_confirmation=True
    )
    
    # Run backtest
    print("Running threshold-based backtest...")
    threshold_results = threshold_gen.backtest(
        data, 
        price_col='Close', 
        time_col='Date',
        coherence_col='Coherence', 
        phase_col='PhaseAlignment',
        momentum_col='Momentum', 
        volatility_col='Volatility'
    )
    
    # Display results
    signals_df, returns_df, performance = threshold_results
    
    print(f"\nThreshold-Based Strategy Performance:")
    print(f"  Total Return: {performance.get('total_return', 0):.2%}")
    print(f"  Annualized Return: {performance.get('annualized_return', 0):.2%}")
    print(f"  Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
    print(f"  Win Rate: {performance.get('win_rate', 0):.2%}")
    print(f"  Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
    print(f"  Number of Trades: {performance.get('number_of_trades', 0)}")
    print(f"  Avg Holding Period: {performance.get('avg_holding_period', 0):.1f} days")
    
    # Create end-of-day strategy for comparison
    print("\nCreating end-of-day strategy for comparison...")
    
    # Simplified EOD strategy (buy when coherence > 0.6 and phase > 0)
    eod_signals = []
    eod_position = 0
    
    for i in range(0, len(data), 1):  # Process daily data
        row = data.iloc[i]
        
        if eod_position == 0:
            # Entry logic
            if row['Coherence'] > 0.6:
                if row['PhaseAlignment'] > 0:
                    signal = 1  # Buy
                elif row['PhaseAlignment'] < 0:
                    signal = -1  # Sell
                else:
                    signal = 0
                
                if signal != 0:
                    eod_signals.append({
                        'time': row['Date'],
                        'price': row['Close'],
                        'signal': signal,
                        'previous_position': 0,
                        'coherence': row['Coherence'],
                        'phase_alignment': row['PhaseAlignment']
                    })
                    eod_position = signal
        else:
            # Exit logic - simplistic for demo
            # Exit after holding for 5 days or if phase alignment crosses zero
            days_held = i - next(j for j, s in enumerate(eod_signals) if s['previous_position'] == 0)
            phase_crossed = (eod_position > 0 and row['PhaseAlignment'] < 0) or \
                           (eod_position < 0 and row['PhaseAlignment'] > 0)
            
            if days_held >= 5 or phase_crossed:
                eod_signals.append({
                    'time': row['Date'],
                    'price': row['Close'],
                    'signal': 0,  # Exit
                    'previous_position': eod_position,
                    'coherence': row['Coherence'],
                    'phase_alignment': row['PhaseAlignment']
                })
                eod_position = 0
    
    # Convert EOD signals to DataFrame and calculate returns
    eod_signals_df = pd.DataFrame(eod_signals)
    
    # Calculate EOD returns
    eod_returns = []
    
    for i in range(len(eod_signals_df) - 1):
        current = eod_signals_df.iloc[i]
        next_sig = eod_signals_df.iloc[i + 1]
        
        # Check if this is an entry signal followed by an exit
        if current['signal'] != 0 and next_sig['signal'] == 0:
            # Calculate return
            entry_price = current['price']
            exit_price = next_sig['price']
            direction = current['signal']  # 1 for long, -1 for short
            
            # Calculate return
            trade_return = direction * (exit_price - entry_price) / entry_price
            
            eod_returns.append({
                'entry_time': current['time'],
                'exit_time': next_sig['time'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'Long' if direction > 0 else 'Short',
                'return': trade_return,
                'holding_period': (next_sig['time'] - current['time']).total_seconds() / (60 * 60 * 24)  # days
            })
    
    eod_returns_df = pd.DataFrame(eod_returns)
    
    # Calculate EOD performance metrics
    eod_performance = {}
    if not eod_returns_df.empty:
        total_return = (1 + eod_returns_df['return']).prod() - 1
        avg_return = eod_returns_df['return'].mean()
        win_rate = (eod_returns_df['return'] > 0).mean()
        avg_holding_period = eod_returns_df['holding_period'].mean()
        
        eod_performance = {
            'total_return': total_return,
            'avg_return_per_trade': avg_return,
            'win_rate': win_rate,
            'avg_holding_period': avg_holding_period,
            'number_of_trades': len(eod_returns_df)
        }

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 1. Load and prepare the data
# 1. Load and prepare the data

df = pd.read_csv('merged_data_1min.csv', parse_dates=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
df.set_index('timestamp', inplace=True)
df = df.sort_values(by='timestamp')
df = df.dropna()
print(df.head())


# 2. Define a hedged correlation-based strategy
def hedged_correlation_strategy(data, base_coin, target_coin, window=1440, threshold=0.7):
    """
    Hedged strategy based on correlation divergence
    """
    print(f"Running hedged strategy for {base_coin} vs {target_coin}...")
    
    # Create copy and ensure proper datetime index
    df = data.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Get last month of data using the data's own timestamps
    end_date = df.index.max()
    start_date = end_date - pd.DateOffset(months=1)
    df = df[df.index >= start_date]
    
    print(f"Analysis period: {start_date} to {end_date}")
    
    # Calculate rolling statistics
    
    # Calculate ratio and z-score
    ratio = df[target_coin] / df[base_coin]
    
    rolling_mean = ratio.mean()
    rolling_std = ratio.std()
    z_score = (ratio - rolling_mean) / rolling_std
    z_score = z_score.fillna(0)
    rolling_corr = df[base_coin].corr(df[target_coin])
    # Generate signals
    signals = pd.DataFrame(index=df.index)
    signals['target_position'] = 0
    signals['base_position'] = 0
    
    # Set positions based on z-score thresholds
    signals.loc[z_score < -1.5, 'target_position'] = 1
    signals.loc[z_score < -1.5, 'base_position'] = -1
    signals.loc[z_score > 1.5, 'target_position'] = -1
    signals.loc[z_score > 1.5, 'base_position'] = 1
    
    print(f"Generated {len(signals[signals['target_position'] != 0])} trading signals")
    return signals, rolling_corr, z_score

# 3. Backtest the hedged strategy
def backtest_hedged(data, signals, base_coin, target_coin, initial_capital=10000, position_size=0.2):
    """Run a backtest on the hedged strategy signals"""
    # # Create position change columns
    # target_positions = signals['target_position'].diff()
    # base_positions = signals['base_position'].diff()

    # # Calculate returns of both assets
    # data_filtered = data.loc[signals.index]
    # target_returns = data_filtered[target_coin].dropna()
    # base_returns = data_filtered[base_coin].dropna()

    # # Calculate each leg's strategy returns (position taken at close and evaluated next day)
    # target_strategy_returns = signals['target_position'].shift(1) * target_returns * 0.98 * position_size
    # base_strategy_returns = signals['base_position'].shift(1) * base_returns * 0.98 * position_size
    """Run a backtest on the hedged strategy signals"""
    # Ensure data alignment
    common_index = data.index.intersection(signals.index)
    data = data.loc[common_index].copy()
    signals = signals.loc[common_index].copy()
    
    print(f"Backtesting on {len(data)} periods")
    
    # Calculate returns
    
    returns = pd.DataFrame(index=data.index)
    returns[target_coin] = data[target_coin]
    returns[base_coin] = data[base_coin]
    returns = returns.fillna(0)
    print("return",returns.tail())
    
    # Calculate strategy returns
    target_positions = signals['target_position'].diff().fillna(0)
    base_positions = signals['base_position'].diff().fillna(0)
    
    target_strategy_returns = signals['target_position'] * returns[target_coin] * 0.98 * position_size
    base_strategy_returns = signals['base_position'] * returns[base_coin] *  0.98 * position_size
    print("target_strategy_returns",target_strategy_returns.head())
    # Combined strategy returns (both legs)
    combined_strategy_returns = (target_strategy_returns + base_strategy_returns)

    # For comparison: single-leg target-only strategy
    target_only_returns = target_strategy_returns  # Just the target leg

    # Calculate cumulative returns
    cumulative_target_returns = (1 + target_strategy_returns).cumprod()
    cumulative_base_returns = (1 + base_strategy_returns).cumprod()
    cumulative_combined_strategy = (1 + combined_strategy_returns).cumprod()
    cumulative_target_only = (1 + target_only_returns).cumprod()

    # Calculate portfolio values
    portfolio_value_combined = initial_capital * cumulative_combined_strategy
    portfolio_value_target_only = initial_capital * cumulative_target_only

    # Calculate metrics for combined strategy
    total_return = (portfolio_value_combined.iloc[-1] / initial_capital - 1) * 100
    annual_return = ((portfolio_value_combined.iloc[-1] / initial_capital) ** (
                17520 / len(portfolio_value_combined)) - 1) * 100
    sharpe_ratio = combined_strategy_returns.mean() / combined_strategy_returns.std() * np.sqrt(365)
    max_drawdown = (portfolio_value_combined / portfolio_value_combined.cummax() - 1).min() * 100

    # Calculate metrics for target-only strategy (for comparison)
    total_return_target = (portfolio_value_target_only.iloc[-1] / initial_capital - 1) * 100
    annual_return_target = ((portfolio_value_target_only.iloc[-1] / initial_capital) ** (
                17520 / len(portfolio_value_target_only)) - 1) * 100
    sharpe_ratio_target = target_only_returns.mean() / target_only_returns.std() * np.sqrt(365)
    max_drawdown_target = (portfolio_value_target_only / portfolio_value_target_only.cummax() - 1).min() * 100

    # Calculate correlation between strategy returns and market
    # Using equal-weighted combination of both assets as proxy for "market"
    market_returns = (target_strategy_returns + base_strategy_returns) / 2
    market_correlation = combined_strategy_returns.corr(market_returns)
    market_correlation_target_only = target_only_returns.corr(market_returns)

    metrics = {
        'Total Return (%)': total_return,
        'Annual Return (%)': annual_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Market Correlation': market_correlation,
        'Target Trades': target_positions[target_positions != 0].count(),
        'Base Trades': base_positions[base_positions != 0].count()
    }

    metrics_target_only = {
        'Total Return (%)': total_return_target,
        'Annual Return (%)': annual_return_target,
        'Sharpe Ratio': sharpe_ratio_target,
        'Max Drawdown (%)': max_drawdown_target,
        'Market Correlation': market_correlation_target_only,
        'Trades': target_positions[target_positions != 0].count()
    }

    return {
        'metrics': metrics,
        'metrics_target_only': metrics_target_only,
        'portfolio_value': portfolio_value_combined,
        'portfolio_value_target_only': portfolio_value_target_only,
        'cumulative_target_returns': cumulative_target_returns,
        'cumulative_base_returns': cumulative_base_returns,
        'cumulative_strategy_returns': cumulative_combined_strategy,
        'cumulative_target_only': cumulative_target_only,
        'signals': signals,
        'corr': market_correlation,
        'corr_target_only': market_correlation_target_only
    }


# 4. Run the backtest for the paired strategy
signals_hedged, corr_hedged, zscore_hedged = hedged_correlation_strategy(
    df, 'virtual', 'ai16z', window=1440,threshold=0.7)
backtest_results_hedged = backtest_hedged(df, signals_hedged, 'virtual', 'ai16z')


# 5. Visualize results with comparison between hedged and target-only
def plot_hedged_backtest_results(data, results, title):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 20), sharex=True)

    # Plot portfolio values
    ax1.plot(results['portfolio_value'], label='Hedged Strategy')
    ax1.plot(results['portfolio_value_target_only'], label='Target-Only Strategy')
    ax1.set_title(f'{title} - Portfolio Value Comparison')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)

    # Plot cumulative returns comparison
    ax2.plot(results['cumulative_target_returns'], label=f'{title.split(" vs ")[1]} Returns')
    ax2.plot(results['cumulative_base_returns'], label=f'{title.split(" vs ")[0]} Returns')
    ax2.plot(results['cumulative_strategy_returns'], label='Hedged Strategy')
    ax2.plot(results['cumulative_target_only'], label='Target-Only Strategy')
    ax2.set_title(f'{title} - Cumulative Returns Comparison')
    ax2.set_ylabel('Cumulative Returns')
    ax2.legend()
    ax2.grid(True)

    # Plot positions for both assets
    ax3.plot(results['signals']['target_position'], label=f'{title.split(" vs ")[1]} Position')
    ax3.plot(results['signals']['base_position'], label=f'{title.split(" vs ")[0]} Position')
    ax3.set_title(f'{title} - Positions')
    ax3.set_ylabel('Position')
    ax3.legend()
    ax3.grid(True)

    # Plot rolling correlation and z-score
    ax4.set_title('Correlation and Z-Score')
    ax4_twin = ax4.twinx()
    ax4.plot(corr_hedged, 'b-', label='Correlation')
    ax4_twin.plot(zscore_hedged, 'r-', label='Z-Score')
    ax4.set_ylabel('Correlation', color='b')
    ax4_twin.set_ylabel('Z-Score', color='r')
    ax4.grid(True)

    # Combine both legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Add metrics as text
    hedged_metrics_text = "HEDGED STRATEGY:\n" + '\n'.join([f"{k}: {v:.4f}" for k, v in results['metrics'].items()])
    target_metrics_text = "\nTARGET-ONLY STRATEGY:\n" + '\n'.join(
        [f"{k}: {v:.4f}" for k, v in results['metrics_target_only'].items()])
    ax1.text(0.1, 0.1, hedged_metrics_text + target_metrics_text, transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.8), fontsize=9)

    plt.tight_layout()
    plt.savefig(f'Hedged_{title.replace(" ", "_")}_backtest.png')
    print(f"Backtest results saved as 'Hedged_{title.replace(' ', '_')}_backtest.png'")

    plt.show()


# Plot results for the hedged strategy
plot_hedged_backtest_results(df, backtest_results_hedged, 'VIRTUALUSDT vs AI16ZUSDT')

# Print comparison metrics
print("\nHedged Strategy Metrics for VIRTUALUSDT vs AI16ZUSDT:")
for k, v in backtest_results_hedged['metrics'].items():
    print(f"{k}: {v:.4f}")

print("\nTarget-Only Strategy Metrics for VIRTUALUSDT vs AI16ZUSDT:")
for k, v in backtest_results_hedged['metrics_target_only'].items():
    print(f"{k}: {v:.4f}")

print(
    f"\nMarket Correlation Reduction: {backtest_results_hedged['corr_target_only'] - backtest_results_hedged['corr']:.4f}")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from pykalman import KalmanFilter
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
import traceback
def calculate_beta_with_rolling_windows(merged_data):
    # Create lists to store results
    rolling_betas = []
    
    # Define timedelta for the rolling window
    window_size = pd.Timedelta(days=30)
    start_time = merged_data['timestamp'].iloc[0] + window_size
    times = []
    
    # Loop through the merged_data with a rolling window
    for end in merged_data['timestamp']:
        if end < start_time:
            continue
        start = end - window_size
        times.append(end)
        # Select the windowed data
        window_data = merged_data[(merged_data['timestamp'] >= start) & (merged_data['timestamp'] <= end)]
        
        if len(window_data) < 2:  # Need at least two observations to calculate beta
            continue  # Skip if we don't have enough data

        data_X = window_data['symbol1']
        data_y = window_data['symbol2']
        
        # Log-transform data
        X = sm.add_constant(np.log(data_X))
        y = np.log(data_y)

        # Fit OLS model
        model = sm.OLS(y, X).fit()
        
        # Fit GLS model without custom covariance
        gls_model = sm.GLS(y, X).fit()  # No sigma parameter
        beta = gls_model.params['symbol1']  
        print(f"beta at {end} is",beta)

        # Calculate spread for the last observation in the current window
        last_X = np.log(data_X.iloc[-1])  # Log of the last X value
        last_y = np.log(data_y.iloc[-1])  # Log of the last y value
        spread = last_y - (last_X * beta) - model.params['const']

        # Append the spread for the last observation
        merged_data.loc[merged_data['timestamp'] == end, 'spread'] = spread
        merged_data.loc[merged_data['timestamp'] == end, 'beta'] = beta

        rolling_betas.append(beta)

    # Perform ADF test on calculated spreads
    adf_result = adfuller(merged_data['spread'].dropna())
    
    # Extract ADF results
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]

    # Print ADF results
    print("ADF Statistic:", adf_statistic)
    print("p-value:", p_value)
    for key, value in critical_values.items():
        print(f"   {key}: {value}")

    return merged_data
import os
def generate_merge_data(symbol_x,symbol_y):
    path = os.getcwd()
    print("path",path)
    file_path1 = path + f'/data/1m/{symbol_x}_1m.csv'
    file_path2 = path + f'/data/1m/{symbol_y}_1m.csv'
    data_x = pd.read_csv(file_path1)
    data_y = pd.read_csv(file_path2)
  
    data_x['timestamp'] = pd.to_datetime(data_x['timestamp'])
    data_y['timestamp'] = pd.to_datetime(data_y['timestamp'])
    data_ai_subset = data_x[['timestamp', 'close']].rename(columns={'close': 'symbol1'})
    data_vr_subset = data_y[['timestamp', 'close']].rename(columns={'close': 'symbol2'})
    merage_data = pd.merge(data_ai_subset, data_vr_subset, on='timestamp', how='inner')
    merage_data = merage_data.dropna()
    merage_data = calculate_beta_with_rolling_windows(merage_data)
    merage_data = merage_data.dropna()
    merage_data.to_csv(path + f'/data/1m/{symbol_x}-{symbol_y}-merge.csv')
    print("merge_data_done!")
    return merage_data
def higher_lower_mean(merge_data):
    max_spread = merge_data['spread'].max()
    min_spread = merge_data['spread'].min()
    max_range = max_spread-(max_spread-min_spread)/2
    print("max_range",max_range)
    min_range = min_spread+(max_spread-min_spread)/2
    merge_data['state'] = np.where(
        (merge_data['spread'] > max_range) & (merge_data['spread'] <= max_spread),
        2, 1)
    state_2_rows = merge_data.loc[merge_data['state'] == 2]
    # Print the rows
    print("Rows where state == 2:")
    print(state_2_rows)
    lower_mean = merge_data.loc[merge_data['state'] == 1, 'spread'].mean()
    higher_mean = merge_data.loc[merge_data['state'] == 2, 'spread'].mean()
    lower_std = merge_data.loc[merge_data['state'] == 1, 'spread'].std()
    higher_std = merge_data.loc[merge_data['state'] == 2, 'spread'].std()
    print("lower_mean",lower_mean)
    print("higher_mean",higher_mean)
    print("low_std",lower_std)
    print("high_std",higher_std)
    
    return lower_mean, higher_mean,lower_std,higher_std

def generate_trading_signal_dynamic(X, lower_mean, higher_mean, lower_std, higher_std, rho):
    # Initialize positions and signals
    X['position_x'] = 0
    X['position_y'] = 0
    X['signal'] = 0  # Initialize trading signal

    # Define bounds for each state
    upper_bound_1 = lower_mean + 0.75 * lower_std
    lower_bound_1 = lower_mean - 0.75 * lower_std
    upper_bound_2 = higher_mean + 0.75 * higher_std
    lower_bound_2 = higher_mean - 0.75 * higher_std

    # Initialize transition counts matrix
    transition_counts = np.zeros((2, 2))
    recent_transitions = []
    rolling_window_size = 100

    transition_counts = np.zeros((2, 2))

    for i in range(len(X)):
        if i > 0:  # Avoid index error
            previous_state = X['state'].iloc[i - 1]
            current_state = X['state'].iloc[i]

            # Add the transition to the recent transitions list
            recent_transitions.append((previous_state, current_state))
            print("Recent Transitions:", recent_transitions)

            # Keep only the last `rolling_window_size` transitions
            if len(recent_transitions) > rolling_window_size:
                recent_transitions.pop(0)

            # Reset transition counts for the rolling window
            transition_counts = np.zeros((2, 2))
            for prev, curr in recent_transitions:
                transition_counts[prev - 1, curr - 1] += 1  # Adjust indices to be zero-based

            print("Transition Counts Matrix:\n", transition_counts)

            # Calculate p and q dynamically
            if transition_counts[0].sum() > 0:
                p = transition_counts[0, 1] / transition_counts[0].sum()
            else:
                p = 0

            if transition_counts[1].sum() > 0:
                q = transition_counts[1, 0] / transition_counts[1].sum()
            else:
                q = 0
            # Generate trading signals
            if X['state'].iloc[i] == 1:
                if p > rho:
                    print("state",X['state'].iloc[i])
                    if X['spread'].iloc[i] < lower_bound_1:
                        X.loc[i, 'position_x'] = -1  # Sell asset X
                        X.loc[i, 'position_y'] = 1   # Buy asset Y
                    elif X['spread'].iloc[i] > upper_bound_1:
                        X.loc[i, 'position_x'] = 1   # Buy asset X
                        X.loc[i, 'position_y'] = -1  # Sell asset Y
                    else:
                        X.loc[i, 'position_x'] = 0
                        X.loc[i, 'position_y'] = 0
                else:
                    X.loc[i, 'position_x'] = 0
                    X.loc[i, 'position_y'] = 0
            elif X['state'].iloc[i] == 2:
                
                if q > rho:
                    print("state",X['state'].iloc[i])
                    if X['spread'].iloc[i] < lower_bound_2:
                        X.loc[i, 'position_x'] = -1  # Sell asset X
                        X.loc[i, 'position_y'] = 1   # Buy asset Y
                    elif X['spread'].iloc[i] > upper_bound_2:
                        X.loc[i, 'position_x'] = 1   # Buy asset X
                        X.loc[i, 'position_y'] = -1  # Sell asset Y
                    else:
                        X.loc[i, 'position_x'] = 0
                        X.loc[i, 'position_y'] = 0
                else:
                    X.loc[i, 'position_x'] = 0
                    X.loc[i, 'position_y'] = 0
            else:
                print("error")
        else:
            continue
    return X
def calculate_pnl(X):
    # Initialize P&L
    X['pnl'] = 0.0
    entry_price_x = 0
    entry_price_y = 0
    PNL = 0
    trades = 0

    for i in range(len(X)):
        # Check if we have an entry position
        if X['position_x'].iloc[i] != 0:
            if entry_price_x == 0:
                entry_price_x = X['symbol1'].iloc[i]
                entry_price_y = X['symbol2'].iloc[i]
            else:
                continue
        # Check for position closing
        if X['position_x'].iloc[i] == 0 and entry_price_x !=0:
            # Calculate P&L
            amount_x = 100 / entry_price_x
            amount_y = abs(100 * X['beta'].iloc[i] / entry_price_y)
            print("amount_x",amount_x)
            print("amount_y",amount_y)
            pnl_x = amount_x * (X['symbol1'].iloc[i] - entry_price_x) * X['position_x'].iloc[i-1]
            pnl_y = amount_y * (X['symbol2'].iloc[i] - entry_price_y) * X['position_y'].iloc[i-1]
            X.loc[i, 'pnl'] = pnl_x + pnl_y
            PNL += pnl_x + pnl_y
            trades += 1
            print("pnl",X['pnl'].iloc[i])
            # Reset entry prices
            entry_price_y = 0
            entry_price_x = 0
    print("PNL",PNL)
    print("trades",trades)
    return X
import matplotlib.pyplot as plt

def draw_picture(X,symbol1,symbol2):
    # Create a figure with 3 subplots
    fig, ax1 = plt.subplots(3, 1, figsize=(12, 12))

    # Plot 1: PnL trend
    ax1[0].plot(X['timestamp'], X['pnl'].cumsum(), label='Cumulative PnL', color='blue')
    ax1[0].set_title('PnL Trend')
    ax1[0].set_xlabel('Timestamp')
    ax1[0].set_ylabel('Cumulative PnL')
    ax1[0].legend()
    ax1[0].grid()

    # Plot 2: Positions (position_x and position_y)
    ax1[1].plot(X['timestamp'], X['position_x'], label='Position X', color='green', alpha=0.7)
    ax1[1].plot(X['timestamp'], X['position_y'], label='Position Y', color='orange', alpha=0.7)
    ax1[1].set_title('Positions')
    ax1[1].set_xlabel('Timestamp')
    ax1[1].set_ylabel('Position')
    ax1[1].legend()
    ax1[1].grid()

    # Plot 3: Symbol prices (symbol1 and symbol2)
    ax2 = ax1[2].twinx()  # Create a secondary y-axis for symbol2
    ax1[2].plot(X['timestamp'], X['symbol1'], label='Symbol1 Price', color='blue')
    ax2.plot(X['timestamp'], X['symbol2'], label='Symbol2 Price', color='red')
    ax1[2].set_title('Symbol Prices')
    ax1[2].set_xlabel('Timestamp')
    ax1[2].set_ylabel('Symbol1 Price', color='blue')
    ax2.set_ylabel('Symbol2 Price', color='red')
    ax1[2].legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1[2].grid()

    # Adjust layout and show the plot
    plt.tight_layout()
    path = os.getcwd()
    picture_path = path + f'/pictures/{symbol1}&{symbol2}.png'
    plt.savefig(picture_path)
def main():
    symbol1 = "TON-USDT-SWAP"
    symbol2 = "NOT-USDT-SWAP"
    #merge_data = generate_merge_data(symbol1,symbol2)
    path = os.getcwd()
    merge_data = pd.read_csv(path + f'/data/1m/{symbol1}-{symbol2}-merge.csv')
    # Rename columns of merge_data
    merge_data.columns = ['uniname', 'timestamp', 'symbol1', 'symbol2', 'spread', 'beta']
    lower_mean, higher_mean,lower_std,higher_std = higher_lower_mean(merge_data)
    rho = 0.01
    merge_data = generate_trading_signal_dynamic(merge_data,lower_mean,higher_mean, lower_std, higher_std, rho)
    print(merge_data)
    # calculate
    pnl_data = calculate_pnl(merge_data)
    draw_picture(pnl_data,symbol1,symbol2)

    print(pnl_data)

main()
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def exit_position(ratio, ai_position, virtual_position, PNL, upper_bound, lower_bound, mean,
                  current_time, entry_time, data_ai, data_vr):
    """
    Modified to use timestamps for position tracking instead of indices
    """

    is_exit = False

    # Find current prices
    ai_current_price = data_ai.loc[data_ai['timestamp'] == current_time, 'close'].iloc[0]
    vr_current_price = data_vr.loc[data_vr['timestamp'] == current_time, 'close'].iloc[0]
    amount1 = 100/ai_current_price
    amount2 = 100/vr_current_price
    # No position case
    if ai_position == 0 and virtual_position == 0:
        if ratio >= lower_bound and ratio <= upper_bound:
            # No position, and ratio is within bounds - do nothing
            pass
        elif ratio < lower_bound:
            # Ratio below lower bound - go long AI, short VR
            ai_position = 1
            virtual_position = -1
            entry_time = current_time
        elif ratio > upper_bound:
            # Ratio above upper bound - go short AI, long VR
            ai_position = -1
            virtual_position = 1
            entry_time = current_time

    # Long AI, Short VR position case
    elif ai_position == 1 and virtual_position == -1:
        if ratio > upper_bound:
            # Signal reversed - close position and open opposite
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]

            # Calculate PNL (long AI, short VR)
            ai_pnl = amount1 * (ai_current_price - ai_entry_price)
            vr_pnl = amount2 * (vr_entry_price - vr_current_price)  # Note: short position
            PNL += ai_pnl + vr_pnl

            # Open new position (reverse)
            ai_position = -1
            virtual_position = 1
            entry_time = current_time

        elif ratio > mean:
            # Ratio back to normal - close position
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]

            # Calculate PNL (long AI, short VR)
            ai_pnl = amount1 * (ai_current_price - ai_entry_price)
            vr_pnl = amount2 * (vr_entry_price - vr_current_price)  # Note: short position
            PNL += ai_pnl + vr_pnl

            # No position
            ai_position = 0
            virtual_position = 0

    # Short AI, Long VR position case
    elif ai_position == -1 and virtual_position == 1:
        if ratio < lower_bound:
            # Signal reversed - close position and open opposite
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]

            # Calculate PNL (short AI, long VR)
            ai_pnl = amount1 * (ai_entry_price - ai_current_price)  # Note: short position
            vr_pnl = amount2 * (vr_current_price - vr_entry_price)
            PNL += ai_pnl + vr_pnl

            # Open new position (reverse)
            ai_position = 1
            virtual_position = -1
            entry_time = current_time

        elif ratio < mean:
            # Ratio back to normal - close position
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]

            # Calculate PNL (short AI, long VR)
            ai_pnl = amount1 * (ai_entry_price - ai_current_price)  # Note: short position
            vr_pnl = amount2 * (vr_current_price - vr_entry_price)
            PNL += ai_pnl + vr_pnl

            # No position
            ai_position = 0
            virtual_position = 0

    return ai_position, virtual_position, PNL, is_exit, entry_time


def calculate_bounds_properly(data_ai, data_vr, current_timestamp, lookback_days=14):
    """Calculate bounds using only historical data"""
    # Define the lookback window end (current point) and start (30 days before)
    lookback_start = current_timestamp - timedelta(days=lookback_days)
    # Filter data to only include the lookback period UP TO current timestamp (no future data)
    hist_ai = data_ai[(data_ai['timestamp'] >= lookback_start) &
                      (data_ai['timestamp'] < current_timestamp)]
    hist_vr = data_vr[(data_vr['timestamp'] >= lookback_start) &
                      (data_vr['timestamp'] < current_timestamp)]

    # Ensure we have enough data to calculate meaningful bounds
    if len(hist_ai) < 5 or len(hist_vr) < 5:  # Arbitrary threshold to ensure enough data
        return None, None

    # Align timestamps and calculate ratio
    common_times = pd.Index(hist_ai['timestamp']).intersection(pd.Index(hist_vr['timestamp']))
    hist_ai = hist_ai[hist_ai['timestamp'].isin(common_times)]
    hist_vr = hist_vr[hist_vr['timestamp'].isin(common_times)]

    # Calculate the ratio
    ratio = hist_ai['close'] / hist_vr['close']
    ratio = ratio.dropna()

    if len(ratio) < 5:  # Another check after alignment
        return None, None

    # Calculate bounds
    mean = ratio.mean()
    std = ratio.std()
    upper_bound = mean + 1.25 * std
    lower_bound = mean - 1.25 * std

    return upper_bound, lower_bound, mean

def backtest(symbol1,symbol2):
    # Load data
    data_ai = pd.read_csv(f"C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\{symbol1}_1m.csv")
    data_vr = pd.read_csv(f"C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\{symbol2}_1m.csv")
    data_ai['timestamp'] = pd.to_datetime(data_ai['timestamp'], format="mixed")
    data_vr['timestamp'] = pd.to_datetime(data_vr['timestamp'], format="mixed")

    # Filter for the backtest period (last 30 days)
    today = pd.Timestamp.now().normalize()
    start_date = today - pd.Timedelta(days=60)
    mask_ai = (data_ai['timestamp'].dt.date >= start_date.date())
    mask_vr = (data_vr['timestamp'].dt.date >= start_date.date())

    # Create the test datasets
    data_ai_test = data_ai[mask_ai].copy()
    data_vr_test = data_vr[mask_vr].copy()

    # Align timestamps
    common_times = pd.Index(data_ai_test['timestamp']).intersection(pd.Index(data_vr_test['timestamp']))
    data_ai_test = data_ai_test[data_ai_test['timestamp'].isin(common_times)]
    data_vr_test = data_vr_test[data_vr_test['timestamp'].isin(common_times)]

    # Ensure data is sorted by timestamp
    data_ai_test = data_ai_test.sort_values('timestamp')
    data_vr_test = data_vr_test.sort_values('timestamp')

    # Initialize tracking variables
    timestamps = []
    pnl_history = [0]  # Start with 0 PNL
    ai_position_history = [0]  # Start with no position
    vr_position_history = [0]  # Start with no position
    ratio_history = []
    upper_bound_history = []
    lower_bound_history = []
    mean_history = []

    # Trading variables
    ai_position = 0
    vr_position = 0
    PNL = 0
    entry_time = None

    # Need a warm-up period to calculate initial bounds
    warmup_period = 10  # Days
    warmup_end = data_ai_test['timestamp'].iloc[0] + pd.Timedelta(days=warmup_period)

    # Skip warmup period for trading but use it for initial bounds
    start_idx = data_ai_test[data_ai_test['timestamp'] > warmup_end].index[0]

    # Run the backtest
    for i in range(len(data_ai_test)):
        current_time = data_ai_test['timestamp'].iloc[i]

        # Skip trading during warmup period
        if current_time <= warmup_end:
            continue

        # Calculate ratio for current time
        ai_price = data_ai_test['close'].iloc[i]
        vr_price = data_vr_test['close'].iloc[i]
        ratio = ai_price / vr_price

        # Calculate bounds using historical data only
        upper_bound, lower_bound, mean = calculate_bounds_properly(
            data_ai, data_vr, current_time, lookback_days=30
        )

        # Skip if bounds couldn't be calculated (not enough history)
        if upper_bound is None or lower_bound is None:
            continue

        # Update trading positions
        ai_position, vr_position, PNL, is_exit, entry_time = exit_position(
            ratio, ai_position, vr_position, PNL, upper_bound, lower_bound, mean,
            current_time, entry_time, data_ai_test, data_vr_test
        )

        # Store historical values
        timestamps.append(current_time)
        pnl_history.append(PNL)
        ai_position_history.append(ai_position)
        vr_position_history.append(vr_position)
        ratio_history.append(ratio)
        upper_bound_history.append(upper_bound)
        lower_bound_history.append(lower_bound)
        mean_history.append(mean)

        # Log trades
        if is_exit:
            print(f"Trade at {current_time}")
            print(f"AI position: {ai_position}")
            print(f"VR position: {vr_position}")
            print(f"Current PNL: {PNL:.2f}")

    # Plot results with the dynamic bounds
    plot_trading_results(
        timestamps,
        pnl_history[1:],  # Skip the initial 0
        ai_position_history[1:],  # Skip the initial 0
        vr_position_history[1:],  # Skip the initial 0
        ratio_history,
        upper_bound_history,
        lower_bound_history,
        mean_history,
        symbol1 = symbol1,
        symbol2 = symbol2
    )

    print(f"Final PNL: {PNL:.2f}")
    results = {
        "timestamps": timestamps,
        'symbol1': symbol1,
        'symbol2': symbol2,
        "pnl": pnl_history[1:],  # Skip the initial 0
        "ai_position": ai_position_history[1:],  # Skip the initial 0
        "vr_position": vr_position_history[1:],  # Skip the initial 0
        "ratio": ratio_history,
    }
    results_1 = {
        'symbol1': symbol1,
        'symbol2': symbol2,
        "pnl": pnl_history.iloc[-1],  # Skip the initial 0
    }
    results = pd.DataFrame(results)
    results.to_csv(f"C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_analysis\\results\\all_pairs_details_results_original.csv",mode='a',header=False,index=False)
    results_1 = pd.DataFrame(results_1)
    results_1.to_csv(f"C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_analysis\\results\\all_pairs_results_original.csv",mode='a',header=False,index=False)
    print(f"Final PNL: {PNL:.2f}")


def plot_trading_results(timestamps, pnl, ai_pos, vr_pos, ratio, upper_bounds, lower_bounds, means,symbol1,symbol2):
    """Plot trading results with dynamic bounds"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 1, figure=fig)

    # Plot 1: PNL over time
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(timestamps, pnl, 'g-', label='PNL')
    ax1.set_title('Cumulative PNL')
    ax1.set_ylabel('PNL Value')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Positions
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(timestamps, ai_pos, 'b-', label=f'{symbol1} Position')
    ax2.plot(timestamps, vr_pos, 'r-', label=f'{symbol2} Position')
    ax2.set_title('Position Changes')
    ax2.set_ylabel('Position')
    ax2.grid(True)
    ax2.legend()

    # Plot 3: Ratio with dynamic bounds
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(timestamps, ratio, 'k-', label='Price Ratio')
    ax3.plot(timestamps, upper_bounds, 'r--', label='Upper Bound')
    ax3.plot(timestamps, lower_bounds, 'g--', label='Lower Bound')
    ax3.plot(timestamps, means, 'b--', label='Mean')
    ax3.set_title('Price Ratio and Bounds')
    ax3.set_ylabel('Ratio')
    ax3.grid(True)
    ax3.legend()

    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_analysis\\pairs_highest_corr\\{symbol1}&{symbol2}.png')
    print("Results plotted and saved as 'trading_results.png'")
    plt.close()


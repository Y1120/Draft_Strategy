import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
from calculate_beta import calculate_beta_with_Kalmen,calculate_bound,calculate_beta
import traceback
"""
def exit_position(spread,beta, ai_position, virtual_position, PNL, upper_bound, lower_bound, mean,
                  current_time, entry_time, data_ai, data_vr,ai_amount,vr_amount):

    print("lower_bound",lower_bound)
    print("upper_bound",upper_bound)
    is_exit = False

    # Find current prices
    ai_current_price = data_ai.loc[data_ai['timestamp'] == current_time, 'close'].iloc[0]
    vr_current_price = data_vr.loc[data_vr['timestamp'] == current_time, 'close'].iloc[0]
    
    # No position case
    if ai_position == 0 and virtual_position == 0:
        if spread >= lower_bound and spread <= upper_bound:
            # No position, and ratio is within bounds - do nothing
            ai_amount = 0
            vr_amount = 0
            pass
        elif spread < lower_bound:
            # Ratio below lower bound - go long AI, short VR
            ai_position = -1
            virtual_position = 1
            entry_time = current_time
            ai_amount = 100/ai_current_price
            vr_amount = beta * ai_amount * ai_current_price / vr_current_price
        elif spread > upper_bound:
            # Ratio above upper bound - go short AI, long VR
            ai_position = 1
            virtual_position = -1
            entry_time = current_time
            ai_amount = 100 / ai_current_price
            vr_amount = beta * ai_amount * ai_current_price / vr_current_price

    # Long AI, Short VR position case
    elif ai_position == 1 and virtual_position == -1:
        if spread < lower_bound:
            # Signal reversed - close position and open opposite
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]

            # Calculate PNL (long AI, short VR)
            ai_pnl = ai_amount * (ai_current_price - ai_entry_price)
            vr_pnl = vr_amount * (vr_entry_price - vr_current_price)  # Note: short position
            PNL += ai_pnl + vr_pnl

            # Open new position (reverse)
            ai_position = -1
            virtual_position = 1
            entry_time = current_time
            #new entry amount
            ai_amount = 100/ai_current_price
            vr_amount = beta * ai_amount * ai_current_price / vr_current_price

        elif spread < mean:
            # Ratio back to normal - close position
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]

            # Calculate PNL (long AI, short VR)
            ai_pnl = ai_amount * (ai_current_price - ai_entry_price)
            vr_pnl = vr_amount * (vr_entry_price - vr_current_price)  # Note: short position
            PNL += ai_pnl + vr_pnl

            # No position
            ai_position = 0
            virtual_position = 0
            ai_amount = 0
            vr_amount = 0

    # Short AI, Long VR position case
    elif ai_position == -1 and virtual_position == 1:
        if spread > upper_bound:
            # Signal reversed - close position and open opposite
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]

            # Calculate PNL (short AI, long VR)
            ai_pnl = ai_amount * (ai_entry_price - ai_current_price)  # Note: short position
            vr_pnl = vr_amount * (vr_current_price - vr_entry_price)
            PNL += ai_pnl + vr_pnl

            # Open new position (reverse)
            ai_position = 1
            virtual_position = -1
            entry_time = current_time
            ai_amount = 100/ai_current_price
            vr_amount = beta * ai_amount * ai_current_price / vr_current_price

        elif spread > mean:
            # Ratio back to normal - close position
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]

            # Calculate PNL (short AI, long VR)
            ai_pnl = ai_amount * (ai_entry_price - ai_current_price)  # Note: short position
            vr_pnl = vr_amount * (vr_current_price - vr_entry_price)
            PNL += ai_pnl + vr_pnl

            # No position
            ai_position = 0
            virtual_position = 0
            ai_amount = 0
            vr_amount = 0

    return ai_position, virtual_position, PNL, is_exit, entry_time,ai_amount,vr_amount
"""
def exit_position(spread,beta, ai_position, virtual_position, PNL, upper_bound, lower_bound, mean,
                  current_time, entry_time, data_ai, data_vr):


    is_exit = False

    # Find current prices
    amount1 = 0
    amount2 = 0
    # No position case

    if ai_position == 0 and virtual_position == 0:
        
        if spread >= lower_bound and spread <= upper_bound:
            # No position, and ratio is within bounds - do nothing
            pass
        elif spread < lower_bound:
            # Ratio below lower bound - go long AI, short VR
            ai_position = -1
            virtual_position = 1
            entry_time = current_time
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == current_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == current_time, 'close'].iloc[0]
        elif spread > upper_bound:
            # Ratio above upper bound - go short AI, long VR
            ai_position = 1
            virtual_position = -1
            entry_time = current_time
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == current_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == current_time, 'close'].iloc[0]

    # Long AI, Short VR position case
    elif ai_position == 1 and virtual_position == -1:
        amount1 = 100/ai_entry_price
        amount2 =  beta * amount1 * ai_entry_price / vr_entry_price
        print("amount1",amount1)
        print("amount2",amount2)
        print("spread",spread)
        if spread < lower_bound:
            # Signal reversed - close position and open opposite
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]
            amount1 = 100/ai_entry_price
            amount2 =  beta * amount1 * ai_entry_price / vr_entry_price
            # Calculate PNL (long AI, short VR)
            ai_pnl = amount1 * (ai_current_price - ai_entry_price)
            vr_pnl = amount2 * (vr_entry_price - vr_current_price)  # Note: short position
            PNL += ai_pnl + vr_pnl

            # Open new position (reverse)
            ai_position = -1
            virtual_position = 1
            entry_time = current_time

        elif spread < mean:
            # Ratio back to normal - close position
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]
            amount1 = 100/ai_entry_price
            amount2 =  beta * amount1 * ai_entry_price / vr_entry_price
            # Calculate PNL (long AI, short VR)
            ai_pnl = amount1 * (ai_current_price - ai_entry_price)
            vr_pnl = amount2 * (vr_entry_price - vr_current_price)  # Note: short position
            PNL += ai_pnl + vr_pnl

            # No position
            ai_position = 0
            virtual_position = 0

    # Short AI, Long VR position case
    elif ai_position == -1 and virtual_position == 1:
        ai_current_price = data_ai.loc[data_ai['timestamp'] == current_time, 'close'].iloc[0]
        vr_current_price = data_vr.loc[data_vr['timestamp'] == current_time, 'close'].iloc[0]
        print("amount1",amount1)
        print("amount2",amount2)
        print("spread",spread)
        if spread > upper_bound:
            # Signal reversed - close position and open opposite
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]
            amount1 = 100/ai_entry_price
            amount2 =  beta * amount1 * ai_entry_price / vr_entry_price

            # Calculate PNL (short AI, long VR)
            ai_pnl = amount1 * (ai_entry_price - ai_current_price)  # Note: short position
            vr_pnl = amount2 * (vr_current_price - vr_entry_price)
            PNL += ai_pnl + vr_pnl

            # Open new position (reverse)
            ai_position = 1
            virtual_position = -1
            entry_time = current_time

        elif spread > mean:
            # Ratio back to normal - close position
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]
            amount1 = 100/ai_entry_price
            amount2 =  beta * amount1 * ai_entry_price / vr_entry_price
            # Calculate PNL (short AI, long VR)
            ai_pnl = amount1 * (ai_entry_price - ai_current_price)  # Note: short position
            vr_pnl = amount2 * (vr_current_price - vr_entry_price)
            PNL += ai_pnl + vr_pnl

            # No position
            ai_position = 0
            virtual_position = 0
    print("pnl",PNL)
    print("ai_amount",amount1)

    return ai_position, virtual_position, PNL, is_exit, entry_time,amount1,amount2
def backtest1(symbol1,symbol2):
    # Load data
    data_ai = pd.read_csv(f"C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\{symbol1}_1m.csv",parse_dates=['timestamp'])
    data_vr = pd.read_csv(f"C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\{symbol2}_1m.csv",parse_dates=['timestamp'])

    
    # Convert 'timestamp' to datetime
    data_ai['timestamp'] = pd.to_datetime(data_ai['timestamp'])
    data_vr['timestamp'] = pd.to_datetime(data_vr['timestamp'])

    data_ai_subset = data_ai[['timestamp', 'close']].rename(columns={'close': 'ai16z'})
    data_vr_subset = data_vr[['timestamp', 'close']].rename(columns={'close': 'virtual'})

    merage_data = pd.merge(data_ai_subset, data_vr_subset, on='timestamp', how='inner')

    merage_data = merage_data.dropna()
    merage_data.columns = ['timestamp', 'ai16z', 'virtual']


    
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
    pnl_history = [0]          # Start with 0 PNL
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
    warmup_period = 30  # Days
    warmup_end = data_ai_test['timestamp'].iloc[0] + pd.Timedelta(days=warmup_period)
    print(warmup_end)
    # Skip warmup period for trading but use it for initial bounds
    start_idx = data_ai_test[data_ai_test['timestamp'] > warmup_end].index[0]
    merage_data,beta = calculate_beta(merage_data)

    #beta.to_csv(f"C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_analysis\\results\\beta_for_{symbol1}_{symbol2}.csv",header=False,index=False)
    #merage_data,beta = calculate_beta_with_Kalmen(merage_data)
    

    # Run the backtest

    amount_ai = []
    amount_vr = []
    spreads = []
    for i in range(len(merage_data)):
        print("beta",beta)
        current_time = merage_data['timestamp'].iloc[i]
        print("current_time",current_time)

        # Skip trading during warmup period
        if current_time <= warmup_end:
            continue
        # Calculate ratio for current time

        
        # Calculate bounds using historical data only
        upper_bound, lower_bound, mean = calculate_bound(
            merage_data, current_time, lookback_days=30
        )

        # Skip if bounds couldn't be calculated (not enough history)
        if upper_bound is None or lower_bound is None:
            continue
        spread = merage_data['spread'].iloc[i]
        print("spread",spread)
        spreads.append(spread)
        # Update trading positions
        ai_position, vr_position, PNL, is_exit, entry_time,ai_amount,vr_amount = exit_position(
            spread, beta,ai_position, vr_position, PNL, upper_bound, lower_bound, mean,
            current_time, entry_time, data_ai_test, data_vr_test
        )
        print("PNL",PNL)

        # Store historical values
        timestamps.append(current_time)
        pnl_history.append(PNL)
        ai_position_history.append(ai_position)
        vr_position_history.append(vr_position)
        ratio_history.append(spread)
        upper_bound_history.append(upper_bound)
        lower_bound_history.append(lower_bound)
        mean_history.append(mean)
        amount_ai.append(ai_amount)
        amount_vr.append(vr_amount)

        # Log trades
        if is_exit:
            print(f"Trade at {current_time}")
            print(f"AI position: {ai_position}")
            print(f"VR position: {vr_position}")
            print(f"Current PNL: {PNL:.2f}")
    num = len(timestamps)
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
        symbol2 = symbol2,
        symbol1_close=merage_data['ai16z'].iloc[-num:],
        symbol2_close=merage_data['virtual'].iloc[-num:]
    )
    results = {
        "timestamps": timestamps,
        'symbol1': merage_data['ai16z'].iloc[-num:],
        'symbol2': merage_data['virtual'].iloc[-num:],
        "ai_position": ai_position_history[1:],  # Skip the initial 0
        "vr_position": vr_position_history[1:],  # Skip the initial 0
        "pnl": pnl_history[1:],  # Skip the initial 0
        

        "ratio": ratio_history,
    }
    
    results = pd.DataFrame(results)
    results.to_csv(f"C:\\Users\\theo\\Desktop\\constant_0.75std.csv",header=True,index=False)
    print(f"Final PNL: {PNL:.2f}")


def plot_trading_results(timestamps, pnl, ai_pos, vr_pos, ratio, upper_bounds, lower_bounds, means,symbol1,symbol2,symbol1_close,symbol2_close):
    """Plot trading results with dynamic bounds"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 1, figure=fig)

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

    ax4 = fig.add_subplot(gs[3])
    ax4 .plot(timestamps, symbol1_close, 'm-', label='Close Price of '+symbol1)
    ax4 .plot(timestamps, symbol2_close, 'c-', label='Close Price of '+symbol2)
    ax4 .set_title('Price Ratio and Bounds')
    ax4 .set_ylabel('Ratio')
    ax4 .grid(True)
    ax4 .legend()
    

    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', rotation=45)
    folder_path = f'C:\\Users\\theo\\Desktop\\'
    file_path = os.path.join(folder_path, f'constant_0.75std.png')

        # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(file_path)
    plt.tight_layout()
    print("Results plotted and saved as 'trading_results.png'")
    plt.close()

def main():
    try:
        backtest1('AI16Z-USDT-SWAP','VIRTUAL-USDT-SWAP')
    except:
        traceback.print_exc()
main()
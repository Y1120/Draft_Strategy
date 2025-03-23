import pandas as pd
from datetime import datetime,timedelta
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
def exit_position(ratio,ai_position,virtual_position,PNL,upper_bound,lower_bound,index_now,last_index,data_ai,data_vr):
    amount_ai = 57
    amount_vr = 16
    is_exit = False
    print("index_now",index_now)
    index = index_now
    last_index = 0
    if ai_position == 0 and virtual_position == 0:
        print("no position found.")
        if ratio >= lower_bound and ratio <= upper_bound:
            ai_position = 0
            vr_position = 0
            print("no position found, no action taken")
        elif ratio < lower_bound:
            ai_position = 1
            vr_position = -1
            last_index = index
        elif ratio > upper_bound:
            ai_position = -1
            vr_position = 1
            last_index = index
        else:
            print("Error in creating position")
        
    elif ai_position == 1 and virtual_position == -1:
        if ratio > upper_bound:
            index = index_now
            is_exit = True
            print("signal reverse, close position and create another one")
            ai_position = -1
            vr_position = 1
            if last_index != 0:
                PNL += amount_ai * (data_ai["close"].iloc[index] - data_ai["close"].iloc[last_index]) + amount_vr * (data_vr["close"].iloc[index] - data_vr["close"].iloc[last_index])
                last_index = index
            else:
                print("error in last_index")
                last_index = index
            # position created
             
        elif ratio < lower_bound:
            print("position not close yet, no action taken")
            ai_position = 1
            vr_position = -1
        elif ratio >= lower_bound and ratio <= upper_bound:
            is_exit = True
            print("go back to normal situation,close position")
            ai_position = 0
            vr_position = 0
            PNL += amount_ai * (data_ai["close"].iloc[index] - data_ai["close"].iloc[last_index]) + amount_vr * (data_vr["close"].iloc[index] - data_vr["close"].iloc[last_index])
            
        else:
            print("signal not strong enough, no action taken")

    elif ai_position== -1 and virtual_position == 1:
        if ratio < lower_bound:
            is_exit = True
            print("signal reverse, close position and create another one")
            ai_position = 1
            vr_position = -1
            PNL += amount_ai * (data_ai["close"].iloc[index] - data_ai["close"].iloc[last_index]) + amount_vr * (data_vr["close"].iloc[index] - data_vr["close"].iloc[last_index])
            last_index = index
            # position created
        elif ratio > upper_bound:
            ai_position = -1
            vr_position = 1
            print("position not close yet, no action taken")
        elif ratio >= lower_bound and ratio <= upper_bound:
            is_exit = True
            print("go back to normal situation, close position")
            ai_position = 0
            vr_position = 0
            PNL += amount_ai * (data_ai["close"].iloc[index] - data_ai["close"].iloc[last_index]) + amount_vr * (data_vr["close"].iloc[index] - data_vr["close"].iloc[last_index])
        else:
            print("signal not strong enough, no action taken")
    else:
        print("no position found. Error in exit_position")
    if ai_position is None or virtual_position is None:
        ai_position = 0
        vr_position = 0
    print("last_index",last_index)
    return ai_position,vr_position,PNL,is_exit,last_index

def calculate_bound(data_ai,data_vr,start_date):
    # Calculate the upper and lower bound for the ratio
    print("start_date",start_date)
    one_month_ago = start_date - timedelta(days = 30)
    print("one_month_ago",one_month_ago)
    data_ai = data_ai.loc[data_ai['timestamp'] > one_month_ago]
    data_vr = data_vr.loc[data_vr['timestamp'] > one_month_ago]
    ratio = data_ai['close'] / data_vr['close']
    print("ratio",ratio)
    ratio = ratio.dropna()
    mean = np.mean(ratio)
    std = np.std(ratio)
    upper_bound = mean + 1.25 * std
    lower_bound = mean - 1.25 * std
    print({"mean":mean,"std":std,"upper_bound":upper_bound,"lower_bound":lower_bound})
    return upper_bound,lower_bound

def main():
    data_ai = pd.read_csv('C:/Users/theo/Desktop/Astra-folder/ai16z-1min.csv',
                           )
    data_vr = pd.read_csv('C:/Users/theo/Desktop/Astra-folder/virtual-1min.csv',
                          )
    data_ai['timestamp'] = pd.to_datetime(data_ai['timestamp'],format = "mixed")
    data_vr['timestamp'] = pd.to_datetime(data_vr['timestamp'],format = "mixed")
    PNL = 0
    today = pd.Timestamp.now().normalize()
    yesterday = today - pd.Timedelta(days=1)
    mask_ai = (data_ai['timestamp'].dt.date >= yesterday.date())
    mask_vr = (data_vr['timestamp'].dt.date >= yesterday.date())
    print(data_ai.tail())
    print(data_vr.tail())
    print("mask_vr",mask_vr)
    data_ai_yesterday = data_ai[mask_ai].copy()
    data_vr_yesterday = data_vr[mask_vr].copy()
    common_times = pd.Index(data_ai_yesterday['timestamp']).intersection(pd.Index(data_vr_yesterday['timestamp']))    
    data_ai_yesterday = data_ai_yesterday[data_ai_yesterday['timestamp'].isin(common_times)]
    data_vr_yesterday = data_vr_yesterday[data_vr_yesterday['timestamp'].isin(common_times)]
   
    # print("3",data_ai_yesterday,len(data_ai_yesterday))
    # print("4",data_vr_yesterday,len(data_vr_yesterday))
    # ai_position = 0
    # vr_position = 0
    # last_index = 0
    # for index in range(len(data_ai_yesterday)):
    #     ratio = data_ai_yesterday["close"].iloc[index] / data_vr_yesterday["close"].iloc[index]
    #     ai_position,vr_position,PNL,is_exit,last_index = exit_position(ratio,ai_position,vr_position,PNL,upper_bound,lower_bound,index,last_index,data_ai,data_vr)
    #     if is_exit:
    #         print("AI position: ",ai_position)
    #         print("VR position: ",vr_position)
    #         print("Last index: ",last_index)
    #         is_exit = False
    #     else:
    #         print("No action taken")
    # print("PNL: ",PNL)
    timestamps = []
    pnl_history = []
    ai_position_history = []
    vr_position_history = []
    ratio_history = []

    ai_position = 0
    vr_position = 0
    last_index = 0
    for index in range(len(data_ai_yesterday)):
        timestamp = data_ai_yesterday["timestamp"].iloc[index]
        ratio = data_ai_yesterday["close"].iloc[index] / data_vr_yesterday["close"].iloc[index]
        upper_bound,lower_bound = calculate_bound(data_ai,data_vr,start_date = data_ai_yesterday["timestamp"].iloc[index])
        ai_position, vr_position, PNL, is_exit, last_index = exit_position(
            ratio, ai_position, vr_position, PNL, upper_bound, lower_bound,
            index, last_index, data_ai_yesterday, data_vr_yesterday
        )
        
        # Store historical values
        timestamps.append(timestamp)
        pnl_history.append(PNL)
        ai_position_history.append(ai_position)
        vr_position_history.append(vr_position)
        ratio_history.append(ratio)
        
        if is_exit:
            print(f"Trade at {timestamp}")
            print(f"AI position: {ai_position}")
            print(f"VR position: {vr_position}")
            print(f"Current PNL: {PNL:.2f}")
            is_exit = False
    
    # Plot results
    plot_trading_results(
        timestamps,
        pnl_history,
        ai_position_history,
        vr_position_history,
        ratio_history,
        upper_bound,
        lower_bound
    )
    
    print(f"Final PNL: {PNL:.2f}")
def plot_trading_results(timestamps, pnl, ai_pos, vr_pos, ratio, upper, lower):
    """Plot trading results with multiple subplots"""
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
    ax2.plot(timestamps, ai_pos, 'b-', label='AI Position')
    ax2.plot(timestamps, vr_pos, 'r-', label='VR Position')
    ax2.set_title('Position Changes')
    ax2.set_ylabel('Position')
    ax2.grid(True)
    ax2.legend()
    
    # Plot 3: Ratio with bounds
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(timestamps, ratio, 'k-', label='Price Ratio')
    ax3.axhline(y=upper, color='r', linestyle='--', label='Upper Bound')
    ax3.axhline(y=lower, color='g', linestyle='--', label='Lower Bound')
    ax3.set_title('Price Ratio and Bounds')
    ax3.set_ylabel('Ratio')
    ax3.grid(True)
    ax3.legend()
    
    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('trading_results.png')
    print("Results plotted and saved as 'trading_results.png'")
    plt.show()

main()
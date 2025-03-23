import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from pykalman import KalmanFilter
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
import os
# calculate the beta
def calculate_beta_with_rolling_windows(merged_data, window_size):
    # Loop through the merged_data with a rolling window
    data_X = merged_data['ai16z'].iloc[-window_size:]
    data_y = merged_data['virtual'].iloc[-window_size:]
        
    if len(data_X) < window_size or len(data_y) < window_size:
        print("Not enough data to calculate beta.")  
        
    # Log-transform data
    X = sm.add_constant(np.log(data_X))
    y = np.log(data_y)
    model = sm.OLS(y, X).fit()  
    gls_model = sm.GLS(y, X).fit()  # No sigma parameter
    beta = gls_model.params['ai16z']  # Get the beta for ai16z

    # Calculate spread for the last observation in the current window
    last_X = np.log(data_X.iloc[-1])  # Log of the last X value
    last_y = np.log(data_y.iloc[-1])  # Log of the last y value
    spread = last_y - (last_X * beta) - model.params['const']
    # Append the spread for the last observation
    merged_data.loc[merged_data.index[-1], 'spread'] = spread

    # Join rolling betas back to the original merged_data DataFrame
    # Perform ADF test on calculated spreads
    adf_result = adfuller(merged_data['spread'].dropna())
    
    # Extract ADF results
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]

    # Print ADF results
    print("ADF Statistic:", adf_statistic)
    print("p-value:", p_value)
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"   {key}: {value}")

    print("Final merged data with rolling betas and spreads:")
    print(merged_data)

    return merged_data,beta
# initial capital == 10
def fetch_historical_data(exchange, symbol):
    try:
        symbol1 = symbol.replace("/USDT:USDT","").lower()
        print(f"Fetching historical data for {symbol1}...")
        path = 'C:\\Users\\theo\\Desktop\\Astra-folder\\virtual_ai16_analysis\\'
        data = pd.read_csv(f'{path}{symbol1}-1min.csv', parse_dates=['timestamp'], index_col='timestamp')
        start_dt = pd.to_datetime(data.index[-1]) + pd.Timedelta(minutes=1)
        end_dt = datetime.now() + pd.Timedelta(minutes=1)
        since = int(start_dt.timestamp() * 1000)
        end = int(end_dt.timestamp() * 1000)
        data_list = []

        while since < end:
            try: 
                
                candles = exchange.fetch_ohlcv(
                    symbol,
                    timeframe='1m',
                    since=since,
                    limit=100
                )
                
                if not candles:
                    print("No more data to fetch")
                    break
                    
                data_list.extend(candles)
                since = candles["timestamp"][-1] + 60 * 1000
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break

        if not data_list:
            print("No data fetched")
            return data['close'].iloc[-1]  # Return last known price from CSV
            
        # Convert list to DataFrame
        df = pd.DataFrame(data_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.sort_values('timestamp', inplace=True)
        
        # Save to CSV
        df.to_csv(f'{symbol1}-1min.csv', mode='a', header=False, index=False)
        print(f"Data saved to {symbol1}-1min.csv")
        
        return float(df['close'].iloc[-1])  # Return latest close price

    except Exception as e:
        print(f"Error in fetch_historical_data: {e}")
        return None

def process_order(exchange, order,symbol):
    try:
        # Handle case where order is a list
        order_status = exchange.fetch_order(order['id'], symbol)

        order_info = [
            order_status['symbol'],
            order_status['side'],
            float(order_status['price']),
            float(order_status['amount']),
            order_status['status']
        ]
        return order_info
    except Exception as e:
        print(f"Error processing order: {str(e)}")
        return None

def exit_position(exchange,spread,beta, ai_position, virtual_position, PNL, upper_bound, lower_bound, mean, current_time, entry_time, entry_ai_amount,entry_vr_amount,data_ai, data_vr):
    is_exit = False
    # Find current prices
    # No position case
    if ai_position == 0 and virtual_position == 0:
        
        if spread >= lower_bound and spread <= upper_bound:
            # No position, and ratio is within bounds - do nothing
            pass
        elif spread < lower_bound:
            # Ratio below lower bound - go long AI, short VR
            entry_time = current_time
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == current_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == current_time, 'close'].iloc[0]
            ai_position = -1
            virtual_position = 1
            entry_ai_amount  = 100/ai_entry_price
            entry_vr_amount = 100 * beta/vr_entry_price
            ai_order = exchange.create_order('AI16Z-USDT-SWAP', 'market', 'buy', entry_ai_amount)
            vr_order = exchange.create_order('VIRTUAL-USDT-SWAP', 'market', 'sell', entry_vr_amount)
          
        elif spread > upper_bound:
            # Ratio above upper bound - go short AI, long VR
            ai_position = 1
            virtual_position = -1
            entry_time = current_time
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == current_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == current_time, 'close'].iloc[0]
            entry_ai_amount  = 100/ai_entry_price
            entry_vr_amount = 100 * beta/vr_entry_price
            ai_order = exchange.create_order('AI16Z-USDT-SWAP', 'market', 'buy', entry_ai_amount)
            vr_order = exchange.create_order('VIRTUAL-USDT-SWAP', 'market', 'sell', entry_vr_amount )

    # Long AI, Short VR position case
    elif ai_position == 1 and virtual_position == -1:
        ai_current_price = data_ai.loc[data_ai['timestamp'] == current_time, 'close'].iloc[0]
        vr_current_price = data_vr.loc[data_vr['timestamp'] == current_time, 'close'].iloc[0]
        if spread < lower_bound:
            # Signal reversed - close position and open opposite
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]

            # close position
            ai_order = exchange.create_order('AI16Z-USDT-SWAP', 'market', 'sell', entry_ai_amount)
            vr_order = exchange.create_order('VIRTUAL-USDT-SWAP', 'market', 'buy', entry_vr_amount )
            ai_pnl = entry_ai_amount * (ai_current_price - ai_entry_price)
            vr_pnl = entry_vr_amount * (vr_entry_price - vr_current_price)  # Note: short position
            PNL += ai_pnl + vr_pnl

            # Open new position (reverse)
            ai_position = -1
            virtual_position = 1
            entry_time = current_time
            entry_ai_amount = 100/ai_current_price
            entry_vr_amount = 100 * beta/vr_current_price
            ai_order1 = exchange.create_order('AI16Z-USDT-SWAP', 'market', 'buy', entry_ai_amount)
            vr_order1 = exchange.create_order('VIRTUAL-USDT-SWAP', 'market', 'sell', entry_vr_amount)

        elif spread < mean:
            
            # Ratio back to normal - close position
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]
            # close position
            ai_order = exchange.create_order('AI16Z-USDT-SWAP', 'market', 'sell', entry_ai_amount)
            vr_order = exchange.create_order('VIRTUAL-USDT-SWAP', 'market', 'buy', entry_vr_amount )

            # Calculate PNL (long AI, short VR)
            ai_pnl = entry_ai_amount * (ai_current_price - ai_entry_price)
            vr_pnl = entry_vr_amount * (vr_entry_price - vr_current_price)  # Note: short position
            PNL += ai_pnl + vr_pnl

            # No position
            ai_position = 0
            virtual_position = 0

    # Short AI, Long VR position case
    elif ai_position == -1 and virtual_position == 1:
        ai_current_price = data_ai.loc[data_ai['timestamp'] == current_time, 'close'].iloc[0]
        vr_current_price = data_vr.loc[data_vr['timestamp'] == current_time, 'close'].iloc[0]
        if spread > upper_bound:
            # Signal reversed - close position and open opposite
            is_exit = True

            # close position
            ai_order = exchange.create_order('AI16Z-USDT-SWAP', 'market', 'buy', entry_ai_amount)
            vr_order = exchange.create_order('VIRTUAL-USDT-SWAP', 'market', 'sell', entry_vr_amount )
            ai_pnl = entry_ai_amount * (ai_entry_price - ai_current_price)  # Note: short position
            vr_pnl = entry_vr_amount * (vr_current_price - vr_entry_price)
            PNL += ai_pnl + vr_pnl

            # Open new position (reverse)
            ai_position = 1
            virtual_position = -1
            entry_time = current_time
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]
            entry_ai_amount = 100/ai_entry_price
            entry_vr_amount =  beta * 100 / vr_entry_price
            ai_order1 = exchange.create_order('AI16Z-USDT-SWAP', 'market', 'sell', entry_ai_amount)
            vr_order1 = exchange.create_order('VIRTUAL-USDT-SWAP', 'market', 'buy', entry_vr_amount )

        elif spread > mean:
            # Ratio back to normal - close position
            is_exit = True

            # close position
            ai_order = exchange.create_order('AI16Z-USDT-SWAP', 'market', 'buy', entry_ai_amount)
            vr_order = exchange.create_order('VIRTUAL-USDT-SWAP', 'market', 'sell', entry_vr_amount )
            # Calculate PNL (short AI, long VR)
            ai_pnl = entry_ai_amount * (ai_entry_price - ai_current_price)  # Note: short position
            vr_pnl = entry_vr_amount * (vr_current_price - vr_entry_price)
            PNL += ai_pnl + vr_pnl

            # No position
            ai_position = 0
            virtual_position = 0

    return ai_position, virtual_position, PNL, is_exit, entry_time,entry_ai_amount,entry_vr_amount

def main():
    Gross_PNL = 0
    base_symbol = 'VIRTUAL-USDT-SWAP'
    target_symbol = 'AI16Z-USDT-SWAP'
    print("start signal generation")
    # Load environment variables
    api_key = os.environ.get('OKEX_KEY')
    secret = os.environ.get('OKEX_SECRET')
    exchange = ccxt.okx({
                "apiKey": api_key, 
                "secret": secret,
            "password":"tradingPa$$word23"
            })
    exchange.set_leverage(symbol = 'AI16Z/USDT:USDT',leverage =  10)
    exchange.set_leverage(symbol='VIRTUAL/USDT:USDT', leverage = 10)
    entry_ai_amount = 0
    entry_vr_amount = 0
    ai_position = 0
    vr_position = 0
    Gross_PNL = 0
    path = 'C:\\Users\\theo\\Desktop\\Astra-folder\\virtual_ai16_analysis\\'
    while True:
        try:
            fetch_historical_data(exchange, 'AI16Z/USDT:USDT')
            fetch_historical_data(exchange, 'VIRTUAL/USDT:USDT')
            data_ai16Z = pd.read_csv(path + 'ai16z-1min.csv')
            data_virtual = pd.read_csv(path + 'virtual-1min.csv')
            #get_bound
            lower_bound,upper_bound,correlation,ratio = get_bound()
            range = f"{lower_bound:.4f} - {upper_bound:.4f}"

            position = {'ai16z': ai_position, 'virtual': vr_position}
            #generate signal
            Gross_PNL,order_ai_info,order_vr_info,reversed_order_ai_info,reversed_order_vr_info = exit_position(
               exchange,spread,beta, ai_position, vr_position, Gross_PNL, upper_bound, lower_bound, mean, current_time, entry_time, entry_ai_amount,entry_vr_amount,data_ai, data_vr)

            time.sleep(calculate_waiting_time())
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            print(traceback.format_exc())

if __name__ == '__main__':
    asyncio.run(main())
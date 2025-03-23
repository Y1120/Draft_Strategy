import ccxt
import os
import pandas as pd
import time
from dotenv import load_dotenv
from telegram_alarm import TelegramNotifier, fetch_data
import asyncio
from pprint import pprint
import json
import traceback
import numpy as np
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from statsmodels.tsa.stattools import coint
import sys
sys.path.append(os.path.join(r'C:\Users\theo\Desktop\Astra-folder\virtual_ai16_analysis'))
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(env_path)
def calculate_waiting_time():
    """Calculate seconds remaining until next minute"""
    now = datetime.now()
    seconds = now.second
    # If seconds is 0, we're at the start of a minute
    if seconds == 0:
        return 0
    # Calculate remaining seconds until next minute
    wait_seconds = 60 - seconds
    print(f"Waiting {wait_seconds} seconds until next minute")
    return wait_seconds

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
                until = since + 300 * 1 * 60 * 1000
                params = {'until': until}
                
                candles = exchange.fetch_ohlcv(
                    symbol,
                    timeframe='1m',
                    since=since,
                    limit=300,
                    params=params
                )
                
                if not candles:
                    print("No more data to fetch")
                    break
                    
                data_list.extend(candles)
                since = until
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

def get_amount(exchange,symbol):
    notion = 10
    try:
        last_trade = exchange.fetch_ticker(symbol)
        last_trade_price = last_trade['last']
        amount = round(notion/last_trade_price,2)
        return amount
    except Exception as e:
        print(f"Error getting amount: {e}")
        return 0

def fetch_position(exchange, symbol):
    try:
        # Parameters can include instType for the type of market (e.g., 'SWAP', 'SPOT')
        position = exchange.fetch_position(symbol)
        position = json.loads(json.dumps(position))
        if position is not None:
            if float(position["info"]["pos"]) != 0:
                side = "long" if float(position['info']['pos']) > 0 else "short"
                amount =float(position["info"]['pos'])
                entry_price = float(position["info"]['avgPx'])
                unrealised_pnl = float(position["info"]['upl'])
                print("side",side,"amount",amount,"entry_price",entry_price,"unrealised_pnl",unrealised_pnl)
            else:
                side = 0
                amount = 0
                entry_price = 0
                unrealised_pnl = 0

        elif position is None:
            side = 0
            amount = 0
            entry_price = 0
            unrealised_pnl = 0
        
        return [symbol,side,amount,entry_price,unrealised_pnl]
    except Exception as e:
        print(f"Error fetching position for {symbol}: {str(e)}")
        return []

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

def calculate_ratio_stats(data_ai16Z, data_virtual):
    try:
        window_size = min(len(data_ai16Z), len(data_virtual))
        data_ai16Z = data_ai16Z.iloc[-window_size:]
        data_virtual = data_virtual.iloc[-window_size:]

        # Calculate ratio as numeric Series
        ratio = pd.Series(
            data_ai16Z['close'] / data_virtual['close'],
            index=data_ai16Z.index,
            dtype=float
        )

        # Replace infinities and clean data
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Calculate stats from cleaned numeric data
        rolling_mean = float(ratio.mean())
        rolling_std = float(ratio.std())
        current_ratio = float(ratio.iloc[-1])

        print(f"Rolling mean: {rolling_mean:.4f}")
        print(f"Rolling std: {rolling_std:.4f}")
        print(f"Current ratio: {current_ratio:.4f}")

        return rolling_mean, rolling_std, current_ratio

    except Exception as e:
        print(f"Error in ratio calculation: {e}")

def enter_position(ratio,exchange,amount_ai,amount_vr,ai_position,virtual_position,PNL,upper_bound,lower_bound):
    target_symbol = 'AI16Z-USDT-SWAP'
    base_symbol = 'VIRTUAL-USDT-SWAP'
    is_exit = False
    reversed_order_ai = None
    reversed_order_vr = None
    is_reversed = False
    order_ai = None
    order_vr = None
    if ai_position[1] == 0 and virtual_position[1] == 0:
        print("no position found.")
        if ratio >= lower_bound and ratio <= upper_bound:
            print("no position found, no action taken")
        elif ratio < lower_bound:
            order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'buy', amount = amount_ai)
            order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'sell', amount = amount_vr)
            order_ai_info = process_order(exchange,order_ai,symbol = target_symbol)
            order_vr_info = process_order(exchange,order_vr,symbol = base_symbol)
        elif ratio > upper_bound:
            order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'sell', amount = amount_ai)
            order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'buy', amount = amount_vr)
            order_ai_info = process_order(exchange,order_ai,symbol = target_symbol)
            order_vr_info = process_order(exchange,order_vr,symbol = base_symbol)
        else:
            print("Error in creating position")
        
    elif ai_position[1] == 'long' and virtual_position[1] == 'short':
        if ratio > upper_bound:
            is_exit = True
            print("signal reverse, close position and create another one")
            order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'sell', amount = amount_ai)
            order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'buy', amount = amount_vr)
            order_ai_info = process_order(exchange,order_ai,symbol = target_symbol)
            order_vr_info = process_order(exchange,order_vr,symbol = base_symbol)
            PNL += amount_ai * (order_ai_info[2] - ai_position[3]) + amount_vr * (virtual_position[3] - order_vr_info[2])
            # position created
            reversed_order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'sell', amount = amount_ai)
            reversed_order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'buy', amount = amount_vr)
           
        elif ratio < lower_bound:
            print("position not close yet, no action taken")
        elif ratio >= lower_bound and ratio <= upper_bound:
            is_exit = True
            print("go back to normal situation,close position")
            order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'sell', amount = amount_ai)
            order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'buy', amount = amount_vr)
            order_ai_info = process_order(exchange,order_ai,symbol = target_symbol)
            order_vr_info = process_order(exchange,order_vr,symbol = base_symbol)
            PNL += amount_ai * (order_ai_info[2] - ai_position[3]) + amount_vr * (virtual_position[3] - order_vr_info[2])
        else:
            print("signal not strong enough, no action taken")

    elif ai_position[1] == 'short' and virtual_position[1] == 'long':
        if ratio < lower_bound:
            is_exit = True
            print("signal reverse, close position and create another one")
            order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'buy', amount = amount_ai)
            order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'sell', amount = amount_vr)
            order_ai_info = process_order(exchange,order_ai,symbol = target_symbol)
            order_vr_info = process_order(exchange,order_vr,symbol = base_symbol)
            PNL += amount_ai * (ai_position[3] - order_ai_info[2] ) + amount_vr * (order_vr_info[2] - virtual_position[3])
            # position created
            reversed_order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'buy', amount = amount_ai)
            reversed_order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'sell', amount = amount_vr)
        elif ratio > upper_bound:
            print("position not close yet, no action taken")
        elif ratio >= lower_bound and ratio <= upper_bound:
            is_exit = True
            print("go back to normal situation, close position")
            order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'buy', amount = amount_ai)
            order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'sell', amount = amount_vr)
            order_ai_info = process_order(exchange,order_ai,symbol = target_symbol)
            order_vr_info = process_order(exchange,order_vr,symbol = base_symbol)
            PNL +=  amount_ai * (ai_position[3] - order_ai_info[2]) + amount_vr * (order_vr_info[2] - virtual_position[3])
        else:
            print("signal not strong enough, no action taken")
    
    else:
        print("no position found. Error in exit_position")

    if order_ai is not None and order_vr is not None:
        return PNL,order_ai_info,order_vr_info,reversed_order_ai_info,reversed_order_vr_info
        
    else:
        return PNL,None,None,None,None

def get_bound():
    try:
        # Load CSV files
        path = 'C:\\Users\\theo\\Desktop\\Astra-folder\\virtual_ai16_analysis'
        data_ai16Z = pd.read_csv(path + '\\ai16z-1min.csv')
        data_virtual = pd.read_csv(path + '\\virtual-1min.csv')
        
        # Convert timestamp columns to datetime with flexible parsing
        data_ai16Z['timestamp'] = pd.to_datetime(data_ai16Z['timestamp'], format='mixed')
        data_virtual['timestamp'] = pd.to_datetime(data_virtual['timestamp'], format='mixed')
        
        # Set timestamp as index and sort
        data_ai16Z.set_index('timestamp', inplace=True)
        data_virtual.set_index('timestamp', inplace=True)
        data_ai16Z.sort_index(inplace=True)
        data_virtual.sort_index(inplace=True)
        print(len(data_ai16Z))
        print(len(data_virtual))
        # Create DataFrame with proper datetime index
        data = pd.DataFrame(index=data_ai16Z.index)
        
        # Calculate ratio
        data["ratio"] = data_ai16Z['close'] / data_virtual['close']
        data.dropna(inplace=True)
        
        # Filter last month's data
        current_date = pd.Timestamp.now()
        one_month_ago = current_date - pd.DateOffset(months=1)
        
        # Filter data using boolean indexing
        common_index = data_ai16Z.index.intersection(data_virtual.index)
        data_ai16Z = data_ai16Z.loc[common_index]
        data_virtual = data_virtual.loc[common_index]

        filtered_ai16Z = data_ai16Z.loc[data_ai16Z.index >= one_month_ago].copy()
        print("filtered",len(filtered_ai16Z))
        filtered_virtual = data_virtual.loc[data_virtual.index >= one_month_ago].copy()
        

        # Calculate correlation on filtered data
        correlation = filtered_ai16Z['close'].corr(filtered_virtual['close'])
        
        # Calculate statistics
        rolling_mean, rolling_std, ratio = calculate_ratio_stats(filtered_ai16Z, filtered_virtual)

        # Calculate bounds
        lower_bound = rolling_mean - 1.25 * rolling_std
        upper_bound = rolling_mean + 1.25 * rolling_std
        
        print(f"Lower bound: {lower_bound:.4f}")
        print(f"Upper bound: {upper_bound:.4f}")
        
        return lower_bound, upper_bound, correlation, ratio

    except Exception as e:
        print(f"Error in get_bound: {e}")
        print(traceback.format_exc())
        return -1, 1, 0, 0

async def main():
    Gross_PNL = 0
    telegram = TelegramNotifier(
        bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
        default_chat_id=os.getenv('TELEGRAM_CHAT_ID')
    )
    print(os.getenv('TELEGRAM_BOT_TOKEN'))
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
    amount_ai = get_amount(exchange,target_symbol)
    amount_vr = get_amount(exchange,base_symbol)
    path = 'C:\\Users\\theo\\Desktop\\Astra-folder\\virtual_ai16_analysis\\'
    while True:
        try:
            exchange.set_leverage(symbol = 'AI16Z/USDT:USDT',leverage =  10)
            exchange.set_leverage(symbol='VIRTUAL/USDT:USDT', leverage = 10)
            fetch_historical_data(exchange, 'AI16Z/USDT:USDT')
            fetch_historical_data(exchange, 'VIRTUAL/USDT:USDT')
            data_ai16Z = pd.read_csv(path + 'ai16z-1min.csv')
            data_virtual = pd.read_csv(path + 'virtual-1min.csv')
            #get_bound
            lower_bound,upper_bound,correlation,ratio = get_bound()
            range = f"{lower_bound:.4f} - {upper_bound:.4f}"
            ai_position = fetch_position(exchange, 'AI16Z/USDT:USDT')
            virtual_position = fetch_position(exchange, 'VIRTUAL/USDT:USDT')
            position = {'ai16z': ai_position, 'virtual': virtual_position}
            #generate signal
            Gross_PNL,order_ai_info,order_vr_info,reversed_order_ai_info,reversed_order_vr_info = enter_position(
                ratio,exchange,amount_ai,amount_vr,ai_position,virtual_position,Gross_PNL,upper_bound,lower_bound)

            await fetch_data(
                    telegram=telegram,
                    virtual_order=order_vr_info,
                    ai_order=order_ai_info,
                    reversed_order_ai=reversed_order_ai_info,
                    reversed_order_vr=reversed_order_vr_info,
                    position=position,
                    gross_pnl=Gross_PNL, 
                    correlation=correlation,
                    range = range,
                    ratio = ratio
                )
            time.sleep(calculate_waiting_time())
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            print(traceback.format_exc())

if __name__ == '__main__':
    asyncio.run(main())
        

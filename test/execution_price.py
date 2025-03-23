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

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(env_path)
sys.path.append(os.path.join("C:/Users/theo/Desktop/Astra-folder/"))
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
    symbol1 = symbol.replace("/USDT:USDT","").lower()
    print(f"Fetching historical data for {symbol1}...")
    data = pd.read_csv( f'"C:\Users\theo\Desktop\Astra-folder\{symbol1}-1min.csv', parse_dates=['timestamp'], index_col='timestamp')
    print('index',data.index[-1])
    start_dt = pd.to_datetime(data.index[-1]) + pd.Timedelta(minutes=1)

    end_dt  = datetime.now()+ pd.Timedelta(minutes=1)
    print('end_dt',end_dt)
    since = int(start_dt.timestamp() * 1000)
    print('since',since)
    end = int(end_dt.timestamp() * 1000)
    print('end',end)
    data_list = []
    while since < end:
        try: 
            until = since + 300 * 1 * 60 * 1000
            params = {
                'until': until
            }
         # Fetch batch of candles
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe='1m',
                since=since,
                limit=300,
                params = params
            )
            
            if not candles:
                print("No more data to fetch")
                break
                
            # Add to data list
            data_list.extend(candles)
            
            # Update since timestamp for next batch
            since = until
            print('sice',since)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    if not data_list:
        print("No data fetched")
        return None
    else:
        data_list = pd.DataFrame(data_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data_list['timestamp'] = pd.to_datetime(data_list['timestamp'], unit='ms')
        data_list.sort_values('timestamp', inplace=True)
        print(data_list.iloc[0])
        print(data_list.iloc[-1])
        data_list.to_csv(f'{symbol1}-1min.csv', mode='a', header=False,index=False)
        print(f"Data saved to {symbol}-1min.csv")
        return data_list['close'].iloc[-1]

def get_lastet_close_price(exchange,symbol):
    symbol1 = symbol.replace("/USDT:USDT","").lower()
    data = pd.read_csv( f'{symbol1}-1min.csv', parse_dates=['timestamp'], index_col='timestamp')
    lastet_close_price = data['close'].iloc[-1]
    return lastet_close_price

def get_amount(exchange,symbol):
    notion = 10
    try:
        last_trade = exchange.fetch_ticker(symbol)
        last_trade_price = last_trade['last']
        amount = round(notion/last_trade_price,2)
        print(f"amount for {symbol} : {amount}")
        return amount
    except Exception as e:
        print(f"Error getting amount: {e}")
        print(f"amount for {symbol} : {amount}")
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

def process_order(exchange,order):
    order_status = exchange.fetch_order(order['id'], order['symbol'])
    order_info = [order_status['symbol'],order_status['side'],float(order_status['price']),float(order_status['amount']),order_status['status']]
    return order_info

def calculate_ratio_stats(data_ai16Z, data_virtual):
    try:
        window_size = min(len(data_ai16Z), len(data_virtual))
        data_ai16Z = data_ai16Z.iloc[-window_size:]
        data_virtual = data_virtual.iloc[-window_size:]

        ai16z_returns = data_ai16Z['close']
        virtual_returns = data_virtual['close']

        # Calculate ratio as numeric Series
        ratio = pd.Series(
            ai16z_returns.values / virtual_returns.values,
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

def calculate_lastest_ratio(start_price_ai,start_price_virtual,now_price_ai,now_price_virtual):
    lastest_ratio = now_price_ai / now_price_virtual
    return lastest_ratio

def exit_position(latest_ratio,exchange,ai_position,virtual_position,PNL,upper_bound,lower_bound):
    ratio = latest_ratio
    target_symbol = 'AI16Z-USDT-SWAP'
    base_symbol = 'VIRTUAL-USDT-SWAP'
    amount_ai = get_amount(exchange,target_symbol)
    amount_vr = get_amount(exchange,base_symbol)
    is_exit = False
    reversed_order_ai = None
    reversed_order_vr = None
    if ai_position[1] == 'long' and virtual_position[1] == 'short':
        if ratio > upper_bound:
            is_exit = True
            print("signal reverse, close position and create another one")
            order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'sell', amount = amount_ai)
            order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'buy', amount = amount_vr)
            order_ai_info = process_order(exchange,order_ai)
            order_vr_info = process_order(exchange,order_vr)
            PNL += amount_ai * (order_ai_info[2] - ai_position[3]) + amount_vr * (order_vr_info[2] - virtual_position[3])
            # position created
            reversed_order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'sell', amount = amount_ai)
            reversed_order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'buy', amount = amount_vr)
        elif ratio < lower_bound:
            print("position not close yet, no action taken")
        elif ratio >= lower_bound and ratio <= upper_bound:
            is_exit = True
            print("go back to normal situation, close position and create another one")
            order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'sell', amount = amount_ai)
            order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'buy', amount = amount_vr)
            order_ai_info = process_order(exchange,order_ai)
            order_vr_info = process_order(exchange,order_vr)
            PNL += amount_ai * (order_ai_info[2] - ai_position[3]) + amount_vr * (virtual_position[3] - order_vr_info[2])
        else:
            print("signal not strong enough, no action taken")

    elif ai_position[1] == 'short' and virtual_position[1] == 'long':
        if ratio < lower_bound:
            print("signal reverse, close position and create another one")
            order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'buy', amount = amount_ai)
            order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'sell', amount = amount_vr)
            order_ai_info = process_order(exchange,order_ai)
            order_vr_info = process_order(exchange,order_vr)
            PNL += amount_ai * (ai_position[3] - order_ai_info[2] ) + amount_vr * (order_vr_info[2] - virtual_position[3])
            # position created
            reversed_order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'buy', amount = amount_ai)
            reversed_order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'sell', amount = amount_vr)
        elif ratio > upper_bound:
            print("position not close yet, no action taken")
        elif ratio >= lower_bound and ratio <= upper_bound:
            print("go back to normal situation, close position and create another one")
            order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'buy', amount = amount_ai)
            order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'sell', amount = amount_vr)
            order_ai_info = process_order(exchange,order_ai)
            order_vr_info = process_order(exchange,order_vr)
            PNL +=  amount_ai * (ai_position[3] - order_ai_info[2]) + amount_vr * (order_vr_info[2] - virtual_position[3])
        else:
            print("signal not strong enough, no action taken")
    
    else:
        print("no position found. Error in exit_position")

    if order_ai is not None and order_vr is not None:
        if reversed_order_ai is not None:
            reversed_order_ai_info = process_order(exchange,reversed_order_ai)
            reversed_order_vr_info = process_order(exchange,reversed_order_vr)
            return PNL,is_exit,order_ai_info,order_vr_info,reversed_order_ai_info,reversed_order_vr_info
        elif reversed_order_ai is None:
            return PNL,is_exit,order_ai_info,order_vr_info,None,None
    else:
        return PNL,is_exit,None,None,None,None
    
def enter_position(ratio,exchange,ai_position,virtual_position,upper_bound,lower_bound):
    is_signal = False
    order_ai = None
    order_vr = None
    target_symbol = 'AI16Z-USDT-SWAP'
    base_symbol = 'VIRTUAL-USDT-SWAP'
    amount_ai = get_amount(exchange,target_symbol)
    amount_vr = get_amount(exchange,base_symbol)
    if ai_position[1] == 0 and virtual_position[1] == 0:
        print("no position found.")
        if ratio < lower_bound:
            is_signal = True
            print("signal generated, create position")
            order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'buy', amount = amount_ai)
            order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'sell', amount = amount_vr)
        elif ratio > upper_bound:
            is_signal = True
            print("signal generated, create position")
            order_ai = exchange.create_order(symbol =  target_symbol, type = 'market', side = 'sell', amount = amount_ai)
            order_vr = exchange.create_order(symbol = base_symbol, type = 'market', side = 'buy', amount = amount_vr)
        elif ratio >= lower_bound and ratio <= upper_bound:
            is_signal = False
            print("keeping on the normal range, no action taken")
        else:
            print("signal not strong enough, no action taken")
    return order_ai,order_vr,is_signal

def get_bound():
    data_ai16Z = pd.read_csv('ai16z-1min.csv', parse_dates=['timestamp'], index_col='timestamp')
    data_virtual = pd.read_csv('virtual-1min.csv', parse_dates=['timestamp'], index_col='timestamp')
    data = pd.DataFrame(index = data_ai16Z.index)
    # get ratio
    data["ratio"]   =  data_ai16Z['close']/data_virtual['close']
    data.dropna(inplace=True)
    current_date = pd.Timestamp.now()
    one_month_ago = current_date - pd.DateOffset(months=1)
    data['ratio'] = data['ratio'].loc[one_month_ago:]

    # Calculate correlation between series
    correlation = data_ai16Z['close'].loc[one_month_ago:].corr(data_virtual['close'].loc[one_month_ago:])
   #print('Correlation:', correlation)
    # get z-score
    rolling_mean,rolling_std,ratio  = calculate_ratio_stats(data_ai16Z, data_virtual)

    lower_bound = rolling_mean - 1.25*rolling_std
    upper_bound = rolling_mean + 1.25*rolling_std
    print("lower bound:", lower_bound)
    print("upper bound:", upper_bound)
    print("current ratio:",ratio)
    return lower_bound,upper_bound,correlation,ratio

async def main():
    Gross_PNL = 0
    telegram = TelegramNotifier(
        bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
        default_chat_id=os.getenv('TELEGRAM_CHAT_ID')
    )
    print(os.getenv('TELEGRAM_BOT_TOKEN'))
    
    while True:
        try:
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
            start_price_ai = fetch_historical_data(exchange, 'AI16Z/USDT:USDT')
            start_price_virtual = fetch_historical_data(exchange,'VIRTUAL/USDT:USDT')
            #get_bound
            lower_bound,upper_bound,correlation,ratio = get_bound()
            range = f"{lower_bound:.4f} - {upper_bound:.4f}"
            ai_position = fetch_position(exchange, 'AI16Z/USDT:USDT')
            virtual_position = fetch_position(exchange, 'VIRTUAL/USDT:USDT')
            #generate signal
            order_ai_created,order_vr_created,is_signal = enter_position(ratio,exchange,ai_position,virtual_position,upper_bound,lower_bound)
            if is_signal is False:
                print("no signal generated")
                await asyncio.sleep(calculate_waiting_time())
                continue
            else:
                await fetch_data(
                    telegram=telegram,
                    virtual_order=order_ai_created,
                    ai_order=order_vr_created,
                    close_order_ai=None,
                    close_order_vr=None,
                    reverse_order_ai=None,
                    reverse_order_vr=None,
                    position=None,
                    gross_pnl=Gross_PNL, 
                    correlation=correlation,
                    range = range,
                    ratio = ratio
                )
                print("signal generated")
                ai_position = fetch_position(exchange, 'AI16Z/USDT:USDT')
                virtual_position = fetch_position(exchange, 'VIRTUAL/USDT:USDT')
                ########################send telegram notification
                #exit position
                while True:
                    await asyncio.sleep(calculate_waiting_time())
                    now_price_ai = fetch_historical_data(exchange, 'AI16Z/USDT:USDT')
                    now_price_virtual = fetch_historical_data(exchange,'VIRTUAL/USDT:USDT')
                    latest_ratio = calculate_lastest_ratio(start_price_ai,start_price_virtual,now_price_ai,now_price_virtual)
                    Gross_PNL,is_exit,order_ai_close,order_vr_close,reversed_order_ai,reversed_order_vr = exit_position(latest_ratio,exchange,ai_position,virtual_position,Gross_PNL,upper_bound,lower_bound)
                    if is_exit is True:
                        ai_position = fetch_position(exchange, 'AI16Z/USDT:USDT')
                        virtual_position = fetch_position(exchange, 'VIRTUAL/USDT:USDT')
                        position = {'ai16z': ai_position, 'virtual': virtual_position}
                        ratio = latest_ratio
                        await fetch_data(
                                telegram=telegram,
                                virtual_order=order_ai_created,
                                ai_order=order_vr_created,
                                close_order_vr = order_ai_close,
                                close_order_ai = order_vr_close,
                                reversed_order_ai = reversed_order_ai,
                                reversed_order_vr = reversed_order_vr,
                                position=position,
                                gross_pnl=Gross_PNL,
                                correlation=correlation,
                                range = range,
                                ratio = ratio
                            )
                        break
                    else:
                        continue            
        except Exception as e:
            print(f"Error in main loop: {e}")
            print(traceback.format_exc())

if __name__ == '__main__':
    asyncio.run(main())
        

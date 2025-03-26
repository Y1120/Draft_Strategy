# add much interval to calculate beta
from dotenv import load_dotenv
import os
import ccxt.async_support as ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
from datetime import datetime
import traceback
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller
from calculate_beta import calculate_beta_with_rolling_windows
import asyncio
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()
import sys
import requests
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()
async def run_in_executor(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

async def process_order(exchange, order, symbol):
    try:
        # Handle case where order is a list
        order_status = await exchange.fetch_order(order['id'], symbol)

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

async def fetch_historical_data(exchange, symbol_x):
    try:
       
        print(f"Fetching historical data for {symbol_x}...")
        path = 'C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\1m\\'
        file_path = f'{path}{symbol_x}_1m.csv'
        data_x_temp = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        start_dt = pd.to_datetime(data_x_temp.index[-1]) + pd.Timedelta(minutes=1)
     
        end_dt = datetime.datetime.now() + pd.Timedelta(minutes=1)
        since = int(start_dt.timestamp() * 1000)
        end = int(end_dt.timestamp() * 1000)
        data_list_x = []

        while since < end:
            try:
                candles = await exchange.fetch_ohlcv(symbol_x, '1m', since, 100)
                if not candles:
                    break
                print(candles)
                data_list_x.extend(candles)
                last_candle = candles[-1]
                since = last_candle[0] + 60 * 1000  # Next minute
            except Exception as e:
                print(f"Error fetching data: {e}")
                break



        df_x = pd.DataFrame(data_list_x, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_x['timestamp'] = pd.to_datetime(df_x['timestamp'], unit='ms')
        df_x.to_csv(f'{path}{symbol_x}_1m.csv', mode='a', header=False, index=False)
        print(f"Data saved to {symbol_x}_1m.csv")

    except Exception as e:
        print(f"Error fetching data: {e}")
async def generate_merge_data(symbol_x,symbol_y):
    path = 'C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\1m\\'
    file_path1 = f'{path}{symbol_x}_1m.csv'
    file_path2 = f'{path}{symbol_y}_1m.csv'
    data_x = pd.read_csv(file_path1)
    data_y = pd.read_csv(file_path2)
  
    data_x['timestamp'] = pd.to_datetime(data_x['timestamp'])
    data_y['timestamp'] = pd.to_datetime(data_y['timestamp'])
    data_ai_subset = data_x[['timestamp', 'close']].rename(columns={'close': 'ai16z'})
    data_vr_subset = data_y[['timestamp', 'close']].rename(columns={'close': 'virtual'})
    merage_data = pd.merge(data_ai_subset, data_vr_subset, on='timestamp', how='inner')
    merage_data = merage_data.dropna()
    merage_data = calculate_beta_with_rolling_windows(merage_data)
    merage_data.to_csv(f'C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\1m\\{symbol_x}-{symbol_y}-merge.csv')

async def retry_async(func, *args, max_retries=3, initial_delay=1, **kwargs):
    """Retry an async function with exponential backoff on certain exceptions."""
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except (ccxt.RequestTimeout, ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            last_exception = e
            if attempt < max_retries - 1:
                sleep_time = delay * (2 ** attempt)  # Exponential backoff
                print(f"API request failed: {str(e)}. Retrying in {sleep_time} seconds... (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(sleep_time)
            else:
                print(f"Max retries reached. Last error: {str(e)}")
                raise
    
    # If we get here, all retries failed
    raise last_exception

async def fetch_merge_data(exchange,symbol_x,symbol_y):
    merge_file_path = f'C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\1m\\{symbol_x}-{symbol_y}-merge.csv'
    if not os.path.exists(merge_file_path):
        await generate_merge_data(symbol_x, symbol_y)
    merage_data = pd.read_csv(f'C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\1m\\{symbol_x}-{symbol_y}-merge.csv', parse_dates=['timestamp'])
    data_x = pd.read_csv(f"C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\1m\\{symbol_x}_1m.csv",parse_dates=['timestamp']).sort_values('timestamp')
    data_y = pd.read_csv(f"C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\1m\\{symbol_y}_1m.csv",parse_dates=['timestamp']).sort_values('timestamp')
  
    data_x['timestamp'] = pd.to_datetime(data_x['timestamp'],format='mixed')
    data_y['timestamp'] = pd.to_datetime(data_y['timestamp'],format='mixed')
  
    start_dt = pd.to_datetime(merage_data['timestamp'].iloc[-1],format='mixed')
    data_x_start_dt = start_dt - timedelta(days = 31)
    data_x = data_x[data_x['timestamp'] > data_x_start_dt]
    data_y = data_y[data_y['timestamp'] > data_x_start_dt]
   

    result_all = []
    for end in data_x['timestamp']:
        end = pd.to_datetime(end,format="mixed")

        if end <= start_dt:
        
            continue
        elif end>start_dt:
            
            start_index = end - timedelta(days= 30)
            windows_x = data_x[(data_x['timestamp'] >= start_index) & (data_x['timestamp'] <= end)]
            windows_y = data_y[(data_y["timestamp"] >= start_index) & (data_y["timestamp"] <= end)]
            windows_x_close = pd.to_numeric(windows_x['close'], errors='coerce').to_numpy()
            windows_y_close = pd.to_numeric(windows_y['close'],errors='coerce').to_numpy()

            X = sm.add_constant(np.log(windows_x_close))
            y = np.log(windows_y_close)
    
            while True:
                if len(X) != len(y):
                    # Replace time.sleep with asyncio.sleep for async functions
                    await asyncio.sleep(5)
                    # Add await keyword to call async functions properly
                    await fetch_historical_data(exchange, symbol_x)
                    await fetch_historical_data(exchange, symbol_y)
                    continue
                else:
                    break
             # Fit OLS model

             # Fit GLS model without custom covariance
            gls_model = sm.GLS(y, X).fit()  # No sigma parameter
            current_beta = gls_model.params[1]
          
            last_X = np.log(windows_x_close[-1])  # Log of the last X value
            last_y = np.log(windows_y_close[-1])  # Log of the last y value
            current_spread = last_y - (last_X * current_beta) -gls_model.params[0]
            print("current_constant",gls_model.params[0])
            results = {
                'timestamp':end,
                symbol_x:last_X,
                symbol_y:last_y,
                "spread":current_spread,
                "beta":current_beta
            }
           
            result_all.append(results)
    result_all = pd.DataFrame(result_all)
    result_all.to_csv(f'C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\1m\\{symbol_x}-{symbol_y}-merge.csv', mode = 'a',header = False,index = False)
    print("merge_data is saved!")

def calculate_waiting_time():
    """Calculate seconds remaining until next minute"""
    now = datetime.datetime.now()
    seconds = now.second
    # If seconds is 0, we're at the start of a minute
    if seconds == 0:
        return 0
    # Calculate remaining seconds until next minute
    wait_seconds = 60 - seconds
    print(f"Waiting {wait_seconds} seconds until next minute")
    return wait_seconds

def get_contract_size(symbol):
    try:
        futures_response = requests.get(
            'https://www.okx.com/api/v5/public/instruments',
            params={'instType': 'SWAP'}
        )
        futures_data = futures_response.json()
        contract_size = [
            item['ctVal']
            for item in futures_data['data'] if item['instId'] == symbol
        ]
        print(symbol,int(contract_size[0]))
        return int(contract_size[0])
            
    except Exception as e:
        print(f"Error getting symbols: {e}")
        return None
class Pairs_trade:
    # pattern of symbol : "BTC-USDT-SWAP"
    def __init__(self,exchange,symbol_x,symbol_y):
        self.symbol_x = symbol_x
        self.symbol_y = symbol_y
        self.contract_size_x = get_contract_size(symbol_x)
        self.contract_size_y = get_contract_size(symbol_y)
        self.data_x = None
        self.data_y = None
        self.position_x = 0
        self.position_y = 0
        self.x_current_price = 0
        self.y_current_price = 0
        self.current_spread = 0
        self.current_beta = 0
        self.current_time = None
        self.entry_amount_x = 0
        self.entry_amount_y = 0
        self.entry_price_x = 0
        self.entry_price_y = 0
        self.entry_time = None
        self.orders = []
        self.upper_bound = 0
        self.lower_bound = 0
        self.mean = 0
        self.PNL = 0
        self.exchange = exchange
        self.is_execute = False
        self.unrealised_pnl = 0
    
    async def get_data(self):
        path = 'C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\1m\\'
        symbol1 = self.symbol_x
        symbol2 = self.symbol_y
        self.data_x = pd.read_csv(f'{path}{symbol1}_1m.csv', parse_dates=['timestamp'])
        self.data_y = pd.read_csv(f'{path}{symbol2}_1m.csv', parse_dates=['timestamp'])

    async def calculate_spread(self):
    
        path = 'C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\1m\\'
        look_back_time = 30
        now = datetime.datetime.now()
        start_time = now - timedelta(days=look_back_time)
        merged_data = pd.read_csv(f'{path}{self.symbol_x}-{self.symbol_y}-merge.csv', parse_dates=['timestamp'])
        merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'],format='mixed')
        
        merged_data_copy = merged_data[(merged_data['timestamp'] >= start_time)]
       
        self.current_beta = merged_data_copy["beta"].iloc[-1]
        self.current_spread = merged_data_copy["spread"].iloc[-1]
        self.mean = merged_data_copy['spread'].mean()
        self.upper_bound = self.mean + 1 * merged_data_copy['spread'].std()
        self.lower_bound = self.mean - 1 * merged_data_copy['spread'].std()
        print("current beta",self.current_beta)

        print("current_spread",self.current_spread)
        print("mean",self.mean)
        print("upper_bound",self.upper_bound)
        print("lower_bound",self.lower_bound)

    async def stop_loss(self):
        # when the unrealised pnl is less than 20% 
        if self.position_x == 0 and self.position_y == 0:
            print("no position")
            is_stop = False
            return
        elif self.position_x == 1:
            x_current_price = self.data_x['close'].iloc[-1]
            y_current_price = self.data_y['close'].iloc[-1]
            self.unrealised_pnl = self.entry_amount_x * (x_current_price - self.entry_price_x) + self.entry_amount_y * (self.entry_price_y - y_current_price)
            if self.unrealised_pnl < -0.2 * self.PNL:
                print("stop loss")
                is_stop = True
                return is_stop
            else:
                is_stop = False
                return is_stop
        elif self.position_x == -1:
            x_current_price = self.data_x['close'].iloc[-1]
            y_current_price = self.data_y['close'].iloc[-1]
            self.unrealised_pnl = self.entry_amount_x * (self.entry_price_x - x_current_price) + self.entry_amount_y * (y_current_price - self.entry_price_y)
            if self.unrealised_pnl < -0.2 * self.PNL:
                print("stop loss")
                is_stop = True
                return is_stop
            else:
                is_stop = False
                return is_stop
            
            
    async def exit_position(self):
        self.is_execute = False

        # Find current prices
        # No position case
        self.current_time = self.data_x['timestamp'].iloc[-1]
        print("current_time",self.current_time)
        if self.position_x == 0 and self.position_y == 0:
            if self.current_spread >= self.lower_bound and self.current_spread <=self.upper_bound:
                # No position, and ratio is within bounds - do nothing
                pass
            elif self.current_spread < self.lower_bound:
                print(f"spread: {self.current_spread};lower_bound:{self.lower_bound};")
                print(f"buy {self.symbol_x} sell {self.symbol_y} , create_position!")
                self.is_execute = True
                # Ratio below lower bound - go long x, short y
                self.entry_time = self.current_time
                price_x = self.data_x.loc[self.data_x['timestamp'] == self.current_time, 'close'].iloc[0]
                price_y = self.data_y.loc[self.data_y['timestamp'] == self.current_time, 'close'].iloc[0]
                self.position_x = -1
                self.position_y = 1
                amount_x  = (10/price_x)/self.contract_size_x
                amount_y = abs((10 * self.current_beta/price_y)/self.contract_size_y)
                print("price_x",price_x)
                print("price_y",price_y)
                print("amount_x",amount_x)
                print("amount_y",amount_y)
                
                #order_x = await self.exchange.create_order(self.symbol_x, 'market', 'sell', amount_x)
                #order_y = await self.exchange.create_order(self.symbol_y, 'market', 'buy', amount_y)
                order_x = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_x, 'market', 'sell', amount_x,
                            max_retries=5)
                order_y = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_y, 'market', 'buy', amount_y,
                            max_retries=5
                        ) 

                order_x_info =await process_order(self.exchange,order_x,self.symbol_x)
                order_y_info = await process_order(self.exchange,order_y,self.symbol_y)
                self.entry_price_x = order_x_info[2]
                self.entry_price_y = order_y_info[2]
                self.entry_amount_x = order_x_info[3]
                self.entry_amount_y = order_y_info[3]
                self.orders.append(order_x_info)
                self.orders.append(order_y_info)
            elif self.current_spread > self.upper_bound:
                print(f"spread: {self.current_spread};upper_bound:{self.upper_bound};")
                print(f"sell {self.symbol_x} buy {self.symbol_y} , create_position!")
                self.is_execute = True
                # Ratio above upper bound - go short x, long y
                self.position_x = 1
                self.position_y = -1
                self.entry_time = self.current_time
                price_x = self.data_x.loc[self.data_x['timestamp'] == self.current_time, 'close'].iloc[0]
                price_y = self.data_y.loc[self.data_y['timestamp'] == self.current_time, 'close'].iloc[0]
                amount_x  = (10/price_x)/self.contract_size_x
                amount_y = abs(10 * self.current_beta/price_y)/self.contract_size_y


                #order_x = await self.exchange.create_order(self.symbol_x, 'market', 'buy', amount_x)
                #order_y = await self.exchange.create_order(self.symbol_y, 'market', 'sell', amount_y)
                order_x = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_x, 'market', 'buy', amount_x,
                            max_retries=5
                        ) 
                order_y = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_y, 'market', 'sell', amount_y,
                            max_retries=5
                        ) 
                order_x_info = await process_order(self.exchange,order_x,self.symbol_x)
                order_y_info = await process_order(self.exchange,order_y,self.symbol_y)
                self.orders.append(order_x_info)
                self.orders.append(order_y_info)
                self.entry_price_x = order_x_info[2]
                self.entry_price_y = order_y_info[2]
                self.entry_amount_x = order_x_info[3]
                self.entry_amount_y = order_y_info[3]

        # Long AI, Short VR position case
        elif self.position_x == 1 and self.position_y == -1:

            if self.current_spread < self.lower_bound:
                print(f"sell {self.symbol_x} buy {self.symbol_y} , close_position!")
                # Signal reversed - close position and open opposite
                self.is_execute = True

                # close position
                #order_x = await self.exchange.create_order(self.symbol_x, 'market', 'sell', self.entry_amount_x)
                #order_y = await self.exchange.create_order(self.symbol_y, 'market', 'buy', self.entry_amount_y )
                order_x = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_x, 'market', 'sell', self.entry_amount_x,
                            max_retries=5
                        ) 
                order_y = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_y, 'market', 'buy', self.entry_amount_y,
                            max_retries=5
                        ) 
                order_x_info = await process_order(self.exchange,order_x,self.symbol_x)
                order_y_info = await process_order(self.exchange,order_y,self.symbol_y)
                self.orders.append(order_x_info)
                self.orders.append(order_y_info)
                x_current_price = order_x_info[2]
                y_current_price = order_y_info[2]
                x_pnl = self.entry_amount_x * (x_current_price - self.entry_price_x)
                y_pnl = self.entry_amount_y * (self.entry_price_y - y_current_price)  # Note: short position
                self.PNL += x_pnl + y_pnl
                print(f"sell {self.symbol_x} buy {self.symbol_y} , create_position!")

                # Open new position (reverse)
                self.position_x = -1
                self.position_y= 1
                self.entry_time = self.current_time
                amount_x = (10/x_current_price)/self.contract_size_x
                amount_y = abs(10 * self.current_beta/y_current_price)/self.contract_size_y
                #order_x1 = await self.exchange.create_order(self.symbol_x, 'market', 'sell', amount_x)
                #order_y1 = await self.exchange.create_order(self.symbol_y, 'market', 'buy', amount_y)
                order_x1 = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_x, 'market', 'sell', amount_x,
                            max_retries=5
                        ) 
                order_y1 = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_y, 'market', 'buy', amount_y,
                            max_retries=5
                        ) 
                order_x1_info = await process_order(self.exchange,order_x1,self.symbol_x)
                order_y1_info = await process_order(self.exchange,order_y1,self.symbol_y)
                self.orders.append(order_x1_info)
                self.orders.append(order_y1_info)
                self.entry_price_x = order_x1_info[2]
                self.entry_price_y = order_y1_info[2]
                self.entry_amount_x = order_x1_info[3]
                self.entry_amount_y = order_y1_info[3]

            elif self.current_spread < self.mean:
                # Ratio back to normal - close position
                self.is_execute = True
                print(f"sell {self.symbol_x} buy {self.symbol_y} , close_position!")
                # close position
                #order_x = await self.exchange.create_order(self.symbol_x, 'market', 'sell', self.entry_amount_x)
                #order_y = await self.exchange.create_order(self.symbol_y, 'market', 'buy', self.entry_amount_y )
                order_x = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_x, 'market', 'sell', self.entry_amount_x,
                            max_retries=5
                        ) 
                order_y = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_y, 'market', 'buy', self.entry_amount_y,
                            max_retries=5
                        ) 
                order_x_info = await process_order(self.exchange,order_x,self.symbol_x)
                order_y_info = await process_order(self.exchange,order_y,self.symbol_y)
                self.orders.append(order_x_info)
                self.orders.append(order_y_info)
                x_current_price = order_x_info[2]
                y_current_price = order_y_info[2]
                x_pnl = self.entry_amount_x * (order_x_info[2] - self.entry_price_x)
                y_pnl = self.entry_amount_y * (order_y_info[2] - y_current_price)  # Note: short position
                self.PNL += x_pnl + y_pnl
                self.position_x = 0
                self.position_y = 0
                self.entry_amount_x = 0
                self.entry_amount_y = 0
                self.entry_price_x = 0
                self.entry_price_y = 0
                self.entry_time = None

        elif self.position_x == -1 and self.position_y == 1:
            if self.current_spread > self.upper_bound:
                # Signal reversed - close position and open opposite
                print(f"buy {self.symbol_x} sell {self.symbol_y} , close_position!")
                self.is_execute = True

                # close position
                #order_x = await self.exchange.create_order(self.symbol_x, 'market', 'buy', self.entry_amount_x)
                #order_y = await self.exchange.create_order(self.symbol_y, 'market', 'sell', self.entry_amount_y )
                order_x = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_x, 'market', 'buy', self.entry_amount_x,
                            max_retries=5
                        ) 
                order_y = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_y, 'market', 'sell', self.entry_amount_y,
                            max_retries=5
                        ) 
                order_x_info = await process_order(self.exchange,order_x,self.symbol_x)
                order_y_info = await process_order(self.exchange,order_y,self.symbol_y)
                self.orders.append(order_x_info)
                self.orders.append(order_y_info)
                x_pnl = self.entry_amount_x * (self.entry_price_x - order_x_info[2])
                y_pnl = self.entry_amount_y * (order_y_info[2] - self.entry_price_y)  # Note: short position
                self.PNL += x_pnl + y_pnl
                print(f"sell {self.symbol_x} buy {self.symbol_y} , create_position!")

                # Open new position (reverse)
                self.position_x = 1
                self.position_y= -1
                self.entry_time = self.current_time
                amount_x = (10/order_x_info[2])/self.contract_size_x
                amount_y = abs(10 * self.current_beta/order_y_info[2])/self.contract_size_y
                #order_x1 = await self.exchange.create_order(self.symbol_x, 'market', 'buy', amount_x)
                #order_y1 = await self.exchange.create_order(self.symbol_y, 'market', 'sell', amount_y)
                order_x1 = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_x, 'market', 'buy', amount_x,
                            max_retries=5
                        ) 
                order_y1 = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_y, 'market', 'sell', amount_y,
                            max_retries=5
                        ) 
                order_x1_info = await process_order(self.exchange,order_x1,self.symbol_x)
                order_y1_info = await process_order(self.exchange,order_y1,self.symbol_y)
                self.orders.append(order_x1_info)
                self.orders.append(order_y1_info)
                self.entry_price_x = order_x1_info[2]
                self.entry_price_y = order_y1_info[2]
                self.entry_amount_x = order_x1_info[3]
                self.entry_amount_y = order_y1_info[3]

            elif self.current_spread > self.mean:
                # Ratio back to normal - close position
                self.is_execute = True
                # close position
                #order_x = await self.exchange.create_order(self.symbol_x, 'market', 'buy', self.entry_amount_x)
                #order_y = await self.exchange.create_order(self.symbol_y, 'market', 'sell', self.entry_amount_y )
                order_x = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_x, 'market', 'buy', self.entry_amount_x,
                            max_retries=5
                        ) 
                order_y = await retry_async(
                            self.exchange.create_order, 
                            self.symbol_y, 'market', 'sell', self.entry_amount_y,
                            max_retries=5
                        ) 
                order_x_info = await process_order(self.exchange,order_x,self.symbol_x)
                order_y_info = await process_order(self.exchange,order_y,self.symbol_y)
                self.orders.append(order_x_info)
                self.orders.append(order_y_info)
                x_pnl = self.entry_amount_x * (self.entry_price_x - order_x_info[2])
                y_pnl = self.entry_amount_y * (order_y_info[2]- self.entry_price_y)  # Note: short position
                self.PNL += x_pnl + y_pnl
                self.position_x = 0
                self.position_y = 0
                self.entry_amount_x = 0
                self.entry_amount_y = 0
                self.entry_price_x = 0
                self.entry_price_y = 0
                self.entry_time = None

async def process_pair(exchange, symbol_x, symbol_y):
    pairs_trade = Pairs_trade(exchange=exchange, symbol_x=symbol_x, symbol_y=symbol_y)
    beta = []
    count_time = 0
    
    while True:
        now = datetime.datetime.now()
        print(f"**************************** Details for {symbol_x} - {symbol_y} at {now} ***************************")
        
        print("count_time",count_time)
        count_time += 1
        await fetch_historical_data(exchange, symbol_x)
        await fetch_historical_data(exchange, symbol_y)
        if count_time == 30:
            await fetch_merge_data(exchange,symbol_x, symbol_y)
            count_time = 0

        await pairs_trade.get_data()
        await pairs_trade.calculate_spread()
        
        await pairs_trade.exit_position()
        print("PNL", pairs_trade.PNL)
        print("orders", pairs_trade.orders)

        print("entry_amount_x", pairs_trade.entry_amount_x)
        print("entry_price_x", pairs_trade.entry_price_x)
        print("entry_amount_y", pairs_trade.entry_amount_y)
        print("entry_price_y", pairs_trade.entry_price_y)
        print("entry_current_time", pairs_trade.entry_time)
        await asyncio.sleep(calculate_waiting_time())
        print(f"**************************** Details for {symbol_x} - {symbol_y} at {now}  ***************************")

async def main():
   
    pairs = [["DOGE-USDT-SWAP", "FOXY-USDT-SWAP"], ["AEVO-USDT-SWAP", "AIXBT-USDT-SWAP"]]

    exchange = ccxt.okx({
        "apiKey": os.getenv("OKX_KEY"),
        "secret": os.getenv('OKX_SECRET'),
        "password": "tradingPa$$word23"
    })
    
    tasks = []
    for symbol_x, symbol_y in pairs:
        await exchange.set_leverage(symbol=symbol_x, leverage=10)
        await exchange.set_leverage(symbol=symbol_y, leverage=10)
        tasks.append(process_pair(exchange, symbol_x, symbol_y))

    await asyncio.gather(*tasks)
    await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
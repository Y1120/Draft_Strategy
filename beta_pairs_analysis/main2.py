# add much interval to calculate beta
from dotenv import load_dotenv
import os
import ccxt
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
load_dotenv()

async def process_order(exchange, order,symbol):
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
        ########## fetch symbol x data ##########
        symbol1 = symbol_x.replace("-USDT-SWAP","").lower()
        print(f"Fetching historical data for {symbol1}...")
        path = 'C:\\Users\\theo\\Desktop\\Astra-folder\\virtual_ai16_analysis\\'
        file_path = f'{path}{symbol1}-1min.csv'
        data_x_temp = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        start_dt = pd.to_datetime(data_x_temp.index[-1]) + pd.Timedelta(minutes=1)
        print('start fetching data at ',start_dt)
        end_dt = datetime.datetime.now() + pd.Timedelta(minutes=1)
        since = int(start_dt.timestamp() * 1000)
        end = int(end_dt.timestamp() * 1000)
        data_list_x = []
        while since < end:
            try: 
                candles = await exchange.fetch_ohlcv(symbol_x,timeframe='1m',since=since,limit=100)
                if not candles:
                    print("No more data to fetch")
                    break
                print(candles)
                data_list_x.extend(candles)
                last_candle = candles[-1]
                since = last_candle[0] + 60 * 1000 # Next minute
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        if not data_list_x:
            print("No symbol1 data fetched")

        df_x = pd.DataFrame(data_list_x, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_x['timestamp'] = pd.to_datetime(df_x['timestamp'], unit='ms')
        df_x.to_csv(f'{path}{symbol1}-1min.csv', mode='a', header=False, index=False)
        print(f"Data saved to {symbol1}-1min.csv")

    except:
        print('error fetching data1')

async def fetch_merge_data(symbol_x,symbol_y):
    symbol1 = symbol_x.replace("-USDT-SWAP","").lower()
    symbol2 = symbol_y.replace("-USDT-SWAP","").lower()
    merage_data = pd.read_csv(f'C:\\Users\\theo\\Desktop\\Astra-folder\\virtual_ai16_analysis\\{symbol1}-{symbol2}-merge.csv', parse_dates=['timestamp'])

    data_x = pd.read_csv(f"C:\\Users\\theo\\Desktop\\Astra-folder\\virtual_ai16_analysis\\{symbol1}-1min.csv",parse_dates=['timestamp']).sort_values('timestamp')
    data_y = pd.read_csv(f"C:\\Users\\theo\\Desktop\\Astra-folder\\virtual_ai16_analysis\\{symbol2}-1min.csv",parse_dates=['timestamp']).sort_values('timestamp')
  
    data_x['timestamp'] = pd.to_datetime(data_x['timestamp'],format='mixed')
    data_y['timestamp'] = pd.to_datetime(data_y['timestamp'],format='mixed')
    print("type",type(data_x['timestamp'].iloc[-1]))
    start_dt = pd.to_datetime(merage_data['timestamp'].iloc[-1],format='mixed')
    data_x_start_dt = start_dt - timedelta(days = 31)
    data_x = data_x[data_x['timestamp'] > data_x_start_dt]
    data_y = data_y[data_y['timestamp'] > data_x_start_dt]
    print("start",start_dt)

    result_all = []
    for end in data_x['timestamp']:
        end = pd.to_datetime(end,format="mixed")
        print('end',end)
        print('start',start_dt)
        if end <= start_dt:
            print('no')
            continue
        elif end>start_dt:
            print('yes')
            start_index = end - timedelta(days= 30)
            print(start_index,'start_index')
            windows_x = data_x[(data_x['timestamp'] >= start_index) & (data_x['timestamp'] <= end)]
            windows_y = data_y[(data_y["timestamp"] >= start_index) & (data_y["timestamp"] <= end)]
            windows_x_close = pd.to_numeric(windows_x['close'], errors='coerce').to_numpy()
            windows_y_close = pd.to_numeric(windows_y['close'],errors='coerce').to_numpy()
        
            X = sm.add_constant(np.log(windows_x_close))
            y = np.log(windows_y_close)
             # Fit OLS model

             # Fit GLS model without custom covariance
            gls_model = sm.GLS(y, X).fit()  # No sigma parameter
            current_beta = gls_model.params[1]
          
            last_X = np.log(windows_x_close[-1])  # Log of the last X value
            last_y = np.log(windows_y_close[-1])  # Log of the last y value
            current_spread = last_y - (last_X * current_beta) -gls_model.params[0]
            results = {
                'timestamp':end,
                symbol1:last_X,
                symbol2:last_y,
                "spread":current_spread,
                "beta":current_beta
            }
            print("results",results)
            
            result_all.append(results)
    result_all = pd.DataFrame(result_all)
    result_all.to_csv(f'C:\\Users\\theo\\Desktop\\Astra-folder\\virtual_ai16_analysis\\{symbol1}-{symbol2}-merge.csv', mode = 'a',header = False,index=False)
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

class Pairs_trade:
    # pattern of symbol : "BTC-USDT-SWAP"
    def __init__(self,exchange,symbol_x,symbol_y):
        self.symbol_x = symbol_x
        self.symbol_y = symbol_y
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
        self.orders = []
        self.upper_bound = 0
        self.lower_bound = 0
        self.mean = 0
        self.PNL = 0
        self.exchange = exchange
        self.is_execute = False
    
    def get_data(self):
        path = 'C:\\Users\\theo\\Desktop\\Astra-folder\\virtual_ai16_analysis\\'
        symbol1 = self.symbol_x.replace("-USDT-SWAP","").lower()
        symbol2 = self.symbol_y.replace("-USDT-SWAP","").lower()
        self.data_x = pd.read_csv(f'{path}{symbol1}-1min.csv', parse_dates=['timestamp'])
        self.data_y = pd.read_csv(f'{path}{symbol2}-1min.csv', parse_dates=['timestamp'])

    def calculate_spread(self):
        path = 'C:\\Users\\theo\\Desktop\\Astra-folder\\virtual_ai16_analysis\\'
        look_back_time = 30
        now = datetime.datetime.now()
        start_time = now - timedelta(days=look_back_time)
        symbol1 = self.symbol_x.replace("-USDT-SWAP","").lower()
        symbol2 = self.symbol_y.replace("-USDT-SWAP","").lower()
        merged_data = pd.read_csv(f'{path}{symbol1}-{symbol2}-merge.csv', parse_dates=['timestamp'])
        merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'],format='mixed')
        print("merge_data",merged_data.head())
        merged_data_copy = merged_data[(merged_data['timestamp'] >= start_time)]
        print("merged_data_copy",merged_data_copy)
        self.current_beta = merged_data_copy["beta"].iloc[-1]
        self.current_spread = merged_data_copy["spread"].iloc[-1]
        self.mean = merged_data_copy['spread'].mean()
        self.upper_bound = self.mean + 0.25 * merged_data_copy['spread'].std()
        self.lower_bound = self.mean - 0.25 * merged_data_copy['spread'].std()
        print("current beta",self.current_beta)
        print("current_spread",self.current_spread)
        print("mean",self.mean)
        print("upper_bound",self.upper_bound)
        print("lower_bound",self.lower_bound)

    def exit_position(self):
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
                amount_x  = 10/price_x
                amount_y = 10 * self.current_beta/price_y
                
                order_x = self.exchange.create_order(self.symbol_x, 'market', 'sell', amount_x)
                order_y = self.exchange.create_order(self.symbol_y, 'market', 'buy', amount_y)
                order_x_info = process_order(self.exchange,order_x,self.symbol_x)
                order_y_info = process_order(self.exchange,order_y,self.symbol_y)
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
                amount_x  = 10/price_x
                amount_y = 10 * self.current_beta/price_y

                order_x = self.exchange.create_order(self.symbol_x, 'market', 'buy', amount_x)
                order_y = self.exchange.create_order(self.symbol_y, 'market', 'sell', amount_y)
                order_x_info = process_order(self.exchange,order_x,self.symbol_x)
                order_y_info = process_order(self.exchange,order_y,self.symbol_y)
                self.orders.append(order_x_info)
                self.orders.append(order_y_info)
                self.entry_price_x = order_x_info[2]
                self.entry_price_y = order_y_info[2]
                self.entry_amount_x = order_x_info[3]
                self.entry_amount_y = order_y_info[3]

        # Long AI, Short VR position case
        elif self.position_x == 1 and self.position_y == -1:

            if self.current_spread < self.lower_bound:
                print(f"spread: {self.current_spread};lower_bound:{self.lower_bound};")
                print(f"sell {self.symbol_x} buy {self.symbol_y} , close_position!")
                # Signal reversed - close position and open opposite
                self.is_execute = True

                # close position
                order_x = self.exchange.create_order(self.symbol_x, 'market', 'sell', self.entry_amount_x)
                order_y = self.exchange.create_order(self.symbol_y, 'market', 'buy', self.entry_amount_y )
                order_x_info = process_order(self.exchange,order_x,self.symbol_x)
                order_y_info = process_order(self.exchange,order_y,self.symbol_y)
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
                amount_x = 10/x_current_price
                amount_y = 10 * self.current_beta/y_current_price
                order_x1 = self.exchange.create_order(self.symbol_x, 'market', 'sell', amount_x)
                order_y1 = self.exchange.create_order(self.symbol_y, 'market', 'buy', amount_y)
                order_x1_info = process_order(self.exchange,order_x1,self.symbol_x)
                order_y1_info = process_order(self.exchange,order_y1,self.symbol_y)
                self.orders.append(order_x1_info)
                self.orders.append(order_y1_info)
                self.entry_price_x = order_x1_info[2]
                self.entry_price_y = order_y1_info[2]
                self.entry_amount_x = order_x1_info[3]
                self.entry_amount_y = order_y1_info[3]

            elif self.current_spread < self.mean:
                # Ratio back to normal - close position
                self.is_execute = True
                print(f"spread: {self.current_spread};mean:{self.mean};")
                print(f"sell {self.symbol_x} buy {self.symbol_y} , close_position!")
                # close position
                order_x = self.exchange.create_order(self.symbol_x, 'market', 'sell', self.entry_amount_x)
                order_y = self.exchange.create_order(self.symbol_y, 'market', 'buy', self.entry_amount_y )
                order_x_info = process_order(self.exchange,order_x,self.symbol_x)
                order_y_info = process_order(self.exchange,order_y,self.symbol_y)
                self.orders.append(order_x_info)
                self.orders.append(order_y_info)
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
                print(f"spread: {self.current_spread};upper_bound:{self.upper_bound};")
                print(f"buy {self.symbol_x} sell {self.symbol_y} , close_position!")
                self.is_execute = True

                # close position
                order_x = self.exchange.create_order(self.symbol_x, 'market', 'buy', self.entry_amount_x)
                order_y = self.exchange.create_order(self.symbol_y, 'market', 'sell', self.entry_amount_y )
                order_x_info = process_order(self.exchange,order_x,self.symbol_x)
                order_y_info = process_order(self.exchange,order_y,self.symbol_y)
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
                amount_x = 10/order_x_info[2]
                amount_y = 10 * self.current_beta/order_y_info[2]
                order_x1 = self.exchange.create_order(self.symbol_x, 'market', 'buy', amount_x)
                order_y1 = self.exchange.create_order(self.symbol_y, 'market', 'sell', amount_y)
                order_x1_info = process_order(self.exchange,order_x1,self.symbol_x)
                order_y1_info = process_order(self.exchange,order_y1,self.symbol_y)
                self.orders.append(order_x1_info)
                self.orders.append(order_y1_info)
                self.entry_price_x = order_x1_info[2]
                self.entry_price_y = order_y1_info[2]
                self.entry_amount_x = order_x1_info[3]
                self.entry_amount_y = order_y1_info[3]

            elif self.current_spread > self.mean:
                # Ratio back to normal - close position
                print(f"spread: {self.current_spread};mean:{self.mean};")
                self.is_execute = True
                # close position
                order_x = self.exchange.create_order(self.symbol_x, 'market', 'buy', self.entry_amount_x)
                order_y = self.exchange.create_order(self.symbol_y, 'market', 'sell', self.entry_amount_y )
                order_x_info = process_order(self.exchange,order_x,self.symbol_x)
                order_y_info = process_order(self.exchange,order_y,self.symbol_y)
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

async def main():
    pairs = [["AI16Z-USDT-SWAP","VIRTUAL-USDT-SWAP"],["1INCH-USDT-SWAP","ALPHA-USDT-SWAP"]]
    exchange = ccxt.okx({
                "apiKey": os.getenv("OKX_KEY"),
                "secret": os.getenv('OKX_SECRET'),
                "password":"tradingPa$$word23"
            }) 

    
    tasks = []
    for symbol_x,symbol_y in pairs:
        await exchange.set_leverage(symbol = symbol_x,leverage =  10)
        await exchange.set_leverage(symbol= symbol_y, leverage = 10)

    while True:
        for symbol_x,symbol_y in pairs:
            fetch_historical_data(exchange, symbol_x)
            fetch_historical_data(exchange,symbol_y)
            fetch_merge_data(symbol_x,symbol_y)
            pairs_trade = Pairs_trade(exchange=exchange, symbol_x=symbol_x, symbol_y=symbol_y)
            await pairs_trade.get_data()
            await pairs_trade.calculate_spread()
            await pairs_trade.exit_position()
            print("PNL",pairs_trade.PNL)
            print("orders",pairs_trade.orders)
            print("entry_amount_x",pairs_trade.entry_amount_x)
            print("entry_price_x",pairs_trade.entry_price_x)
            print("entry_amount_y",pairs_trade.entry_amount_y)
            print("entry_price_y",pairs_trade.entry_price_y)
            print("entry_current_time",pairs_trade.entry_time)
            time.sleep(calculate_waiting_time())
        await asyncio.sleep(calculate_waiting_time()) 


main()
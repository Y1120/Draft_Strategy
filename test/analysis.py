import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from datetime import datetime
import time
from statsmodels.tsa.stattools import coint
import ccxt
from dotenv import load_dotenv
import os
load_dotenv()
def fetch_historical_data(exchange, symbol):
    symbol1 = symbol.replace("/USDT:USDT","").lower()
    print(f"Fetching historical data for {symbol1}...")
    data = pd.read_csv( f'{symbol1}-5min.csv', parse_dates=['timestamp'], index_col='timestamp')
    print('index',data.index[-1])
    start_dt = pd.to_datetime(data.index[-1]) + pd.Timedelta(minutes=5)

    end_dt  = datetime.now()+ pd.Timedelta(minutes=5)
    print('end_dt',end_dt)
    since = int(start_dt.timestamp() * 1000)
    print('since',since)
    end = int(end_dt.timestamp() * 1000)
    print('end',end)
    data_list = []
    while since < end:
        try: 
            until = since + 300 * 5 * 60 * 1000
            params = {
                'until': until
            }
         # Fetch batch of candles
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe='5m',
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
        data_list.to_csv(f'{symbol1}-5min.csv', mode='a', header=False,index=False)
        print(f"Data saved to {symbol}-5min.csv")
        return data_list
    
def calculate_corr_and_coin():
    # Read data with consistent datetime parsing
    data_ai16Z = pd.read_csv('ai16z-5min.csv', 
                            parse_dates=['timestamp'],
                            index_col='timestamp',
                            date_parser=lambda x: pd.to_datetime(x, format='ISO8601'))
    
    data_virtual = pd.read_csv('virtual-5min.csv',
                              parse_dates=['timestamp'],
                              index_col='timestamp',
                              date_parser=lambda x: pd.to_datetime(x, format='ISO8601'))

    # Filter last 3 months
    three_months_ago = pd.Timestamp.now() - pd.DateOffset(months=3)
    data_ai16Z = data_ai16Z[data_ai16Z.index >= three_months_ago]
    data_virtual = data_virtual[data_virtual.index >= three_months_ago]

    # Calculate correlation on returns
    corr = data_ai16Z['close'].pct_change().dropna().corr(
        data_virtual['close'].pct_change().dropna()
    )
    print(f'Pearson Correlation: {corr}')
    corr1 = data_ai16Z['close'].corr(data_virtual['close'])
    print(f'Pearson Correlation: {corr1}')
    # Test cointegration
    score, p_value, _ = coint(data_ai16Z['close'].values, 
                             data_virtual['close'].values)
    print(f'Cointegration test statistic: {score}')
    print(f'P-value: {p_value}')
    print("Cointegrated" if p_value < 0.05 else "Not cointegrated")

if __name__ == '__main__':
    api_key = os.environ.get('OKEX_KEY')
    secret = os.environ.get('OKEX_SECRET')

    exchange = ccxt.okx({
        "apiKey": api_key, 
        "secret": secret,
       "password":"tradingPa$$word23"
    })
    exchange.load_markets()
    fetch_historical_data(exchange, 'AI16Z/USDT:USDT')
    fetch_historical_data(exchange, 'VIRTUAL/USDT:USDT')
    calculate_corr_and_coin()

import requests
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import traceback
import math
import pprint
# get all symbols with perp
def extract_inst_ids(data):
    inst_ids = [item['instId'] for item in data]

    list_time = [item['listTime'] for item in data]
    dic = dict(zip(inst_ids,list_time))
    return dic

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
        print(int(contract_size[0]))
            
    except Exception as e:
        print(f"Error getting symbols: {e}")
        return None

def get_all_symbols():
    try:
        # Get perpetual futures symbols
        futures_response = requests.get(
            'https://www.okx.com/api/v5/public/instruments',
            params={'instType': 'SWAP'}
        )
        futures_data = futures_response.json()
        filtered_items = [
            item
            for item in futures_data['data'] if 'CFX-USDT-SWAP' in item['instId']
        ]
        print(filtered_items)
        if futures_data and futures_data.get('code') == '0' and 'data' in futures_data:
            symbols_dict = {
                item['instId']: item['listTime'] 
                for item in futures_data['data'] if 'USDT-SWAP' in item['instId']

            }
            print("Found symbols:", symbols_dict)
            return symbols_dict
            
        else:
            print("Invalid API response:", futures_data)
            return {}
            
    except Exception as e:
        print(f"Error getting symbols: {e}")
        return {}

def get_symbols_data(symbol, listTime, interval):
    print(f"Fetching data for {symbol} with interval {interval}")
    exchange = ccxt.okx()

    print("listTime", listTime)
    start_ts = int(listTime)
    print("start_ts", start_ts)
    
    # 将时间戳转换为可读的 datetime 格式
    start_datetime = pd.to_datetime(start_ts, unit='ms')
    
    # 设置比较时间
    threshold_datetime = pd.Timestamp('2025-01-01 00:00:00')

    if start_datetime < threshold_datetime:
        start_ts = int(threshold_datetime.timestamp() * 1000)  # 转换为毫秒
        start = pd.to_datetime(start_ts, unit='ms')
        print("Adjusted start_ts:", start_ts)
    else:
        start = start_datetime

    print("start",start_ts)
    end = pd.to_datetime(start) + timedelta(minutes = 100)
    end_ts = int(end.timestamp() * 1000)
    print(end_ts)
    end1 = '2025-03-20 16:00:00'
    end_ts1 = pd.to_datetime(end1) + timedelta(minutes = 100)
    end_ts1 = int(pd.Timestamp(end1).timestamp() * 1000)

    data_list = []
    
    while end_ts < end_ts1:
        try:
            params = {
                'until': end_ts
            }
            candles = exchange.fetch_ohlcv(symbol, interval, 
                                            since=start_ts, limit = 100,
                                            params = params)
           
            if len(candles) == 0:
                print("No more data available")
                break
            else:
                start_ts = end_ts
                start_datetime = pd.to_datetime(start_ts, unit='ms')
                end = start_datetime + timedelta(minutes = 100)
                end_ts = int(end.timestamp() * 1000)
                print("end",end)
                print(start_ts, end_ts)
                data_list.extend(candles)
            time.sleep(0.1)
            print("time-sleep-10")
        except Exception as e:
            print(f"Error fetching {symbol}")
            print(traceback.format_exc())
            break

    df =    pd.DataFrame(data_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.to_csv(f'C:/Users/theo/Desktop/Astra-folder/pairs_data_symbols/30m/{symbol}_{interval}.csv')
import os
def check_generate_data(symbol):
    file_path = f'C:/Users/theo/Desktop/Astra-folder/pairs_data_symbols/30m/{symbol}_30m.csv'
    if os.path.exists(file_path):
        print(f"Data for {symbol} already exists")
        return True
    else:
        return False
if __name__ == '__main__':
    get_contract_size('CFX-USDT-SWAP')
    get_contract_size('DOGE-USDT-SWAP')


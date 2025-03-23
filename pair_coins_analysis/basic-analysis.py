import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import coint
from coins_data_collection import get_all_symbols
from telegram_alarm import TelegramNotifier, fetch_data
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from backtest_new  import backtest
from backtest_beta import backtest1
load_dotenv()
async def corr_and_cointegration(symbol1, symbol2, interval):
    try:
        is_pair = False
        telegram = TelegramNotifier(
            bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            default_chat_id=os.getenv('TELEGRAM_CHAT_ID')
        )

        # Read and prepare data
        data_sym1 = pd.read_csv(f'C:/Users/theo/Desktop/Astra-folder/pairs_data_symbols/{symbol1}_{interval}.csv', 
                               parse_dates=['timestamp'])
        data_sym2 = pd.read_csv(f'C:/Users/theo/Desktop/Astra-folder/pairs_data_symbols/{symbol2}_{interval}.csv', 
                               parse_dates=['timestamp'])
            # 当前时间
        now = datetime.now()
        # Set index and sort
        data_sym1.set_index('timestamp', inplace=True)
        data_sym2.set_index('timestamp', inplace=True)
        data_sym1.sort_index(inplace=True)
        data_sym2.sort_index(inplace=True)
        # 计算日期范围
        start_date = now - timedelta(days=90)
        end_date = now - timedelta(days=60)
        print(f"Start date: {start_date}, End date: {end_date}")
        data_sym1 = data_sym1[(data_sym1.index >= start_date) & (data_sym1.index < end_date)]
        data_sym2 = data_sym2[(data_sym2.index >= start_date) & (data_sym2.index < end_date)]
        # Validate data
        if data_sym1.empty or data_sym2.empty:
            print(f"Empty data for {symbol1} or {symbol2}")
            return None, False

        # Align timestamps
        common_index = data_sym1.index.intersection(data_sym2.index)
        if len(common_index) < 100:  # Minimum data points required
            print(f"Insufficient overlapping data points for {symbol1} and {symbol2}")
            return None, False

        data_sym1 = data_sym1.loc[common_index]
        data_sym2 = data_sym2.loc[common_index]

        # Calculate correlation
        correlation = data_sym1['close'].corr(data_sym2['close'])
        if not np.isfinite(correlation):
            print(f"Invalid correlation for {symbol1} and {symbol2}")
            return None, False

        print(f'Pearson Correlation: {correlation:.4f}')

        # Cointegration test with data validation
        if not (np.all(np.isfinite(data_sym1['close'])) and np.all(np.isfinite(data_sym2['close']))):
            print(f"Invalid price data for {symbol1} or {symbol2}")
            return None, False

        score, p_value, _ = coint(data_sym1['close'].values, data_sym2['close'].values)
        print(f'Cointegration test statistic: {score:.4f}')
        print(f'P-value: {p_value:.4f}')

        alpha = 0.1
        if p_value < alpha and correlation > 0.93:
            is_pair = True
            print(f"{symbol1} and {symbol2} are cointegrated and correlated")
            data = {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "correlation": float(correlation),
                "p_value": float(p_value),
                "coint_score": float(score)
            }
            return data, is_pair
        return None, False

    except Exception as e:
        print(f"Error analyzing {symbol1} and {symbol2}: {str(e)}")
        return None, False

if __name__ == '__main__':
    symbols = get_all_symbols()
    data_list = []
    analyzed_pairs = set() 
    
    try:
        for symbol1, listTime in symbols.items():
            for symbol2 in symbols.keys():
                if symbol1 >= symbol2 or (symbol1, symbol2) in analyzed_pairs:  # 跳过重复和自对
                    continue
                
                print(f'\nAnalyzing {symbol1} vs {symbol2}...')
                data, is_pair = asyncio.run(corr_and_cointegration(symbol1, symbol2, '1m'))
                if is_pair is True and data is not None:
                    data_list.append(data)
                    analyzed_pairs.add((symbol1, symbol2))  # 添加到已分析的符号对集合
                    print(f"\nTesting pair: {symbol1} - {symbol2}")
                    result = backtest(symbol1, symbol2)
                    result1 = backtest1(symbol1, symbol2)
                    result2 = {
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "correlation": data["correlation"],
                        "p_value": data["p_value"],
                        "coint_score": data["coint_score"],
                        "result": result
                    }
                    result2= pd.DataFrame(result2)
                    result2.to_csv(f"C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_analysis\\results\\pair_data.csv",mode='a',header=False,index=False)

        # 保存结果
        if data_list:
            df = pd.DataFrame(data_list)
            df.to_csv('pair_data.csv', index=False)
            print(f"\nSuccessfully saved {len(data_list)} pairs to pair_data.csv")
        else:
            print("\nNo valid pairs found")
            
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
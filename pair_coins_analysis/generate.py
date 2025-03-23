from coins_data_collection import get_all_symbols
from calculate_beta import calculate_beta_with_rolling_windows
import pandas as pd
def generate_merge_data(symbol_x,symbol_y):
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
    merage_data = pd.DataFrame(merage_data)
    merage_data.to_csv(f'C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\1m\\{symbol_x}-{symbol_y}-merge.csv')
import os
def check_pairs(symbol1,symbol2):
    path = 'C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\1m\\'
    folder_path = "C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\results_csv\\"
    file_path = os.path.join(folder_path, f'{symbol1} & {symbol2}.csv')
    if os.path.exists(file_path):
        print("file exists")
        return True
    else:
        print("file not exists")
        return False   
if __name__ == '__main__':
    symbols = get_all_symbols()
    for symbol1, listTime in symbols.items():
        for symbol2 in symbols.keys():
            is_pair = check_pairs(symbol1,symbol2)
            if is_pair:
               
                generate_merge_data(symbol1,symbol2)
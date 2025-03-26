
import os
from pathlib import Path
import pandas as pd
from statsmodels.tsa.stattools import adfuller
# read file
def get_all_filenames(directory_path):
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: Directory not found - {directory_path}")
        return []
    file_list = [file.name for file in dir_path.iterdir() if file.is_file()]    
    print(f"Found {len(file_list)} files in {directory_path}")
    return file_list
# summery the pnl result

def get_results(file_name):
    file_path = f"C:\\Users\\theo\\Desktop\\Astra-folder\\pairs_data_symbols\\results_csv\\{file_name}"
    file = pd.read_csv(file_path)
    column_names = [
        'uniname', 'timestamps', 'symbol1', 'symbol2', 'pnl',
        'ai_position', 'vr_position', 'amount_ai', 'amount_vr', 'ratio', 'beta'
    ]
    if len(file.columns) == len(column_names):
        file.columns = column_names
    else:
        # If columns don't match exactly, try to map the most important ones
        # and keep the rest as is
        rename_map = {}
        for i, col in enumerate(file.columns):
            if i < len(column_names):
                rename_map[col] = column_names[i]
        file = file.rename(columns=rename_map)
        print(f"Warning: Column count mismatch in {file_name}. Partial renaming applied.")
   
    print(file_path)
    print("file", file)
    base_name = file_name
    trades_time = 0
    is_entry = 0
    entry_time = 0
    trades_lasting_time = 0
    notional_y = pd.DataFrame(columns=['timestamp','notion_y','symbol','final_pnl'])
    if '.' in base_name:
        base_name = base_name.rsplit('.', 1)[0]
        parts = base_name.split('&')
        symbol1 = parts[0]
        symbol2 = parts[1]
    for index, row in file.iterrows():
        if is_entry == 0:
            if row["ai_position"] == 1 or row["ai_position"] == -1:
                entry_time = row["timestamps"]
                is_entry = 1
                notion_y = 100*row['beta']
                new_row = pd.DataFrame({
                    'timestamp': [row["timestamps"]],
                    'notion_y': [notion_y],
                    'symbol': file_name.split('_')[0],
                    "final_pnl": file['pnl'].iloc[-1]
                })
                print("new_row",new_row)
                notional_y = pd.concat([notional_y, new_row], ignore_index=True)
            elif row["ai_position"] == 0:
                continue
        elif is_entry == 1:
            if row["ai_position"] == 0:
                close_time = row["timestamps"]
                # Calculate time difference in minutes
                time_diff = (pd.to_datetime(close_time) - pd.to_datetime(entry_time)).total_seconds() / 60
                trades_lasting_time += time_diff
                trades_time += 1
                is_entry = 0
    try:
        pnl = file['pnl'].iloc[-1]
        average_trades_lasting_time = trades_lasting_time / trades_time if trades_time > 0 else 0
        beta_mean = file["beta"].dropna().mean()
        beta_std = file["beta"].dropna().std()
        ratio = float(file['ratio'])
        adf_result = adfuller(ratio.dropna())
        adf_stat = adf_result[0]  # Test statistic
        adf_pvalue = adf_result[1]  # p-value
        corr = file['symbol1'].corr(file['symbol2'])
        
    except:
        pnl = file.iloc[-1, 4]
        average_trades_lasting_time = trades_lasting_time / trades_time if trades_time > 0 else 0
        beta_mean = file.iloc[:,10].dropna().mean()
        beta_std = file.iloc[:,10].dropna().std()
        adf_pvalue = None
        corr = None
    
    return {
        'symbol1': symbol1,
        'symbol2': symbol2,
        'pnl': pnl,
        'trades_count': trades_time,
        'avg_duration': average_trades_lasting_time,
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        "adf_p_value": adf_pvalue,
        "correlation": corr
    },notional_y

        

if __name__ == "__main__":
    target_dir = r"C:\Users\theo\Desktop\Astra-folder\pairs_data_symbols\results_csv"
    filenames = get_all_filenames(target_dir)
    results = pd.DataFrame(columns = ['symbol1', 'symbol2', 'pnl', 'trades_count', 'avg_duration', 'beta_mean', 'beta_std','adf_p_value','correlation'])
    notion_all = pd.DataFrame(columns=['timestamp','notion_y',"symbol",'final_pnl'])
    for i, filename in enumerate(filenames):
        try:
            # Get results for this file
            result_dict,notional_y = get_results(filename)
            notion_all = pd.concat([notion_all,notional_y],ignore_index=True)
            # Proper way to append to DataFrame - create single row DF and concat
            result_df = pd.DataFrame([result_dict])
            results = pd.concat([results, result_df], ignore_index=True)
            
            # Print progress
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(filenames)} files")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Sort results by PnL (descending)
    results = results.sort_values(by='pnl', ascending=False)
    
    # Save results to CSV
    output_path = os.path.join(os.path.dirname(target_dir), "pair_trading_summary_all111.csv")
    notional_path = os.path.join(os.path.dirname(target_dir), "notion_all.csv")
    notion_all.to_csv(notional_path,index=False)
    results.to_csv(output_path, index=False)
    
    # Print summary
    print("\nAnalysis Complete!")
    print(f"Analyzed {len(results)} pair trading strategies")
    print(f"Top 5 strategies by PnL:")
    print(results.head(5)[['symbol1', 'symbol2', 'pnl', 'trades_count']])
    print(f"\nResults saved to: {output_path}")
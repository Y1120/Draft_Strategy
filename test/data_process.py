import pandas as pd
from datetime import datetime
import time
def merge_price_data():
    # Read CSVs with datetime index
    ai16z_data = pd.read_csv('C:/Users/theo/Desktop/Astra-folder/ai16z-1min.csv',
                            parse_dates=['timestamp'])
    virtual_data = pd.read_csv('C:/Users/theo/Desktop/Astra-folder/virtual-1min.csv',
                              parse_dates=['timestamp'])

    # Select and rename columns
    ai16z_data = ai16z_data[['timestamp', 'close']].rename(columns={'close': 'ai16z'})
    virtual_data = virtual_data[['timestamp', 'close']].rename(columns={'close': 'virtual'})

    # Merge on timestamp
    merged_data = pd.merge(virtual_data, ai16z_data, 
                          on='timestamp', 
                          how='inner')
    
    # Set timestamp as index
    merged_data.set_index('timestamp', inplace=True)
    merged_data.sort_index(inplace=True)
    
    # Save merged data
    merged_data.to_csv('merged_data_1min.csv')
    return merged_data
merge_price_data()

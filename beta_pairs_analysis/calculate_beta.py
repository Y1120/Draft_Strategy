"""
Define three ways of calculating beta: Basic beta calculation is based on the OLS regression model, but different on each time points:
1. Assume beta is stable after a specific time;
2. Assume beta is unstable and use roliing windows;
3. Assume beta is unstable and use Kalmen Filter;
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from pykalman import KalmanFilter
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
import traceback
def calculate_beta(merged_data):

    data_X = merged_data['ai16z']
    data_y = merged_data['virtual']
    
    # Log-transform data
    X = sm.add_constant(np.log(data_X))
    y = np.log(data_y)

    # Fit OLS model
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    
    # Calculate the variance of the residuals
    residual_variance = np.var(residuals, ddof=1)  # Use ddof=1 for sample variance

    # Fit GLS model without custom covariance
    gls_model = sm.GLS(y, X).fit()  # No sigma parameter
    beta = gls_model.params

    # Calculate spread
    merged_data['spread'] = np.log(data_y) - np.log(data_X) * beta[1] - beta[0]
    adf_result = adfuller(merged_data['spread'].dropna())
    # Extract results
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]

    # Print results
    print("ADF Statistic:", adf_statistic)
    print("p-value:", p_value)
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"   {key}: {value}")
    print("spread", merged_data['spread'])
    print("merage", merged_data)
    print("beta",beta)
    print("betaaaaa",beta['ai16z'])
    return merged_data, beta['ai16z']


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
def calculate_beta_with_rolling_windows_ECM(merged_data):
    # Create lists to store results
    rolling_betas = []
    
    # Define timedelta for the rolling window
    window_size = pd.Timedelta(days=30)
    start_time = merged_data['timestamp'].iloc[0] + window_size
    times = []
    
    # Initialize spread as a column
    merged_data['spread'] = np.nan
    last_spread = None
    # Loop through the merged_data with a rolling window
    for end in merged_data['timestamp']:
        if end < start_time:
            continue
        print("end",end)    
        start = end - window_size
        times.append(end)
        # Select the windowed data
        window_data = merged_data[(merged_data['timestamp'] >= start) & (merged_data['timestamp'] <= end)]
        
        if len(window_data) < 2:  # Need at least two observations to calculate beta
            continue  # Skip if we don't have enough data

        data_X = window_data['ai16z']
        data_y = window_data['virtual']

        
        # Log-transform data
        X = sm.add_constant(np.log(data_X))
        y = np.log(data_y)

        # Fit OLS model
        model = sm.OLS(y, X).fit()
        
        # Get beta
        beta = model.params['ai16z']

        # Calculate spread for the last observation in the current window
        last_X = np.log(data_X.iloc[-1])
        last_y = np.log(data_y.iloc[-1])
        print("last_y1",last_y)
        # Use previous spread to adjust beta
        if last_spread is not None:
            #last_spread = merged_data['spread'].iloc[-1]
            adjusted_last_y = last_y - last_spread  # Adjusting last observation using previous spread
        else:
            adjusted_last_y = last_y  # If no previous spread, just use last_y
        
        print("const",model.params['const'])
        last_spread = adjusted_last_y - (last_X * beta) - model.params['const']
        print("spread",last_spread)
        print("beta",beta)
        print("last_X",last_X)
        print("last_y",adjusted_last_y)

        # Append the spread and beta for the last observation
        merged_data.loc[merged_data['timestamp'] == end, 'spread'] = last_spread
        merged_data.loc[merged_data['timestamp'] == end, 'beta'] = beta
        rolling_betas.append(beta)

    # Perform ADF test on calculated spreads
    adf_result = adfuller(merged_data['spread'].dropna())
    
    # Extract ADF results
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]

    # Print ADF results
    print("ADF Statistic:", adf_statistic)
    print("p-value:", p_value)
    for key, value in critical_values.items():
        print(f"   {key}: {value}")

    return merged_data

def calculate_beta_with_rolling_windows_ECM_MRP(merged_data, gamma=1):
    # Create lists to store results
    rolling_betas = []
    
    # Define timedelta for the rolling window
    window_size = pd.Timedelta(days=30)
    start_time = merged_data['timestamp'].iloc[0] + window_size
    times = []
    
    # Initialize columns for spread and position
    merged_data['spread'] = np.nan
    merged_data['position'] = np.nan
    last_spread = None

    # Loop through the merged_data with a rolling window
    for end in merged_data['timestamp']:
        if end < start_time:
            continue
        start = end - window_size
        times.append(end)
        
        # Select the windowed data
        window_data = merged_data[(merged_data['timestamp'] >= start) & (merged_data['timestamp'] <= end)]
        
        if len(window_data) < 2:
            continue  # Skip if we don't have enough data

        data_X = window_data['ai16z']
        data_y = window_data['virtual']

        # Log-transform data
        X = sm.add_constant(np.log(data_X))
        y = np.log(data_y)

        # Fit OLS model
        model = sm.OLS(y, X).fit()
        
        # Get beta
        beta = model.params['ai16z']

        # Calculate spread for the last observation in the current window
        last_X = np.log(data_X.iloc[-1])
        last_y = np.log(data_y.iloc[-1])
        
        if last_spread is not None:
            adjusted_last_y = last_y - last_spread
        else:
            adjusted_last_y = last_y
        
        last_spread = adjusted_last_y - (last_X * beta) - model.params['const']

        # Calculate position based on the spread and chosen weights
        position = np.array([1, -gamma]) @ np.array([last_y, last_spread])
        
        # Append results
        merged_data.loc[merged_data['timestamp'] == end, 'spread'] = last_spread
        merged_data.loc[merged_data['timestamp'] == end, 'beta'] = beta
        merged_data.loc[merged_data['timestamp'] == end, 'position'] = position
        rolling_betas.append(beta)

    # Perform ADF test on calculated spreads
    adf_result = adfuller(merged_data['spread'].dropna())
    
    # Extract ADF results
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]

    # Print ADF results
    print("ADF Statistic:", adf_statistic)
    print("p-value:", p_value)
    for key, value in critical_values.items():
        print(f"   {key}: {value}")

    return merged_data

def calculate_beta_with_rolling_windows(merged_data):
    # Create lists to store results
    rolling_betas = []
    
    # Define timedelta for the rolling window
    window_size = pd.Timedelta(days=30)
    start_time = merged_data['timestamp'].iloc[0] + window_size
    times = []
    
    # Loop through the merged_data with a rolling window
    for end in merged_data['timestamp']:
        if end < start_time:
            continue
        start = end - window_size
        times.append(end)
        # Select the windowed data
        window_data = merged_data[(merged_data['timestamp'] >= start) & (merged_data['timestamp'] <= end)]
        
        if len(window_data) < 2:  # Need at least two observations to calculate beta
            continue  # Skip if we don't have enough data

        data_X = window_data['ai16z']
        data_y = window_data['virtual']
        
        # Log-transform data
        X = sm.add_constant(np.log(data_X))
        y = np.log(data_y)

        # Fit OLS model
        model = sm.OLS(y, X).fit()
        
        # Fit GLS model without custom covariance
        gls_model = sm.GLS(y, X).fit()  # No sigma parameter
        beta = gls_model.params['ai16z']  # Get the beta for ai16z

        # Calculate spread for the last observation in the current window
        last_X = np.log(data_X.iloc[-1])  # Log of the last X value
        last_y = np.log(data_y.iloc[-1])  # Log of the last y value
        spread = last_y - (last_X * beta) - model.params['const']

        # Append the spread for the last observation
        merged_data.loc[merged_data['timestamp'] == end, 'spread'] = spread
        merged_data.loc[merged_data['timestamp'] == end, 'beta'] = beta

        rolling_betas.append(beta)

    # Perform ADF test on calculated spreads
    adf_result = adfuller(merged_data['spread'].dropna())
    
    # Extract ADF results
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]

    # Print ADF results
    print("ADF Statistic:", adf_statistic)
    print("p-value:", p_value)
    for key, value in critical_values.items():
        print(f"   {key}: {value}")

    return merged_data

def calculate_beta_with_Kalmen(meraged_data):
    data_X = meraged_data['ai16z']
    data_y = meraged_data['virtual']
    obs_mat = sm.add_constant(np.log(data_X).values, prepend=False)[:, np.newaxis]

    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, 
                    initial_state_mean=np.ones(2),
                    initial_state_covariance=np.ones((2, 2)),
                    transition_matrices=np.eye(2),
                    observation_matrices=obs_mat,
                    observation_covariance=0.5,
                    transition_covariance=0.000001 * np.eye(2))
    state_means, state_covs = kf.filter(np.log(data_y).values)
    slope=state_means[:, 0] 
    intercept=state_means[:, 1]


    kl_spread = np.log(data_y) - np.log(data_X) * slope - intercept
    meraged_data['spread'] = kl_spread
        # Perform ADF test on calculated spreads
    adf_result = adfuller(meraged_data['spread'].dropna())
    
    # Extract ADF results
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]

    # Print ADF results
    print("ADF Statistic:", adf_statistic)
    print("p-value:", p_value)
    for key, value in critical_values.items():
        print(f"   {key}: {value}")
    beta = pd.DataFrame({
        'timestamp': meraged_data['timestamp'],
        'beta': slope
    })    
    return meraged_data,beta
    

def calculate_bound(merged_data,current_timestamp,lookback_days):
    """Calculate bounds using only historical data"""
    # Define the lookback window end (current point) and start (30 days before)
    lookback_start = current_timestamp - timedelta(days=lookback_days)
    data_ai = merged_data[['ai16z','timestamp']]
    data_vr = merged_data[['virtual','timestamp']]
    # Filter data to only include the lookback period UP TO current timestamp (no future data)
    merged_data_copy = merged_data[(merged_data['timestamp'] >= lookback_start) &
                      (merged_data['timestamp'] < current_timestamp)]
    # Ensure we have enough data to calculate meaningful bounds
    if len(merged_data_copy) < 5:  # Arbitrary threshold to ensure enough data
        return None, None
    ratio =merged_data_copy['spread']
    ratio = ratio.dropna()
    # Align timestamps and calculate ratio

    if len(ratio) < 5:  # Another check after alignment
        return None, None

    # Calculate bounds
    mean = ratio.mean()
    std = ratio.std()
    upper_bound = mean + 1 * std
    lower_bound = mean - 1 * std

    return upper_bound, lower_bound, mean

def exit_position(ratio, ai_position, virtual_position, PNL, upper_bound, lower_bound, mean,
                  current_time, entry_time, data_ai, data_vr,amount1,amount2):
    """

    Modified to use timestamps for position tracking instead of indices
    """

    is_exit = False

    # Find current prices
    ai_current_price = data_ai.loc[data_ai['timestamp'] == current_time, 'close'].iloc[0]
    vr_current_price = data_vr.loc[data_vr['timestamp'] == current_time, 'close'].iloc[0]
    # No position case
    if ai_position == 0 and virtual_position == 0:
        if ratio >= lower_bound and ratio <= upper_bound:
            # No position, and ratio is within bounds - do nothing
            pass
        elif ratio < lower_bound:
            # Ratio below lower bound - go long AI, short VR
            ai_position = 1
            virtual_position = -1
            entry_time = current_time
        elif ratio > upper_bound:
            # Ratio above upper bound - go short AI, long VR
            ai_position = -1
            virtual_position = 1
            entry_time = current_time

    # Long AI, Short VR position case
    elif ai_position == 1 and virtual_position == -1:
        if ratio > upper_bound:
            # Signal reversed - close position and open opposite
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]

            # Calculate PNL (long AI, short VR)
            ai_pnl = amount1 * (ai_current_price - ai_entry_price)
            vr_pnl = amount2 * (vr_entry_price - vr_current_price)  # Note: short position
            PNL += ai_pnl + vr_pnl

            # Open new position (reverse)
            ai_position = -1
            virtual_position = 1
            entry_time = current_time

        elif ratio > mean:
            # Ratio back to normal - close position
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]

            # Calculate PNL (long AI, short VR)
            ai_pnl = amount1 * (ai_current_price - ai_entry_price)
            vr_pnl = amount2 * (vr_entry_price - vr_current_price)  # Note: short position
            PNL += ai_pnl + vr_pnl

            # No position
            ai_position = 0
            virtual_position = 0

    # Short AI, Long VR position case
    elif ai_position == -1 and virtual_position == 1:
        if ratio < lower_bound:
            # Signal reversed - close position and open opposite
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]

            # Calculate PNL (short AI, long VR)
            ai_pnl = amount1 * (ai_entry_price - ai_current_price)  # Note: short position
            vr_pnl = amount2 * (vr_current_price - vr_entry_price)
            PNL += ai_pnl + vr_pnl

            # Open new position (reverse)
            ai_position = 1
            virtual_position = -1
            entry_time = current_time

        elif ratio < mean:
            # Ratio back to normal - close position
            is_exit = True

            # Find entry prices
            ai_entry_price = data_ai.loc[data_ai['timestamp'] == entry_time, 'close'].iloc[0]
            vr_entry_price = data_vr.loc[data_vr['timestamp'] == entry_time, 'close'].iloc[0]

            # Calculate PNL (short AI, long VR)
            ai_pnl = amount1 * (ai_entry_price - ai_current_price)  # Note: short position
            vr_pnl = amount2 * (vr_current_price - vr_entry_price)
            PNL += ai_pnl + vr_pnl

            # No position
            ai_position = 0
            virtual_position = 0

    return ai_position, virtual_position, PNL, is_exit, entry_time
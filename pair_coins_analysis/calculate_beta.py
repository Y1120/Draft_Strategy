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
    is_cointegration = False
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
    if p_value < 0.2:
        is_cointegration = True
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
    return merged_data, beta['ai16z'],is_cointegration

def calculate_beta_with_rolling_windows(merged_data):
    try:
        is_cointegration = False
        # Create lists to store results
        rolling_betas = []
        
        # Define timedelta for the rolling window
        window_size = pd.Timedelta(days=30)
        print(merged_data['timestamp'].iloc[0])

        start_time = merged_data['timestamp'].iloc[0] + window_size
        end_time = merged_data['timestamp'].iloc[0] + pd.Timedelta(days=60)

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
        cointegration_test_data = merged_data[(merged_data['timestamp'] > start_time) & (merged_data['timestamp'] < end_time)]
        adf_result = adfuller(cointegration_test_data['spread'].dropna())
        
        # Extract ADF results
        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]
        if p_value < 1:
            is_cointegration = True
        # Print ADF results
        print("ADF Statistic:", adf_statistic)
        print("p-value:", p_value)
        for key, value in critical_values.items():
            print(f"   {key}: {value}")
        return merged_data,is_cointegration,p_value
    except:
        print("error getting cointegration")
        traceback.print_exc()
        is_cointegration = False
        p_value = 999
        return merged_data,is_cointegration,p_value

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
        return None, None, None
    ratio =merged_data_copy['spread']
    ratio = ratio.dropna()
    # Align timestamps and calculate ratio

    if len(ratio) < 5:  # Another check after alignment
        return None, None, None

    # Calculate bounds
    mean = ratio.mean()
    std = ratio.std()
    upper_bound = mean + 1 * std
    lower_bound = mean - 1 * std

    return upper_bound, lower_bound, mean

def calculate_beta_with_rolling_windows1111(merged_data):
    try:
        is_cointegration = False
        # Create lists to store results
        rolling_betas = []
        
        # Define timedelta for the rolling window
        window_size = pd.Timedelta(days=30)
        print(merged_data['timestamp'].iloc[0])

        start_time = merged_data['timestamp'].iloc[0] + window_size
        end_time = merged_data['timestamp'].iloc[0] + pd.Timedelta(days=60)

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
            X = sm.add_constant(data_X)
            y = data_y

            # Fit OLS model
            model = sm.OLS(y, X).fit()
            
            # Fit GLS model without custom covariance
            gls_model = sm.GLS(y, X).fit()  # No sigma parameter
            beta = gls_model.params['ai16z']  # Get the beta for ai16z

            # Calculate spread for the last observation in the current window
            last_X = data_X.iloc[-1]  # Log of the last X value
            last_y = data_y.iloc[-1]  # Log of the last y value
            spread = last_y - (last_X * beta) - model.params['const']

            # Append the spread for the last observation
            merged_data.loc[merged_data['timestamp'] == end, 'spread'] = spread
            merged_data.loc[merged_data['timestamp'] == end, 'beta'] = beta
            rolling_betas.append(beta)

        # Perform ADF test on calculated spreads
        cointegration_test_data = merged_data[(merged_data['timestamp'] > start_time) & (merged_data['timestamp'] < end_time)]
        adf_result = adfuller(cointegration_test_data['spread'].dropna())
        
        # Extract ADF results
        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]
        if p_value < 1:
            is_cointegration = True
        # Print ADF results
        print("ADF Statistic:", adf_statistic)
        print("p-value:", p_value)
        for key, value in critical_values.items():
            print(f"   {key}: {value}")
        return merged_data,is_cointegration,p_value
    except:
        print("error getting cointegration")
        traceback.print_exc()
        is_cointegration = False
        p_value = 999
        return merged_data,is_cointegration,p_value
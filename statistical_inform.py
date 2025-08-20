import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import adfuller

# Read the CSV file (please modify the file path to your actual file path)
input_file = 'data/df_world.csv'  
df = pd.read_csv(input_file, parse_dates=['Date'], index_col='Date')

# Calculate log returns
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

def calculate_realized_volatility(data, window_length=20, return_last=None):
    """
    Calculates the realized volatility for each data column using a rolling window.
    
    Parameters:
    - data (pd.DataFrame): A DataFrame containing a 'Date' column and multiple data columns.
    - window_length (int): The length of the rolling window (default is 20).
    - return_last (int or None): If specified, returns only the last n rows of the result. If None, returns all.
    
    Returns:
    - pd.DataFrame: A DataFrame with 'Date' as the index, containing the realized volatility for each column.
    """
    # Sort by date and reset the index
    data = data.sort_values('Date').reset_index(drop=True)
    
    # Create the result DataFrame, carrying over the 'Date' column
    result_df = pd.DataFrame({'Date': data['Date']})
    result_df = result_df.iloc[window_length:].set_index('Date')

    
    # Iterate over all columns except for 'Date'
    for col in data.columns.drop('Date'):
        realized_vols = []
        for i in range(20, len(data)):
            # 21-day data slice
            window_data = data[col].iloc[i-21:i]
            window_data = 100 * window_data
            # Square the values
            squared_returns = np.square(window_data)
            # Calculate the mean and then the square root
            realized_vol = np.sqrt(np.mean(squared_returns, axis=0))
            realized_vols.append(realized_vol)
        result_df[col] = realized_vols
    
    # Return only the last n rows if specified.
    if return_last is not None and return_last < len(result_df):
        result_df = result_df.iloc[-return_last:]
    
    return result_df

# Example usage:
# df is a DataFrame with a 'Date' column and 10 data columns
# vol_df = calculate_realized_volatility(df, window_length=20, return_last=50)
# print(vol_df)

# List to store the statistical results
results = []

# Select only numeric columns for calculation, dropping any NaNs
for col in df.select_dtypes(include=[np.number]).columns:
    data = df[col].dropna()
    
    mean_val = data.mean()
    max_val = data.max()
    min_val = data.min()
    std_val = data.std()
    skew_val = skew(data)
    kurt_val = kurtosis(data)
    
    # Jarque-Bera test: returns the test statistic and p-value
    try:
        jb_stat, jb_p = jarque_bera(data)
    except Exception as e:
        jb_stat, jb_p = np.nan, np.nan
    
    # ADF test: uses the adfuller function, returns the test statistic and p-value
    try:
        adf_result = adfuller(data)
        adf_stat = adf_result[0]
        adf_p = adf_result[1]
    except Exception as e:
        adf_stat, adf_p = np.nan, np.nan
    
    results.append({
        'Column': col,
        'Mean': mean_val,
        'Max': max_val,
        'Min': min_val,
        'Std': std_val,
        'Skewness': skew_val,
        'Kurtosis': kurt_val,
        'JarqueBera_stat': jb_stat,
        'JarqueBera_p': jb_p,
        'ADF_stat': adf_stat,
        'ADF_p': adf_p
    })

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Save the results to a CSV file (specify the desired filename)
output_file = 'statistics_results_logreturns.csv'
results_df.to_csv(output_file, index=False)

print(f"Statistics saved to {output_file}")

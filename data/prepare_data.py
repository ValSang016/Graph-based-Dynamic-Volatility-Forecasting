# prepare_data.py
import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from config import CONFIG
from utils import compute_hurst_series, detect_regimes

def calculate_full_realized_volatility(data_np, lookback=20):
    """
    Calculates the realized volatility time series for the entire period without slicing.
    lookback: The number of past days to use for the volatility calculation (e.g., 20).
    """
    realized_vols = []
    # Start calculation after enough data for the first lookback window is available
    for i in range(lookback, len(data_np)):
        # Volatility at time 'i' is calculated using data from 'i-lookback' to 'i'
        window_data = data_np[i-lookback:i] * 100
        squared_returns = np.square(window_data)
        realized_vol = np.sqrt(np.mean(squared_returns, axis=0))
        realized_vols.append(realized_vol)
    return np.array(realized_vols)

def main():
    """
    Performs time-consuming data preprocessing tasks in advance and saves the results
    to a 'cache/' directory. This script only needs to be run once before starting
    the experiments.
    """
    CACHE_DIR = 'cache'
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Cache directory '{CACHE_DIR}' created or already exists.")

    # --- 1. Calculate and Cache Realized Volatility ---
    print("Calculating and caching realized volatility...")
    log_return_df = pd.read_csv(CONFIG.LOG_RETURN_PATH)
    data_np = log_return_df.drop(columns=['Date']).values
    
    # Calculate the volatility time series for the entire period
    full_volatility = calculate_full_realized_volatility(data_np, lookback=20)
    
    # Convert to DataFrame and save
    vol_df = pd.DataFrame(full_volatility)
    vol_path = os.path.join(CACHE_DIR, 'realized_volatility_full.csv')
    vol_df.to_csv(vol_path, index=False)
    print(f"Full realized volatility series saved to {vol_path}")

    # --- 2. Calculate and Cache Hurst Exponent and Regime Change Dates ---
    print("\nCalculating and caching Hurst exponent series...")
    hurst_df = compute_hurst_series(CONFIG.WORLD_INDEX_PATH)
    hurst_path = os.path.join(CACHE_DIR, 'hurst_series.csv')
    hurst_df.to_csv(hurst_path)
    print(f"Hurst exponent series saved to {hurst_path}")

    # Calculate regime change dates for various sensitivities (num_prev_colors)
    regime_params = [6, 7, 8] # All num_prev_colors values to be used in experiments
    all_regime_dates = {}
    print(f"Calculating regime change dates for sensitivities: {regime_params}...")
    for num_prev in tqdm(regime_params):
        dates = detect_regimes(hurst_df, num_prev_colors=num_prev)
        all_regime_dates[f'sensitivity_{num_prev}'] = dates
    
    regime_path = os.path.join(CACHE_DIR, 'regime_change_dates.json')
    with open(regime_path, 'w') as f:
        json.dump(all_regime_dates, f, indent=4)
    print(f"Regime change dates saved to {regime_path}")

    print("\n✅ Data preparation complete. You can now run experiments much faster.")

if __name__ == '__main__':
    main()

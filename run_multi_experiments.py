# run_multi_experiments.py
import argparse
import pandas as pd
import numpy as np
import os
from itertools import product

from engine import ModelManager
from config import CONFIG

import numpy as np
import pandas as pd

def calculate_metrics(df):
    """Calculates performance metrics from a results dataframe."""
    metrics = {}
    preds = df.filter(like='Prediction').values
    acts = df.filter(like='Actual').values
    
    errors = preds - acts
    
    # --- RMSE and MAE ---
    metrics['RMSE'] = np.sqrt(np.mean(errors**2))
    metrics['MAE'] = np.mean(np.abs(errors))
    
    # --- MAPE (added) ---
    # Using eps=1e-8 to prevent division by zero, consistent with your example
    metrics['MAPE'] = np.mean(np.abs(errors / (acts + 1e-8))) * 100.0
    
    # --- Correlation (added) ---
    # Calculates correlation for each asset and then averages the results.
    correlations = []
    num_assets = acts.shape[1]
    for i in range(num_assets):
        y_true_asset = acts[:, i]
        y_pred_asset = preds[:, i]
        
        # Handle cases with zero variance to avoid errors
        if np.std(y_true_asset) == 0 or np.std(y_pred_asset) == 0:
            correlations.append(np.nan)
        else:
            corr_matrix = np.corrcoef(y_true_asset, y_pred_asset)
            correlations.append(corr_matrix[0, 1])
            
    # Use nanmean to safely average, ignoring any assets that had zero variance
    metrics['Correlation'] = np.nanmean(correlations)
    
    # --- Hit Ratio (directional accuracy) ---
    # This remains the same as it correctly calculates directional changes.
    pred_diff = np.sign(np.diff(preds, axis=0))
    act_diff = np.sign(np.diff(acts, axis=0))
    metrics['Hit_Ratio'] = np.mean(pred_diff == act_diff)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run multiple volatility prediction experiments and summarize results. (Training period is fixed to 750.)")
    parser.add_argument('--runs', type=int, default=5, help='Number of times to repeat each experiment.')
    
    args = parser.parse_args()
    
    # --- Experiment Grid ---
    experiment_grid = {
        'model_type': ['gcn-ete', 'gcn-te', 'gcn-granger', 'gcn-pearson', 'lstm', 'gru'],
        'window_size': [20, 40, 60],
        'use_regime_detection': [True, False]
    }
    
    keys, values = zip(*experiment_grid.items())
    experiments = [dict(zip(keys, v)) for v in product(*values)]
    
    all_results = []

    # --- Load Data Once ---
    log_return_df = pd.read_csv(CONFIG.LOG_RETURN_PATH, parse_dates=['Date'])

    # --- Run All Experiments ---
    for i, params in enumerate(experiments):
        print(f"\n--- Running Experiment {i+1}/{len(experiments)}: {params} ---")
        
        run_metrics = []
        for run in range(args.runs):
            print(f"  Run {run+1}/{args.runs}...")
            manager = ModelManager(
                data=log_return_df,
                model_type=params['model_type'],
                window_size=params['window_size'],
                use_regime_detection=params['use_regime_detection'],
                training_period=750 # Fixed for consistency
            )
            manager.run_experiment()
            results_df = manager.get_results_df()
            metrics = calculate_metrics(results_df)
            run_metrics.append(metrics)
        
        # Aggregate results for this experiment configuration
        df_metrics = pd.DataFrame(run_metrics)
        summary = {
            'model': params['model_type'],
            'window_size': params['window_size'],
            'regime_detection': params['use_regime_detection'],
            'RMSE_mean': df_metrics['RMSE'].mean(),
            'RMSE_std': df_metrics['RMSE'].std(),
            'MAE_mean': df_metrics['MAE'].mean(),
            'MAE_std': df_metrics['MAE'].std(),
            'Hit_Ratio_mean': df_metrics['Hit_Ratio'].mean(),
            'Hit_Ratio_std': df_metrics['Hit_Ratio'].std(),
        }
        all_results.append(summary)

    # --- Save Summary ---
    summary_df = pd.DataFrame(all_results)
    summary_path = "multi_experiment_summary.xlsx"
    summary_df.to_excel(summary_path, index=False)
    
    print(f"\n✅ All experiments complete. Summary saved to {summary_path}")
    print(summary_df)

if __name__ == '__main__':
    main()
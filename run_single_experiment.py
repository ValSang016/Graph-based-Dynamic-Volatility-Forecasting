# run_single_experiment.py
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import os
from datetime import datetime

from engine import ModelManager
from config import CONFIG
import utils

def calculate_metrics(df):
    """Calculates performance metrics from a results dataframe."""
    metrics = {}
    preds = df.filter(like='Prediction').values
    acts = df.filter(like='Actual').values
    
    errors = preds - acts
    
    # --- RMSE and MAE ---
    metrics['RMSE'] = np.sqrt(np.mean(errors**2))
    metrics['MAE'] = np.mean(np.abs(errors))
    
    # --- MAPE ---
    metrics['MAPE'] = np.mean(np.abs(errors / (acts + 1e-8))) * 100.0
    
    # --- Correlation ---
    correlations = []
    for i in range(acts.shape[1]):
        y_true, y_pred = acts[:, i], preds[:, i]
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            correlations.append(np.nan)
        else:
            correlations.append(np.corrcoef(y_true, y_pred)[0, 1])
    metrics['Correlation'] = np.nanmean(correlations)
    
    # --- Hit Ratio (directional accuracy) ---
    pred_diff = np.sign(np.diff(preds, axis=0))
    act_diff = np.sign(np.diff(acts, axis=0))
    metrics['Hit_Ratio'] = np.mean(pred_diff == act_diff)
    
    # Return as a pandas Series for easy integration with groupby().apply()
    return pd.Series(metrics)

def plot_detailed_results(results_df, model_name, save_dir):
    """Generates and saves detailed plots for a single run, including all assets."""
    
    # --- 1. Plot Actual vs. Predicted for ALL assets ---
    num_assets = len([col for col in results_df.columns if col.startswith('Actual_')])
    print(f"Generating prediction plots for {num_assets} assets...")
    for i in range(1, num_assets + 1):
        plt.figure(figsize=(15, 7))
        plt.plot(results_df['Date'], results_df[f'Actual_{i}'], label='Actual Volatility', color='navy')
        plt.plot(results_df['Date'], results_df[f'Prediction_{i}'], label='Predicted Volatility', color='orangered', alpha=0.9)
        plt.title(f'Volatility Prediction for Asset {i} ({model_name})')
        plt.xlabel('Date')
        plt.ylabel('Realized Volatility')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_vs_actual_asset_{i}.png'))
        plt.close()

    # --- 2. Plot Hurst Exponent with Regime Changes ---
    hurst_df = utils.compute_hurst_series(CONFIG.WORLD_INDEX_PATH)
    change_dates = utils.detect_regimes(hurst_df)
    
    plt.figure(figsize=(15, 7))
    plt.plot(hurst_df.index, hurst_df['Hurst'], label='Hurst Exponent', color='black')
    for date in change_dates:
        plt.axvline(pd.to_datetime(date), color='red', linestyle='--', linewidth=1, label='Regime Change' if date == change_dates[0] else "")
    plt.axhline(0.5, color='gray', linestyle=':', label='H=0.5 (Random Walk)')
    plt.title('Hurst Exponent and Detected Regime Changes')
    plt.xlabel('Date')
    plt.ylabel('Hurst Value')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hurst_exponent.png'))
    plt.close()
    
    print(f"All detailed plots have been saved to the directory: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run a single volatility prediction experiment.")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['gcn-ete', 'gcn-te', 'gcn-granger', 'gcn-pearson', 'lstm', 'gru'],
                        help='Model to run.')
    parser.add_argument('--window_size', type=int, default=40, help='Input window size for the model.')
    parser.add_argument('--training_period', type=int, default=750, help='Length of the training data window.')
    parser.add_argument('--no_regime', action='store_true', help='Disable Hurst-based regime detection and fine-tuning.')
    
    args = parser.parse_args()

    # --- Setup ---
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"results/results_single_{args.model}_{now}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting single experiment for model: {args.model}")
    print(f"Results will be saved in: {save_dir}")

    # --- Load Data ---
    log_return_df = pd.read_csv(CONFIG.LOG_RETURN_PATH, parse_dates=['Date'])

    # --- Initialize and Run Manager ---
    manager = ModelManager(
        data=log_return_df,
        model_type=args.model,
        window_size=args.window_size,
        training_period=args.training_period,
        use_regime_detection=not args.no_regime,
        debug=True
    )
    manager.run_experiment()

    # --- Process and Save Results ---
    results_df = manager.get_results_df()
    
    # Add a 'Year' column for yearly analysis
    results_df['Year'] = pd.to_datetime(results_df['Date']).dt.year

    # Calculate yearly metrics
    yearly_metrics = results_df.groupby('Year').apply(calculate_metrics).reset_index()

    # Calculate overall metrics
    overall_metrics = calculate_metrics(results_df)
    overall_metrics['Year'] = 'Overall' # Label this row as 'Overall'
    
    # Combine yearly and overall metrics into one DataFrame
    summary_df = pd.concat([yearly_metrics, overall_metrics.to_frame().T], ignore_index=True)
    
    print("\n--- Performance Summary ---")
    print(summary_df.to_string())
    
    # Save both predictions and metrics to a single Excel file with two sheets
    excel_path = os.path.join(save_dir, 'experiment_results.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        results_df.to_excel(writer, sheet_name='Predictions', index=False)
        summary_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)
    
    print(f"\nPredictions and performance metrics saved to {excel_path}")

    # --- Generate Plots ---
    plot_detailed_results(results_df, args.model, save_dir)
    
    print("\n✅ Single experiment finished successfully.")

if __name__ == '__main__':
    if CONFIG.CUDA_AVAILABLE:
        torch.cuda.set_device(CONFIG.GPU)
        print("Cuda device is set to GPU:", CONFIG.GPU)
    else : 
        print("Cuda device is set to CPU")
    main()

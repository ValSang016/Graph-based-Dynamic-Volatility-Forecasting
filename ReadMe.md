# Volatility Prediction with GCN and RNN Models using Regime Detection and Non-linear/Linear causality

This project implements and evaluates various deep learning models (GCN, LSTM, GRU) for predicting stock market volatility. It features a dynamic approach where models can be fine-tuned based on market regime changes detected using the Hurst exponent.

## Reference
This repository is the official implementation of the following paper:

- Lee, Sangheon, and Poongjin Cho. "Graph-Based Stock Volatility Forecasting with Effective Transfer Entropy and Hurst-Based Regime Adaptation." Fractal and Fractional 9.6 (2025): 339. https://doi.org/10.3390/fractalfract9060339


##  Project Structure

-   `/data`: Contains the raw time-series data.
-   `config.py`: Central configuration file for all hyperparameters.
-   `models.py`: Definitions for all PyTorch models (TENet(Gcn), LSTM, GRU).
-   `dataset.py`: Data loading and preprocessing classes.
-   `utils.py`: Helper functions for Hurst exponent calculation, regime detection, and adjacency matrix creation.
-   `engine.py`: The core `ModelManager` class that orchestrates the training and evaluation pipeline.
-   `run_single_experiment.py`: Script to run a single, detailed experiment and generate visualizations.
-   `run_multi_experiments.py`: Script to run a batch of all model configurations for multiple iterations and summarize the results.
-   `requirements.txt`: A list of all Python dependencies.

##  Getting Started

### 1. Setup

First, clone the repository and create a virtual environment:

```bash
git clone https://github.com/ValSang016/Graph-based-Dynamic-Volatility-Forecasting.git
python -m venv venv
cd Graph-based-Dynamic-Volatility-Forecasting
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Next, install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Running a Single Experiment

To run one specific model configuration and see detailed output plots (e.g., actual vs. predicted volatility), use `run_single_experiment.py`. This is ideal for deep-diving into a specific model's performance.

**Usage:**

```bash
python run_single_experiment.py --model <model_name> [options]
```

**Available Models:**
-   `gcn-ete`
-   `gcn-te`
-   `gcn-granger`
-   `gcn-pearson`
-   `lstm`
-   `gru`

**Example:**

Run an LSTM model with a window size of 40 and regime detection enabled:

```bash
python run_single_experiment.py --model lstm --window_size 40
```

Run a GCN-ETE model without regime detection:

```bash
python run_single_experiment.py --model gcn-ete --no_regime --window_size 20
```

The results, including plots and an Excel file with predictions, will be saved to a new directory named `results_single_<model_name>_<timestamp>`.

### 3. Running Multiple Experiments for Comparison

To systematically evaluate all models and configurations, use `run_multi_experiments.py`. This script will loop through all predefined models, run each one multiple times, and generate a summary Excel file with the mean and standard deviation of key performance metrics (RMSE, MAE, Hit Ratio).

**Usage:**

```bash
python run_multi_experiments.py --runs <number_of_runs>
```

**Example:**

Run every model configuration 5 times and generate a summary:

```bash
python run_multi_experiments.py --runs 5
```

The final output will be a single Excel file named `multi_experiment_summary.xlsx` containing the aggregated results, perfect for comparing model performance.

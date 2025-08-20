# engine.py
import os
import gc
import copy
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import optuna
from tqdm import tqdm

import models
import utils
from dataset import WindowedDataset
from config import CONFIG

class ModelManager:
    def __init__(self, data, model_type='lstm', window_size=20, training_period=750, 
                 use_regime_detection=True, num_prev_colors=8, use_cache=True, debug=False):
        
        self.raw_data = data
        self.model_type = model_type.lower()
        self.window_size = window_size
        self.training_period = training_period
        self.use_regime_detection = use_regime_detection
        self.num_prev_colors = num_prev_colors
        self.use_cache = use_cache
        self.debug = debug

        self.device = torch.device(CONFIG.DEVICE)
        self.criterion = nn.MSELoss().to(self.device)
        self.model = None

        # Data setup
        self.data_np = np.array(data.iloc[:, 1:])
        self._load_data_from_cache_or_compute() # Load from cache or compute on-the-fly
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        # Results storage
        self.predictions = []
        self.actuals = []
        self.best_params = {}

    def _load_data_from_cache_or_compute(self):
        """Loads pre-calculated data from cache, or computes it if files are not found."""
        cache_dir = 'cache'
        vol_path = os.path.join(cache_dir, 'realized_volatility_full.csv')
        hurst_path = os.path.join(cache_dir, 'hurst_series.csv')
        regime_path = os.path.join(cache_dir, 'regime_change_dates.json')

        # 1. Load Realized Volatility
        if self.use_cache and os.path.exists(vol_path):
            if self.debug: print("Loading cached realized volatility...")
            full_volatility_series = pd.read_csv(vol_path).values
        else:
            if self.debug: print("Cache not found. Calculating realized volatility on-the-fly...")
            full_volatility_series = self._calculate_full_realized_volatility(self.data_np)
        
        # Slice the full volatility series according to the experiment's window_size
        # Original logic: result = realized_vols[window-20:]
        self.data_y = full_volatility_series[self.window_size - 20:]

        # 2. Load Regime Change Dates
        if self.use_regime_detection:
            if self.use_cache and os.path.exists(hurst_path) and os.path.exists(regime_path):
                if self.debug: print("Loading cached regime change dates...")
                with open(regime_path, 'r') as f:
                    all_regimes = json.load(f)
                # Select the regime change dates corresponding to the current experiment's sensitivity setting
                self.change_dates = all_regimes.get(f'sensitivity_{self.num_prev_colors}', [])
            else:
                if self.debug: print("Cache not found. Calculating regime change dates on-the-fly...")
                hurst_df = utils.compute_hurst_series(CONFIG.WORLD_INDEX_PATH)
                self.change_dates = utils.detect_regimes(hurst_df, self.num_prev_colors)
        
        if self.debug and self.use_regime_detection:
            print(f"Using {len(self.change_dates)} regime change dates for sensitivity '{self.num_prev_colors}'.")

    def _calculate_full_realized_volatility(self, data_np, lookback=20):
        """On-the-fly calculation function in case cache files are missing."""
        realized_vols = []
        for i in range(lookback, len(data_np)):
            window_data = data_np[i-lookback:i] * 100
            squared_returns = np.square(window_data)
            realized_vols.append(np.sqrt(np.mean(squared_returns, axis=0)))
        return np.array(realized_vols)

    def _build_model(self, params):
        """Builds and returns the appropriate model based on self.model_type."""
        input_size = self.data_np.shape[1]
        output_size = input_size

        if self.model_type.startswith('gcn'):
            matrix_method = self.model_type.split('-')[1]
            initial_matrix = utils.get_adjacency_matrix(self.train_x_initial, method=matrix_method)
            model = models.TENet(
                CONFIG, initial_matrix, self.window_size,
                hid1=params['hid1'], hid2=params['hid2']
            ).to(self.device)
        elif self.model_type == 'lstm':
            model = models.LSTMModel(
                input_size, params['hidden_size'], params['num_layers'],
                output_size, params['dropout']
            ).to(self.device)
        elif self.model_type == 'gru':
            model = models.GRUModel(
                input_size, params['hidden_size'], params['num_layers'],
                output_size, params['dropout']
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return model

    def _train_validate_step(self, model, train_loader, val_loader, lr):
        """A single training and validation cycle for a model."""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(CONFIG.EPOCHS):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                preds = model(x_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                    preds = model(x_val)
                    total_val_loss += self.criterion(preds, y_val).item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
        
        return best_model_state, best_val_loss

    def _objective(self, trial, train_dataset, val_dataset):
        """Optuna objective function."""
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16])
        }
        if self.model_type.startswith('gcn'):
            params["hid1"] = trial.suggest_int("hid1", 10, 100)
            params["hid2"] = trial.suggest_int("hid2", 10, 100)
        else: # LSTM/GRU
            params["hidden_size"] = trial.suggest_int("hidden_size", 10, 100)
            params["num_layers"] = trial.suggest_int("num_layers", 1, 5)
            params["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)

        model = self._build_model(params)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        best_model_state, val_loss = self._train_validate_step(model, train_loader, val_loader, params['lr'])
        trial.set_user_attr("best_model_state", best_model_state)
        
        del model, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

        return val_loss

    def fine_tune(self, train_dataset, val_dataset):
        # Fine-tuning logic (retraining with fewer epochs)
        train_loader = DataLoader(train_dataset, batch_size=self.best_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.best_params['batch_size'], shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.best_params['lr'])
        
        for epoch in range(CONFIG.FINE_TUNE_EPOCHS):
            self.model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                preds = self.model(x_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
            
    def run_experiment(self):
        """Main entry point to run the entire experiment workflow."""
        train_split = int(self.training_period * 0.7)
        
        # Initial data split for hyperparameter tuning
        self.train_x_initial = self.data_np[:train_split + self.window_size]
        val_x_initial = self.data_np[train_split : self.training_period + self.window_size]
        train_y_initial = self.data_y[:train_split]
        val_y_initial = self.data_y[train_split : self.training_period]

        self.scaler_x.fit(np.ascontiguousarray(self.train_x_initial))
        self.scaler_y.fit(np.ascontiguousarray(train_y_initial))

        train_ds = WindowedDataset(self.train_x_initial, train_y_initial, self.window_size, CONFIG.HORIZON, self.scaler_x, self.scaler_y)
        val_ds = WindowedDataset(val_x_initial, val_y_initial, self.window_size, CONFIG.HORIZON, self.scaler_x, self.scaler_y)

        # 1. Find best hyperparameters with Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self._objective(trial, train_ds, val_ds), n_trials=CONFIG.N_TRIALS)
        
        self.best_params = study.best_params
        best_model_state = study.best_trial.user_attrs["best_model_state"]
        
        # 2. Build and load the best model
        self.model = self._build_model(self.best_params)
        self.model.load_state_dict(best_model_state)

        # 3. Walk-forward prediction and retraining
        for day in tqdm(range(self.training_period, len(self.data_np) - self.window_size), desc=f"Predicting {self.model_type}"):
            current_date = self.raw_data['Date'].iloc[day].strftime('%Y-%m-%d')
            
            # Detect regime change and fine-tune the model
            if self.use_regime_detection and current_date in self.change_dates:
                if self.debug: print(f"\nRegime change detected on {current_date}. Fine-tuning model.")
                start_idx = max(0, day - self.training_period)
                retrain_x_data = self.data_np[start_idx : day + self.window_size]
                retrain_y_data = self.data_y[start_idx : day]
                
                self.scaler_x.fit(np.ascontiguousarray(retrain_x_data))
                self.scaler_y.fit(np.ascontiguousarray(retrain_y_data))
                
                retrain_ds = WindowedDataset(retrain_x_data, retrain_y_data, self.window_size, CONFIG.HORIZON, self.scaler_x, self.scaler_y)

                if self.model_type.startswith('gcn'):
                    matrix_method = self.model_type.split('-')[1]
                    new_matrix = utils.get_adjacency_matrix(retrain_x_data, method=matrix_method)
                    self.model.update_A(new_matrix)

                self.fine_tune(retrain_ds, retrain_ds)

            # Perform prediction for the current day
            test_x = self.data_np[day : day + self.window_size]
            test_y = self.data_y[day]
            
            scaled_x = self.scaler_x.transform(np.ascontiguousarray(test_x))
            x_tensor = torch.tensor(scaled_x[None, :, :], dtype=torch.float32).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                prediction_scaled = self.model(x_tensor)

            prediction_unscaled = self.scaler_y.inverse_transform(prediction_scaled.cpu().numpy())
            
            self.predictions.append(prediction_unscaled.flatten())
            self.actuals.append(test_y)
        
        del study, train_ds, val_ds
        gc.collect()
        torch.cuda.empty_cache()

    def get_results_df(self):
        """Returns a DataFrame containing predictions, actuals, and performance metrics."""
        preds = np.array(self.predictions)
        acts = np.array(self.actuals)
        
        dates = self.raw_data['Date'].iloc[self.training_period : self.training_period + len(preds)]
        
        pred_cols = {f'Prediction_{i+1}': preds[:, i] for i in range(preds.shape[1])}
        act_cols = {f'Actual_{i+1}': acts[:, i] for i in range(acts.shape[1])}
        
        df = pd.DataFrame(index=dates)
        df = df.assign(**pred_cols, **act_cols)
        
        return df.reset_index().rename(columns={'index': 'Date'})

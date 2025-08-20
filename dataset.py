# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

def calculate_realized_volatility(data, window_size, lookback=20):
    """Calculates realized volatility for each asset."""
    realized_vols = []
    for i in range(lookback, len(data)):
        window_data = data[i-lookback:i] * 100
        squared_returns = np.square(window_data)
        realized_vol = np.sqrt(np.mean(squared_returns, axis=0))
        realized_vols.append(realized_vol)
    
    # Align target data with input data
    return np.array(realized_vols[window_size-lookback:])

class WindowedDataset(Dataset):
    def __init__(self, data_x, data_y, window_size, horizon, scaler_x, scaler_y):
        self.data_x = data_x
        self.data_y = data_y
        self.window_size = window_size
        self.horizon = horizon
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
    
    def __len__(self):
        return max(0, len(self.data_x) - self.window_size - self.horizon + 1)
    
    def __getitem__(self, idx):
        window_x = self.data_x[idx : idx + self.window_size]
        label = self.data_y[idx]

        window_scaled = self.scaler_x.transform(window_x)
        label_scaled = self.scaler_y.transform(np.array([label]))
        
        return torch.tensor(window_scaled, dtype=torch.float32), torch.tensor(label_scaled.flatten(), dtype=torch.float32)
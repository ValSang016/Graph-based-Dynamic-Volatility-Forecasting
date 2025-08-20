# config.py
import torch

class Config:
    # -- Data Parameters --
    N_E = 10  # Number of assets/nodes
    HORIZON = 1
    
    # -- GCN Model (TENet) Parameters --
    K_SIZE = [3, 5, 7]
    CHANNEL_SIZE = 12

    # -- General Training Parameters --
    EPOCHS = 100         # Epochs for initial training
    FINE_TUNE_EPOCHS = 20 # Epochs for fine-tuning after regime change
    
    # -- Optuna Parameters --
    N_TRIALS = 20 # Number of Optuna trials for hyperparameter search

    # -- Device Configuration --
    GPU = 0
    DEVICE = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
    CUDA_AVAILABLE = torch.cuda.is_available()

    # -- Data Paths --
    LOG_RETURN_PATH = './data/dataset/log_df_etf10.csv'
    WORLD_INDEX_PATH = './data/dataset/df_world.csv'

# Instantiate the configuration
CONFIG = Config()
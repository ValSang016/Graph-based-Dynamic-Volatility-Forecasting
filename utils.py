# utils.py
import pandas as pd
import numpy as np
from hurst import compute_Hc
from statsmodels.tsa.stattools import grangercausalitytests

# --- Regime Detection ---
def compute_hurst_series(world_index_path='data/df_world.csv', window=250):
    df = pd.read_csv(world_index_path, parse_dates=['Date'], index_col='Date')
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    hurst_vals, dates = [], []
    for i in range(window, len(df)):
        series = df['Log_Return'].iloc[i-window:i].dropna()
        H, _, _ = compute_Hc(series, kind='change', simplified=True)
        hurst_vals.append(H)
        dates.append(df.index[i])
    
    return pd.DataFrame({'Hurst': hurst_vals}, index=dates)

def detect_regimes(hurst_df, num_prev_colors=8, thresholds=0.5):
    change_dates = []
    prev_color = 'blue' # Start with a neutral color
    
    for i in range(10, len(hurst_df)):
        recent_hurst = hurst_df['Hurst'].iloc[i-10:i]
        
        is_trending = (recent_hurst >= thresholds).sum() >= num_prev_colors
        is_reverting = (recent_hurst < thresholds).sum() >= num_prev_colors
        
        color = 'red' if is_trending else 'blue' if is_reverting else prev_color
        
        if color != prev_color:
            change_dates.append(hurst_df.index[i].strftime('%Y-%m-%d'))
            prev_color = color
            
    return change_dates

# --- Adjacency Matrix Calculations ---
def _value_to_bin(value_list, m, M):
    width = (M - m) / 3.0
    return [min(int((v - m) / width), 2) for v in value_list]

def _compute_TE(X):
    m, M = np.min(X), np.max(X)
    n = X.shape[1]
    TE = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            xn1, xn, yn = X[1:, j], X[:-1, j], X[:-1, i]
            xn1_b, xn_b, yn_b = _value_to_bin(xn1, m, M), _value_to_bin(xn, m, M), _value_to_bin(yn, m, M)
            
            x_freq = np.unique(xn_b, return_counts=True)
            x1_freq = np.unique(np.stack([xn_b, xn1_b], axis=1), return_counts=True, axis=0)
            y_freq = np.unique(np.stack([xn_b, yn_b], axis=1), return_counts=True, axis=0)
            xy_freq = np.unique(np.stack([xn_b, xn1_b, yn_b], axis=1), return_counts=True, axis=0)
            
            p_x, p_xx1, p_xy, p_xyz = (f[1] / f[1].sum() for f in [x_freq, x1_freq, y_freq, xy_freq])
            
            te = 0.0
            for k, triple in enumerate(xy_freq[0]):
                p_xyz_k = p_xyz[k]
                idx_xx1 = np.where((x1_freq[0] == triple[:2]).all(axis=1))[0][0]
                idx_xy  = np.where((y_freq[0] == triple[[0,2]]).all(axis=1))[0][0]
                idx_x   = np.where(x_freq[0] == triple[0])[0][0]
                te += p_xyz_k * np.log2((p_xyz_k * p_x[idx_x]) / (p_xx1[idx_xx1] * p_xy[idx_xy] + 1e-9) + 1e-9)
            TE[i, j] = te
    return TE

def _compute_ETE(X, TE_matrix, n_random):
    m, M = np.min(X), np.max(X)
    n = X.shape[1]
    RTE_matrix_all = np.zeros([X.shape[1], X.shape[1], n_random])
    for nn in range(n_random):
        RTE_matrix = np.zeros((X.shape[1], X.shape[1]))
        for i in range(n):
            for j in range(n):
                if i == j: continue
                xn1, xn, yn = X[1:, j], X[:-1, j], X[:-1, i]
                xn1_b, xn_b, yn_b = _value_to_bin(xn1, m, M), _value_to_bin(xn, m, M), _value_to_bin(yn, m, M)
                np.random.shuffle(yn_b)

                x_freq = np.unique(xn_b, return_counts=True)
                x1_freq = np.unique(np.stack([xn_b, xn1_b], axis=1), return_counts=True, axis=0)
                y_freq = np.unique(np.stack([xn_b, yn_b], axis=1), return_counts=True, axis=0)
                xy_freq = np.unique(np.stack([xn_b, xn1_b, yn_b], axis=1), return_counts=True, axis=0)
                
                p_x, p_xx1, p_xy, p_xyz = (f[1] / f[1].sum() for f in [x_freq, x1_freq, y_freq, xy_freq])
                
                RTE_xy = 0
                for k, triple in enumerate(xy_freq[0]):
                    p_xyz_k = p_xyz[k]
                    idx_xx1 = np.where((x1_freq[0] == triple[:2]).all(axis=1))[0][0]
                    idx_xy  = np.where((y_freq[0] == triple[[0,2]]).all(axis=1))[0][0]
                    idx_x   = np.where(x_freq[0] == triple[0])[0][0]
                    RTE_xy += p_xyz_k * np.log2((p_xyz_k * p_x[idx_x]) / (p_xx1[idx_xx1] * p_xy[idx_xy] + 1e-9) + 1e-9)
                RTE_matrix[i, j] = RTE_xy
        RTE_matrix_all[:, :, nn] = RTE_matrix
        
    ETE_matrix = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if i == j:
                continue
            TE = TE_matrix[i, j]
            rte_array = RTE_matrix_all[i, j, :]
            if TE - np.mean(rte_array) - np.std(rte_array) / (n_random ** 0.5) > 0:
                ETE_matrix[i, j] = TE - np.mean(rte_array)
                
    if np.all(ETE_matrix == 0):
        matrix = np.full((10, 10), 0.001)
        np.fill_diagonal(matrix, 0)
        return matrix

    return ETE_matrix

def get_adjacency_matrix(data, method='ete', n_random=25):
    """Factory function to get the specified adjacency matrix."""
    if method == 'te':
        return _compute_TE(data)
    elif method == 'ete':
        te_matrix = _compute_TE(data)
        return _compute_ETE(data, te_matrix, n_random)
    elif method == 'granger':
        df = pd.DataFrame(data)
        gc_matrix = np.zeros((df.shape[1], df.shape[1]))
        for i in range(df.shape[1]):
            for j in range(df.shape[1]):
                if i != j:
                    test_result = grangercausalitytests(df[[j, i]], maxlag=15, verbose=False)
                    p_values = [round(test_result[lag][0]['ssr_chi2test'][1], 4) for lag in range(1, 16)]
                    gc_matrix[i, j] = 1 - np.min(p_values) # Invert p-value to show strength
        return gc_matrix
    elif method == 'pearson':
        corr_matrix = pd.DataFrame(data).corr().abs().values
        np.fill_diagonal(corr_matrix, 0)
        return corr_matrix
    else:
        raise ValueError(f"Unknown matrix method: {method}")
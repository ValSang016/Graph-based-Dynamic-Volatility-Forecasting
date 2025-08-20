# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layer import DenseGraphConv

class TENet(nn.Module):
    def __init__(self, args, A, window_size, hid1, hid2):
        super(TENet, self).__init__()
        self.n_e = args.N_E
        self.window_size = window_size

        # Normalize and store adjacency matrix A
        A_norm = self._normalize_A(np.array(A, dtype=np.float32))
        self.A = nn.Parameter(torch.from_numpy(A_norm), requires_grad=False)

        self.conv1 = nn.Conv2d(1, args.CHANNEL_SIZE, kernel_size=(1, args.K_SIZE[0]))
        self.conv2 = nn.Conv2d(1, args.CHANNEL_SIZE, kernel_size=(1, args.K_SIZE[1]))
        self.conv3 = nn.Conv2d(1, args.CHANNEL_SIZE, kernel_size=(1, args.K_SIZE[2]))

        d = (len(args.K_SIZE) * self.window_size - sum(args.K_SIZE) + len(args.K_SIZE)) * args.CHANNEL_SIZE
        
        self.gnn1 = DenseGraphConv(d, hid1)
        self.gnn2 = DenseGraphConv(hid1, hid2)
        self.gnn3 = DenseGraphConv(hid2, 1)

    def _normalize_A(self, A):
        A = (A - A.min()) / (A.max() - A.min() + 1e-9)
        np.fill_diagonal(A, 1)
        return A

    def forward(self, x):
        batch_size = x.size(0)
        A_expanded = self.A.unsqueeze(0).repeat(batch_size, 1, 1)
        
        c = x.permute(0, 2, 1).unsqueeze(1)
        
        a1 = self.conv1(c).permute(0, 2, 1, 3).reshape(batch_size, self.n_e, -1)
        a2 = self.conv2(c).permute(0, 2, 1, 3).reshape(batch_size, self.n_e, -1)
        a3 = self.conv3(c).permute(0, 2, 1, 3).reshape(batch_size, self.n_e, -1)
        
        x_conv = F.relu(torch.cat([a1, a2, a3], dim=2))
        
        x1 = F.relu(self.gnn1(x_conv, A_expanded))
        x2 = F.relu(self.gnn2(x1, A_expanded))
        x3 = self.gnn3(x2, A_expanded)
        
        return x3.squeeze(-1)

    def update_A(self, new_A):
        new_A_norm = self._normalize_A(np.array(new_A, dtype=np.float32))
        with torch.no_grad():
            self.A.copy_(torch.from_numpy(new_A_norm))

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take the output from the last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, 
                          dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :] # Take the output from the last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        return self.net(x)


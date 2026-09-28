import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)
torch.manual_seed(42)

class BasinClassifier(nn.Module):
    def __init__(self, input_dim=2, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)


import os
import torch
import random
import time

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from QuickDataset import QuickDataset2
from mamba_ssm import Mamba
from include.mama import JeuralJetwork
from mamba_ssm.ops.triton.layer_norm import RMSNorm

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=28,
        d_conv=4,
        expand=2,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))

class NeuralNetwork(nn.Module):
    def __init__(self, n_dim, out_dim, input_len, output_len, neural_count=64):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_dim = n_dim
        self.out_dim = out_dim

        #nn.LayerNorm(neural_count)
        self.input_proj = nn.Sequential(
            nn.Linear(n_dim, neural_count),
            RMSNorm(neural_count)
        )

        self.mamba = nn.Sequential(
                MambaBlock(d_model=neural_count),
                MambaBlock(d_model=neural_count),
        )

        self.head = nn.Sequential(
            nn.Linear(neural_count, neural_count),
            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(neural_count, neural_count),
            nn.GELU(),

            nn.Linear(neural_count, out_dim * output_len),
        )

    def forward(self, vec):
        x = self.input_proj(vec) # [B, T, D] -> [B, T, H]
        x = self.mamba(x) # [B, T, H]
        x = x[:, -1] # [B, H]
        #x = x.mean(dim=1)
        x = self.head(x) # [B, out_dim * output_len]
        
        #print(x.shape) #maybe train the data on sequences instead of windows?
        #x = x.mean(dim=1)
        
        x = x.view(vec.size(0), self.output_len, self.out_dim) # [B, output_len, out_dim]

        return x


"""AIA CNN-GRU architecture reused from Watchman77/solar-flare-aia-training.

The three classes are preserved from 19A2_Temporal_AIA_CNN_GRU_Cycle24_Training.ipynb.
Only the module wrapper is new. No upstream weights or normalization are reused.
The new 72-hour label/role contract is supplied by this repository's loader.
Provenance: results/aia72_canary_20261002/source_provenance.json.
"""

import torch
from torch import nn

DROPOUT = 0.30

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)

class FrameCNN(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(6, 32, 0.05),
            ConvBlock(32, 64, 0.05),
            ConvBlock(64, 128, 0.10),
            ConvBlock(128, 192, 0.10),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, embed_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.proj(self.encoder(x))

class TemporalAIACNNGRU(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=192):
        super().__init__()
        self.frame_encoder = FrameCNN(embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        b,t,c,h,w = x.shape
        z = self.frame_encoder(x.reshape(b*t,c,h,w)).reshape(b,t,-1)
        _, hlast = self.gru(z)
        return self.head(hlast[-1]).squeeze(-1)

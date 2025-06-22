# Core libraries
import math
import random
import logging

# Data handling
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# PyTorch ecosystem
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchdiffeq import odeint

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Project configuration
import config

###############################################################################
# 1. CONFIG & SETUP
###############################################################################
SEED = 42
SEQUENCE_LENGTH = 12
BATCH_SIZE = 16

torch.manual_seed(SEED)
np.random.seed(SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

###############################################################################
# 1. Model Architectures
###############################################################################
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Compute the div_term (same as in "Attention is All You Need")
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class SimpleFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.3):
        super(SimpleFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()  # Using GELU activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        # Residual connection + layer normalization
        return self.norm(x + out)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.3):
        super(EncoderLayer, self).__init__()
        # Using batch_first=True so that input is of shape [batch, seq_len, d_model]
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ffn = SimpleFeedForward(d_model, dim_feedforward, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention; returns output and attention weights (ignored here)
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + self.dropout(src2))
        src = self.ffn(src)
        return src

class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, n_head, dim_feedforward, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, dim_feedforward, dropout) 
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=256, n_head=10, num_layers=10, 
                 dim_feedforward=2048, dropout=0.3):
        """
        input_dim: Number of input features per time step.
        output_dim: Number of outputs (e.g. 1 for a univariate forecast).
        d_model: Embedding dimension.
        n_head: Number of attention heads.
        num_layers: Number of encoder layers.
        dim_feedforward: Hidden dimension of the feed-forward network.
        dropout: Dropout probability.
        """
        super(Transformer, self).__init__()
        # Choose device based on availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Embedding: projects from input_dim to d_model
        self.src_tok_embedding = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pos_embedding = SinusoidalPositionalEncoding(d_model)
        
        # The encoder expects input in shape [seq_len, batch, d_model]. We will perform permutation.
        self.encoder = Encoder(d_model, num_layers, n_head, dim_feedforward, dropout)
        # Final fully connected layer maps from d_model to output_dim
        self.final_fc = nn.Linear(d_model, output_dim)
        
        self.initialize_weights()
        self.to(self.device)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, src):
        """
        src: [batch, seq_len, input_dim]
        Steps:
          1. Apply token embedding and normalization.
          2. Add positional encoding.
          3. Permute for the encoder: [seq_len, batch, d_model].
          4. Pass through encoder.
          5. Permute back to [batch, seq_len, d_model].
          6. Use the last time step and map to output_dim.
        """
        # (1) Embed and normalize:
        x = self.src_tok_embedding(src)   # -> [batch, seq_len, d_model]
        x = self.norm(x)
        # (2) Add positional encoding:
        x = self.pos_embedding(x)           # still [batch, seq_len, d_model]
        # (3) Permute to [seq_len, batch, d_model] for the encoder:
        x = x.permute(1, 0, 2)
        # (4) Pass through encoder:
        x = self.encoder(x)
        # (5) Permute back to [batch, seq_len, d_model]:
        x = x.permute(1, 0, 2)
        # (6) Use the last time step (or other pooling) and project to output_dim:
        last_time_step = x[:, -1, :]       # [batch, d_model]
        output = self.final_fc(last_time_step)  # [batch, output_dim]
        return output

# Optimizer Configuration
def configure_optimizer(model, learning_rate=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    return optimizer, scheduler

# --- Set Random Seeds for Reproducibility ---
seed = 42  # You can choose any integer
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
cudnn.benchmark = False
torch.use_deterministic_algorithms(False)

###############################################################################
# 1. Positional Encoding
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        We'll add position embeddings for 'seq_len' steps.
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0).to(x.device)
        return x

###############################################################################
# 2. Seasonal-Trend Decomposition (Reflection)
###############################################################################
class SeriesDecomposition(nn.Module):
    """
    Uses reflection padding for consistent output length, avoiding dimension mismatch.
    Decomposes x into a 'trend' (moving_mean) and 'remainder'.
    """
    def __init__(self, kernel_size=5):
        super(SeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        # We'll do manual reflection padding; hence 'padding=0'
        self.moving_avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            count_include_pad=False
        )

    def forward(self, x):
        """
        x shape: (B, seq_len, d_model).
        Steps:
          1) permute => (B, d_model, seq_len)
          2) reflection pad
          3) apply avg pool => shape (B, d_model, seq_len)
          4) permute back => (B, seq_len, d_model)
          5) remainder => x_orig - moving_mean
        """
        B, seq_len, d_model = x.shape

        # Permute => (B, d_model, seq_len)
        x = x.permute(0, 2, 1)

        # Reflection pad: total pad on last dim = kernel_size - 1
        pad_l = self.kernel_size // 2
        pad_r = self.kernel_size - 1 - pad_l
        x_padded = F.pad(x, (pad_l, pad_r), mode='reflect')

        # Now do avg pooling
        moving_mean = self.moving_avg(x_padded)   # => (B, d_model, seq_len)
        # Permute back => (B, seq_len, d_model)
        moving_mean = moving_mean.permute(0, 2, 1)

        # Original x => (B, seq_len, d_model)
        x_orig = x.permute(0, 2, 1)
        remainder = x_orig - moving_mean
        return moving_mean, remainder

###############################################################################
# 3. Auto-Correlation Mechanism
###############################################################################
class AutoCorrelationLayer(nn.Module):
    """
    Simplified frequency-based correlation for each head.
    """
    def __init__(self, d_model, n_heads):
        super(AutoCorrelationLayer, self).__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection   = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection= nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)  # smaller dropout for smaller model

    def forward(self, queries, keys, values):
        """
        shapes: (B, seq_len, d_model)
        We'll do an FFT-based correlation for each head.
        """
        B, L_Q, D = queries.shape
        _, L_K, _ = keys.shape

        # Project
        queries = self.query_projection(queries).view(B, L_Q, self.n_heads, self.d_head)
        keys    = self.key_projection(keys).view(B, L_K, self.n_heads, self.d_head)
        values  = self.value_projection(values).view(B, L_K, self.n_heads, self.d_head)

        # FFT
        queries_fft = torch.fft.rfft(queries, dim=1)
        keys_fft    = torch.fft.rfft(keys,    dim=1)

        # correlation in freq domain
        corr = queries_fft * torch.conj(keys_fft)
        corr = torch.fft.irfft(corr, n=L_Q, dim=1)

        corr = torch.softmax(corr, dim=1)

        out = torch.einsum("blhd,bshd->blhd", corr, values)
        out = out.contiguous().view(B, L_Q, -1)
        out = self.output_projection(out)
        return out

###############################################################################
# 4. Autoformer Encoder Block
###############################################################################
class AutoformerEncoderLayer(nn.Module):
    """
    One smaller block:
      - Decomposition => trend + seasonal
      - Self-attention => seasonal => LN
      - 1D conv => feed-forward => LN
      - final decomposition => new trend + seasonal
    """
    def __init__(self, d_model, n_heads, d_ff, moving_avg_kernel, dropout=0.1):
        super(AutoformerEncoderLayer, self).__init__()
        self.decomp = SeriesDecomposition(moving_avg_kernel)
        self.self_attn = AutoCorrelationLayer(d_model, n_heads)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff,   out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x shape: (B, seq_len, d_model)
        """
        # Decompose
        trend1, seasonal1 = self.decomp(x)
        # Residual
        seasonal1 = seasonal1 + x

        # Self-attn
        attn_output = self.self_attn(seasonal1, seasonal1, seasonal1)
        seasonal2 = self.dropout(attn_output)
        seasonal2 = self.norm1(seasonal1 + seasonal2)

        # Feed-forward
        ff_output = self.conv1(seasonal2.permute(0, 2, 1))  
        ff_output = F.gelu(ff_output)
        ff_output = self.dropout(self.conv2(ff_output))
        ff_output = ff_output.permute(0, 2, 1)
        seasonal3 = self.norm2(seasonal2 + ff_output)

        # final decomposition
        trend2, seasonal3 = self.decomp(seasonal3)
        trend = trend1 + trend2
        return trend, seasonal3

###############################################################################
# 5. Autoformer Encoder
###############################################################################
class AutoformerEncoder(nn.Module):
    def __init__(self, layer, n_layers):
        super(AutoformerEncoder, self).__init__()
        self.layers = nn.ModuleList([layer for _ in range(n_layers)])

    def forward(self, x):
        """
        x shape: (B, seq_len, d_model).
        accumulates the trend
        """
        trend = torch.zeros_like(x)
        for layer in self.layers:
            trend_layer, x = layer(x)
            trend = trend + trend_layer
        return trend, x

###############################################################################
# 6. Smaller Autoformer Model
###############################################################################
class Autoformer(nn.Module):
    """
    A smaller Autoformer architecture for improved training stability
    and reduced overfitting.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=128,        # smaller than 256
        n_heads=4,         # fewer heads
        d_ff=256,          # smaller feed-forward
        e_layers=2,        # fewer encoder layers
        moving_avg_kernel=5,
        dropout=0.1,
        device='cpu'
    ):
        super(Autoformer, self).__init__()
        self.device = device

        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        # Build the smaller encoder
        encoder_layer = AutoformerEncoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            moving_avg_kernel=moving_avg_kernel,
            dropout=dropout
        )
        self.encoder = AutoformerEncoder(encoder_layer, e_layers)

        # Final projection
        self.projection = nn.Linear(d_model, output_dim)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x shape: (B, seq_len, input_dim)
        Steps:
          1) embedding => (B, seq_len, d_model)
          2) positional encoding => (B, seq_len, d_model)
          3) encoder => (trend, seasonal)
          4) final output from last time step => projection
        """
        x = self.embedding(x)                 # => (B, seq_len, d_model)
        x = self.positional_encoding(x)       # => (B, seq_len, d_model)
        trend, seasonal = self.encoder(x)     # => each (B, seq_len, d_model)
        # Use last time step
        output = trend[:, -1, :] + seasonal[:, -1, :]  # => (B, d_model)
        output = self.projection(output)               # => (B, output_dim)
        return output

# --- ProbSparse Attention ---
class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(ProbSparseAttention, self).__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1 / (self.d_head ** 0.5)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        B, L_Q, D = queries.shape
        _, L_K, _ = keys.shape

        # Linear projections
        queries = self.query_projection(queries).view(B, L_Q, self.n_heads, self.d_head)
        keys = self.key_projection(keys).view(B, L_K, self.n_heads, self.d_head)
        values = self.value_projection(values).view(B, L_K, self.n_heads, self.d_head)

        # Compute scaled dot-product attention
        scores = torch.einsum("blhd,bshd->bhls", queries, keys) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply sparsity by keeping top-k scores
        top_k = max(1, int(0.1 * L_K))  # Top 10% of keys
        sparse_indices = scores.topk(top_k, dim=-1)[1]
        sparse_mask = torch.zeros_like(scores).scatter_(-1, sparse_indices, 1)
        sparse_scores = scores * sparse_mask

        # Compute attention weights and apply to values
        attn_weights = F.softmax(sparse_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum("bhls,bshd->blhd", attn_weights, values)
        attn_output = attn_output.contiguous().view(B, L_Q, D)
        return self.output_projection(attn_output)

# --- Informer Encoder Block ---
class InformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super(InformerEncoderBlock, self).__init__()
        self.self_attn = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ProbSparse Attention
        attn_output = self.self_attn(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x

# --- ConvLayer for Input Feature Extraction ---
class ConvLayer(nn.Module):
    def __init__(self, d_model):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=3, 
            padding=1, 
            padding_mode='circular'
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x.permute(0, 2, 1)  # Switch to [batch_size, d_model, seq_len]
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1)  # Back to [batch_size, seq_len, d_model]
        return x

# --- Informer Encoder ---
class InformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, n_layers, dropout=0.1):
        super(InformerEncoder, self).__init__()
        self.conv_layer = ConvLayer(d_model)
        self.encoder_blocks = nn.ModuleList([
            InformerEncoderBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)
        ])

    def forward(self, x):
        # ConvLayer for feature extraction
        x = self.conv_layer(x)
        
        # Pass through each encoder block
        for block in self.encoder_blocks:
            x = block(x)
        return x

# --- Informer Model ---
class Informer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, n_heads, ff_dim, n_layers, dropout=0.1, device='cuda'):
        super(Informer, self).__init__()
        self.device = device
        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder = InformerEncoder(d_model, n_heads, ff_dim, n_layers, dropout)
        self.fc_out = nn.Linear(d_model, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Move input to the appropriate device
        x = x.to(self.device)

        # Embedding
        x = self.embedding(x)
        
        # Encoder
        x = self.encoder(x)

        # Select last time step and project
        x = x[:, -1, :]  # Shape: [batch_size, d_model]
        x = self.fc_out(x)  # Shape: [batch_size, output_dim]
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# 1. Basis Functions (Trend, Seasonality, Generic)
###############################################################################
class TrendBasis(nn.Module):
    """
    Creates polynomial basis of degree (theta_size - 1).
    """
    def __init__(self, degree, length):
        super(TrendBasis, self).__init__()
        self.degree = degree
        self.length = length
        # 'time' from 0 to 1
        self.register_buffer('time', torch.linspace(0, 1, steps=length))  # shape: (length,)

    def forward(self, theta):
        """
        theta shape: [batch_size, degree+1]
        returns backcast shape: [batch_size, length]
        """
        batch_size = theta.size(0)
        # Replicate time for batch
        time = self.time.unsqueeze(0).repeat(batch_size, 1)  # [B, length]
        # Polynomial basis => shape [B, length, degree+1]
        basis = torch.stack([time ** i for i in range(self.degree + 1)], dim=2)
        # Multiply by theta and sum over the degree dimension
        backcast = (basis * theta.unsqueeze(1)).sum(dim=2)
        return backcast


class SeasonalityBasis(nn.Module):
    """
    Creates sine/cosine basis for a specified number of harmonics.
    """
    def __init__(self, harmonics, length):
        super(SeasonalityBasis, self).__init__()
        self.harmonics = harmonics
        self.length = length
        self.register_buffer('time', torch.linspace(0, 2 * torch.pi, steps=length))

    def forward(self, theta):
        """
        theta shape: [batch_size, 2*harmonics]
        returns backcast: [batch_size, length]
        """
        batch_size = theta.size(0)
        time = self.time.unsqueeze(0).repeat(batch_size, 1)  # [B, length]
        # Build sine and cosine basis
        sines = [torch.sin((i + 1) * time) for i in range(self.harmonics)]
        cosines = [torch.cos((i + 1) * time) for i in range(self.harmonics)]
        # Stack them => shape [B, length, 2*harmonics]
        basis = torch.stack(sines + cosines, dim=2)
        # Multiply by theta and sum over the harmonic dimension
        backcast = (basis * theta.unsqueeze(1)).sum(dim=2)
        return backcast


class GenericBasis(nn.Module):
    """
    Generic basis simply interprets 'theta' as the output directly.
    For 'generic' block, we reshape it back to (B, backcast_length, num_features).
    """
    def __init__(self, size):
        super(GenericBasis, self).__init__()
        self.size = size

    def forward(self, theta):
        """
        theta shape: [batch_size, size]
        returns same shape => used for backcast/forecast in generic block
        """
        return theta  # identity

###############################################################################
# 2. NBeatsBlock
###############################################################################
class NBeatsBlock(nn.Module):
    """
    A single NBeats block:
      - block_type: 'generic', 'trend', 'seasonality'
      - if 'generic': backcast/forecast are direct expansions
      - if 'trend'/'seasonality': we use TrendBasis or SeasonalityBasis
    """
    def __init__(self, 
                 input_size,          # e.g. 12 => sequence length
                 hidden_size,         # e.g. 128
                 theta_size,          # e.g. for 'trend', = degree+1; for 'seasonality', = 2*harmonics
                 n_layers, 
                 block_type='generic',
                 backcast_length=None,
                 forecast_length=None,
                 num_features=1,
                 dropout=0.3):
        super(NBeatsBlock, self).__init__()
        self.block_type = block_type
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.num_features = num_features

        # Calculate input dimension based on input_size and num_features
        input_dim = input_size * num_features  

        # Build fully connected layers with ReLU activation, Dropout, and LayerNorm
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(hidden_size))
            input_dim = hidden_size  # Update input_dim for next layer
        self.fc = nn.Sequential(*layers)

        if block_type == 'generic':
            # For 'generic', backcast and forecast are directly inferred from theta
            self.backcast_theta_size = backcast_length * num_features
            self.forecast_theta_size = forecast_length * num_features
            total_theta_size = self.backcast_theta_size + self.forecast_theta_size
            self.theta_layer = nn.Linear(hidden_size, total_theta_size)

            self.backcast_basis = GenericBasis(size=self.backcast_theta_size)
            self.forecast_basis = GenericBasis(size=self.forecast_theta_size)
        else:
            # For 'trend' or 'seasonality', use respective basis functions
            self.theta_layer = nn.Linear(hidden_size, theta_size)
            if block_type == 'trend':
                # Degree = theta_size - 1
                self.backcast_basis = TrendBasis(degree=theta_size - 1, length=backcast_length)
                self.forecast_basis = TrendBasis(degree=theta_size - 1, length=forecast_length)
            elif block_type == 'seasonality':
                # harmonics = theta_size // 2
                self.backcast_basis = SeasonalityBasis(harmonics=theta_size // 2, length=backcast_length)
                self.forecast_basis = SeasonalityBasis(harmonics=theta_size // 2, length=forecast_length)

    def forward(self, x):
        """
        x shape: (B, backcast_length, num_features)
        => flatten => (B, backcast_length * num_features)
        => pass through fc => produce theta => decode => (backcast, forecast)
        """
        batch_size = x.size(0)
        # Flatten the input: (B, backcast_length, num_features) => (B, backcast_length * num_features)
        x = x.view(batch_size, -1)
        # Pass through fully connected layers
        x_fc = self.fc(x)  # shape: (B, hidden_size)
        # Produce theta parameters
        theta = self.theta_layer(x_fc)  # shape depends on block_type

        if self.block_type == 'generic':
            # Split theta into backcast and forecast
            theta_backcast = theta[:, :self.backcast_theta_size]
            theta_forecast = theta[:, self.backcast_theta_size:]

            # Generate backcast and forecast using GenericBasis
            backcast = self.backcast_basis(theta_backcast).view(
                batch_size, self.backcast_length, self.num_features
            )
            forecast = self.forecast_basis(theta_forecast).view(
                batch_size, self.forecast_length, self.num_features
            )
        else:
            # Generate backcast and forecast using TrendBasis or SeasonalityBasis
            backcast_1d = self.backcast_basis(theta)  # shape: (B, backcast_length)
            forecast_1d = self.forecast_basis(theta)  # shape: (B, forecast_length)

            # Expand to match num_features: (B, backcast_length, 1) -> (B, backcast_length, num_features)
            backcast = backcast_1d.unsqueeze(-1).expand(-1, -1, self.num_features)
            forecast = forecast_1d.unsqueeze(-1).expand(-1, -1, self.num_features)

        return backcast, forecast

###############################################################################
# 3. NBeats Model
###############################################################################
class NBeats(nn.Module):
    """
    NBeats model with a stack of NBeatsBlock.
    stack_types = ['trend', 'seasonality', 'generic'] or any repeated pattern
    """
    def __init__(
        self,
        input_size,           # backcast_length, e.g., 12
        hidden_size,          # e.g., 128
        n_blocks,             # number of blocks, e.g., 2
        n_layers,             # layers per block, e.g., 2
        theta_size,           # dict for 'trend', 'seasonality'
        stack_types,          # e.g., ['trend','seasonality','generic']
        backcast_length,      # e.g., 12
        forecast_length,      # e.g., 12
        num_features=1,       # e.g., 15
        device='cuda',
        dropout=0.3
    ):
        super(NBeats, self).__init__()
        self.device = device
        self.blocks = nn.ModuleList()

        for block_id in range(n_blocks):
            block_type = stack_types[block_id % len(stack_types)]

            if block_type == 'generic':
                block = NBeatsBlock(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    theta_size=None,  # 'generic' doesn't use explicit theta_size
                    n_layers=n_layers,
                    block_type='generic',
                    backcast_length=backcast_length,
                    forecast_length=forecast_length,
                    num_features=num_features,
                    dropout=dropout
                )
            else:
                # 'trend' or 'seasonality'
                # e.g., theta_size['trend'] = degree+1, theta_size['seasonality'] = 2*harmonics
                block = NBeatsBlock(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    theta_size=theta_size[block_type],
                    n_layers=n_layers,
                    block_type=block_type,
                    backcast_length=backcast_length,
                    forecast_length=forecast_length,
                    num_features=num_features,
                    dropout=dropout
                )
            self.blocks.append(block)

        self.to(device)

    def forward(self, x):
        """
        x shape => (B, backcast_length, num_features)
        Successively subtract each block's backcast => accumulate forecast
        Final forecast => (B, forecast_length, num_features)
        """
        x = x.to(self.device)
        residual = x
        forecast_sum = 0.0
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            # Update residual by removing the backcast
            residual = residual - backcast
            # Accumulate forecast
            forecast_sum = forecast_sum + block_forecast
        # Final forecast
        return forecast_sum

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(output_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        self.project_residual = nn.Linear(input_size, output_size) if input_size != output_size else None
            
    def forward(self, x):
        residual = x  # [batch_size, *, input_size]
        x = self.fc1(x)  # [batch_size, *, hidden_size]
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch_size, *, output_size]
        if self.project_residual is not None:
            residual = self.project_residual(residual)  # [batch_size, *, output_size]
        gate = torch.sigmoid(self.gate(x))  # [batch_size, *, output_size]
        x = gate * x + (1 - gate) * residual  # [batch_size, *, output_size]
        x = self.layer_norm(x)
        return x

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, num_inputs, hidden_size, dropout=0.1):
        super(VariableSelectionNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_inputs = num_inputs
        self.fc = nn.ModuleList([nn.Linear(input_dim, hidden_size) for _ in range(num_inputs)])
        self.gate = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_inputs)])
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, num_vars, input_dim]
        Returns:
            selected: Tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, num_vars, input_dim = x.size()
        x = x.view(batch_size * seq_len, num_vars, input_dim)
        
        # Compute gates and apply softmax
        weights = torch.cat([self.gate[i](x[:, i, :]).unsqueeze(1) for i in range(self.num_inputs)], dim=1)  # [batch_size * seq_len, num_vars, 1]
        weights = self.softmax(weights)  # [batch_size * seq_len, num_vars, 1]
        
        # Compute embeddings
        embeddings = torch.cat([self.fc[i](x[:, i, :]).unsqueeze(1) for i in range(self.num_inputs)], dim=1)  # [batch_size * seq_len, num_vars, hidden_size]
        
        # Weighted sum
        selected = torch.sum(weights * embeddings, dim=1)  # [batch_size * seq_len, hidden_size]
        selected = self.dropout(selected)
        selected = selected.view(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        return selected

class Time2Vec(nn.Module):
    def __init__(self, embed_size):
        super(Time2Vec, self).__init__()
        self.embed_size = embed_size
        self.wb = nn.Parameter(torch.randn(1))
        self.bb = nn.Parameter(torch.randn(1))
        self.wa = nn.Parameter(torch.randn(1, embed_size - 1))
        self.ba = nn.Parameter(torch.randn(1, embed_size - 1))
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, 1]
        Returns:
            embeddings: Tensor of shape [batch_size, seq_len, embed_size]
        """
        v1 = self.wb * x + self.bb  # [batch_size, seq_len, 1]
        vn = torch.sin(torch.matmul(x, self.wa) + self.ba)  # [batch_size, seq_len, embed_size - 1]
        return torch.cat([v1, vn], dim=-1)  # [batch_size, seq_len, embed_size]

class TFTBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super(TFTBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.grn1 = GatedResidualNetwork(d_model, ff_dim, d_model, dropout)
        
        self.grn2 = GatedResidualNetwork(d_model, ff_dim, d_model, dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self-Attention
        attn_output, _ = self.self_attn(x, x, x)  # [batch_size, seq_len, d_model]
        x = self.layer_norm1(x + attn_output)
        x = self.grn1(x)
        
        # Feed Forward
        x = self.layer_norm2(x + self.grn2(x))
        
        return x  # [batch_size, seq_len, d_model]

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_sizes, output_size, hidden_size, n_heads, ff_dim, dropout,
                 seq_len, num_static_vars, num_known_vars, num_observed_vars, device='cuda'):
        """
        A Temporal Fusion Transformer (TFT) example.
        
        This version’s forward method accepts either:
          - A dictionary with keys: 'static', 'known', 'observed', 'time_feat'
          - OR a tensor of shape [B, seq_len, total_features], where
            total_features = num_known_vars + num_observed_vars + time_feat_dim.
            (In that case, dummy static features (zeros) are created.)
        
        Args:
            input_sizes (dict): Dictionary with input sizes, e.g., 
                {'static': 32, 'known': 64, 'observed': 64, 'time_feat': 6}.
            output_size (int): Output dimension.
            hidden_size (int): Hidden dimension for embeddings and transformer.
            n_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward network dimension inside the transformer.
            dropout (float): Dropout rate.
            seq_len (int): Sequence length.
            num_static_vars (int): Number of static variables.
            num_known_vars (int): Number of known (time-varying) variables.
            num_observed_vars (int): Number of observed (time-varying) variables.
            device (str): 'cuda' or 'cpu'.
        """
        super(TemporalFusionTransformer, self).__init__()
        self.device = device
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_static_vars = num_static_vars
        self.num_known_vars = num_known_vars
        self.num_observed_vars = num_observed_vars
        
        # For time features, use the provided dimension (here, 6).
        self.time_feat_dim = input_sizes.get('time_feat', 1)
        
        # Embedding layers for each input type.
        self.static_embedding = nn.Linear(num_static_vars, hidden_size)
        self.known_embedding = nn.Linear(num_known_vars, hidden_size)
        self.observed_embedding = nn.Linear(num_observed_vars, hidden_size)
        self.time_embedding = nn.Linear(self.time_feat_dim, hidden_size)
        
        # Transformer encoder block.
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads,
                                                   dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final output layer.
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.to(device)
    
    def forward(self, inputs):
        """
        Forward method accepts either a dictionary or a tensor.
        
        If a dictionary:
          - 'static':    [B, num_static_vars]
          - 'known':     [B, seq_len, num_known_vars]
          - 'observed':  [B, seq_len, num_observed_vars]
          - 'time_feat': [B, seq_len, time_feat_dim]
          
        If a tensor:
          - Assume shape [B, seq_len, total_features]
          - total_features must equal: num_known_vars + num_observed_vars + time_feat_dim.
          - In that case, dummy static features (zeros) of shape [B, num_static_vars] are created.
        """
        # If inputs is a dictionary, unpack it.
        if isinstance(inputs, dict):
            try:
                x_static   = inputs['static'].to(self.device)      # [B, num_static_vars]
                x_known    = inputs['known'].to(self.device)         # [B, seq_len, num_known_vars]
                x_observed = inputs['observed'].to(self.device)       # [B, seq_len, num_observed_vars]
                time_feat  = inputs['time_feat'].to(self.device)      # [B, seq_len, time_feat_dim]
            except KeyError as e:
                raise ValueError(f"Missing key in inputs dictionary: {e}")
        else:
            # Assume inputs is a tensor of shape [B, seq_len, total_features]
            B, L, total_features = inputs.shape
            expected_total = self.num_known_vars + self.num_observed_vars + self.time_feat_dim
            if total_features != expected_total:
                raise ValueError(f"Expected tensor with last dimension {expected_total}, got {total_features}")
            # Split along the feature dimension.
            x_known = inputs[:, :, :self.num_known_vars]  # first part
            x_observed = inputs[:, :, self.num_known_vars:self.num_known_vars + self.num_observed_vars]
            time_feat = inputs[:, :, self.num_known_vars + self.num_observed_vars:]
            # Create dummy static features (zeros)
            x_static = torch.zeros(B, self.num_static_vars, device=self.device)
        
        # (Optional) If time_feat is empty, replace with zeros.
        if time_feat.size(-1) == 0:
            B = time_feat.size(0)
            time_feat = torch.zeros(B, self.seq_len, self.time_feat_dim, device=self.device)
        
        # Embedding layers.
        static_emb   = self.static_embedding(x_static)       # [B, hidden_size]
        known_emb    = self.known_embedding(x_known)           # [B, seq_len, hidden_size]
        observed_emb = self.observed_embedding(x_observed)     # [B, seq_len, hidden_size]
        time_emb     = self.time_embedding(time_feat)          # [B, seq_len, hidden_size]
        
        # Combine temporal embeddings.
        temporal_emb = known_emb + observed_emb + time_emb     # [B, seq_len, hidden_size]
        
        # Incorporate static embedding at each time step.
        static_emb_expanded = static_emb.unsqueeze(1).expand(-1, self.seq_len, -1)
        transformer_input = temporal_emb + static_emb_expanded  # [B, seq_len, hidden_size]
        
        # Transformer expects shape: [seq_len, B, hidden_size]
        transformer_input = transformer_input.transpose(0, 1)
        transformer_output = self.transformer_encoder(transformer_input)
        
        # Use the final time step's output.
        final_output = transformer_output[-1]  # [B, hidden_size]
        output = self.fc(final_output)         # [B, output_size]
        return output
     
class DeepAR(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, 
                 num_layers=2, dropout=0.3, device='cuda'):
        super(DeepAR, self).__init__()
        self.device = device
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for (features + target_in)
        self.lstm = nn.LSTM(
            input_dim + 1,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.ln = nn.LayerNorm(hidden_dim)  # layer norm to stabilize
        self.dropout = nn.Dropout(dropout)  # dropout after LSTM
        # A small hidden layer before final output
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # -> [mu, sigma_raw]

    def forward(self, x, target_in=None, prediction_len=None):
        batch_size, seq_len, _ = x.size()
        
        if target_in is not None:
            # TRAINING / TEACHER-FORCING
            # 1) concat features + target_in
            lstm_input = torch.cat([x, target_in.unsqueeze(-1)], dim=-1)
            
            # 2) forward LSTM
            lstm_out, _ = self.lstm(lstm_input)
            
            # 3) LN + dropout
            lstm_out = self.ln(lstm_out)
            lstm_out = self.dropout(lstm_out)

            # 4) FC layers
            hidden = F.relu(self.fc1(lstm_out))  # (batch, seq_len, hidden_dim)
            params = self.fc2(hidden)           # (batch, seq_len, 2)

            mu, sigma_raw = params[..., 0], params[..., 1]
            # Use a bigger offset, e.g. 1e-3
            sigma = F.softplus(sigma_raw) + 1e-3
            
            return mu, sigma
        
        else:
            # INFERENCE
            if prediction_len is None:
                prediction_len = self.output_dim
            
            # pass historical x with zero target to init LSTM state
            zero_target = torch.zeros(batch_size, seq_len, 1).to(self.device)
            init_input = torch.cat([x, zero_target], dim=-1)
            _, (h, c) = self.lstm(init_input)  # shape of h,c => (num_layers, batch, hidden_dim)

            mus, sigmas = [], []
            # Start from zero (or last known target) as the "seed"
            last_target = torch.zeros(batch_size, 1).to(self.device)

            for _ in range(prediction_len):
                lstm_input = torch.cat([x[:, -1:, :], last_target.unsqueeze(-1)], dim=-1)
                # LSTM cell by cell or single-step LSTM
                lstm_out, (h, c) = self.lstm(lstm_input, (h, c))

                # LN + dropout
                lstm_out = self.ln(lstm_out)
                lstm_out = self.dropout(lstm_out)

                # FC
                hidden = F.relu(self.fc1(lstm_out))
                params = self.fc2(hidden).squeeze(1)  # => (batch, 2)

                mu_t, sigma_raw_t = params[:, 0], params[:, 1]
                sigma_t = F.softplus(sigma_raw_t) + 1e-3

                mus.append(mu_t.unsqueeze(-1))
                sigmas.append(sigma_t.unsqueeze(-1))

                # update last_target with predicted mu
                last_target = mu_t.detach()

            # shape => (batch, prediction_len)
            mus = torch.cat(mus, dim=1)
            sigmas = torch.cat(sigmas, dim=1)
            
            return mus, sigmas

    def loss_fn(self, y_true, mu, sigma):
        """
        Negative log-likelihood of Gaussian: -log p(y | mu, sigma)
        """
        dist = torch.distributions.Normal(mu, sigma)
        return -dist.log_prob(y_true).mean()

    def to(self, device):
        self.device = device
        return super().to(device)
    
# --- Reformer ---
class Reformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, n_heads, ff_dim, n_layers, dropout=0.1, device='cuda'):
        super(Reformer, self).__init__()
        self.device = device  # Add device attribute
        self.embedding = nn.Linear(input_dim, d_model)  # Embed input to d_model dimension
        self.reformer_blocks = nn.ModuleList([
            ReformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, output_dim)  # Output layer

    def forward(self, x):
        # Ensure data is on the correct device
        x = x.to(self.device)
        # Embed input sequence
        x = self.embedding(x)          # Shape: [batch_size, seq_len, d_model]
        x = x.permute(1, 0, 2)         # Prepare shape for attention: [seq_len, batch_size, d_model]
        # Pass through each Reformer block
        for block in self.reformer_blocks:
            x = block(x)
        x = x.permute(1, 0, 2)         # Revert shape: [batch_size, seq_len, d_model]
        # Select the output from the last time step
        x = x[:, -1, :]                # Shape: [batch_size, d_model]
        # Final output layer to match output_dim
        x = self.fc_out(x)             # Shape: [batch_size, output_dim]
        return x

class ReformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super(ReformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class TideNet(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dim=128, 
        output_dim=1, 
        num_lstm_layers=2,
        num_attention_heads=4,
        dropout=0.3,
        device='cuda'
    ):
        """
        A more complex TideNet architecture with:
          - Multi-scale dilated convolutions
          - Gating mechanism for feature fusion
          - Stacked LSTM
          - Multi-head attention
          - Residual connections
          - Dropout & layer normalization
          
        Manual padding is used so all conv branches produce the same temporal dimension.
        This is necessary for older PyTorch versions that do not support padding="same".
        """
        super(TideNet, self).__init__()
        self.device = device

        # ---------------------------------------------------------------------
        # 1) Multi-Scale Dilated Convolution Branches
        #    We use kernel_size=3 with different dilations (1, 2, 4).
        #    Padding is manually computed:
        #      padding = ((kernel_size - 1) * dilation) // 2
        # ---------------------------------------------------------------------
        
        # For kernel_size=3, dilation=1 => padding=1
        self.branch1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            dilation=1,
            padding=1  # ensures 'same' length for dilation=1
        )
        # For kernel_size=3, dilation=2 => padding=2
        self.branch2 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            dilation=2,
            padding=2
        )
        # For kernel_size=3, dilation=4 => padding=4
        self.branch3 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            dilation=4,
            padding=4
        )

        # Gating mechanism to combine these branches (instead of naive concat).
        # We'll flatten across the channel dimension, pass through a small FC to
        # get a gating factor, then multiply back.
        self.gate_fc = nn.Linear(hidden_dim * 3, hidden_dim * 3) 
        
        # 1x1 convolution to fuse the gated concatenation into hidden_dim
        self.conv_fuse = nn.Conv1d(
            in_channels=hidden_dim * 3,
            out_channels=hidden_dim,
            kernel_size=1
        )

        self.conv_ln = nn.LayerNorm(hidden_dim)  # LN after multi-scale fusion
        self.dropout = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # 2) Stacked LSTM
        # ---------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout
        )
        self.lstm_ln = nn.LayerNorm(hidden_dim)

        # ---------------------------------------------------------------------
        # 3) Multi-Head Attention
        #   - batch_first=False in older PyTorch versions.
        #   - We'll transpose to shape (seq_len, batch, hidden_dim).
        # ---------------------------------------------------------------------
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=False  # older PyTorch default
        )
        self.attn_ln = nn.LayerNorm(hidden_dim)
        
        # ---------------------------------------------------------------------
        # 4) Residual + Final FC
        # ---------------------------------------------------------------------
        self.final_fc = nn.Linear(hidden_dim, output_dim)
        
        # Move the model to the specified device
        self.to(device)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        Returns:
            predictions: Tensor of shape (batch, output_dim)
        """
        # Ensure input on the correct device
        x = x.to(self.device)

        # ---------------------------------------------------------------------
        # Multi-Scale Dilated Convolution
        #  1) Reshape to (batch, channels=in_dim, seq_len)
        #  2) Branch outputs all => (batch, hidden_dim, seq_len)
        #  3) Gating + Fusion => (batch, seq_len, hidden_dim)
        # ---------------------------------------------------------------------
        x_conv = x.transpose(1, 2)  # => (batch, input_dim, seq_len)

        b1 = F.relu(self.branch1(x_conv))  # => (batch, hidden_dim, seq_len)
        b2 = F.relu(self.branch2(x_conv))  # => (batch, hidden_dim, seq_len)
        b3 = F.relu(self.branch3(x_conv))  # => (batch, hidden_dim, seq_len)

        # Concatenate along the channel dimension: (batch, hidden_dim*3, seq_len)
        cat = torch.cat([b1, b2, b3], dim=1)

        # Gating:
        #  Flatten => (batch, seq_len, hidden_dim*3), pass through gate_fc => gating factors
        cat_trans = cat.transpose(1, 2)  # => (batch, seq_len, hidden_dim*3)
        gates = torch.sigmoid(self.gate_fc(cat_trans))  # => (batch, seq_len, hidden_dim*3)
        gated_cat = cat_trans * gates  # => (batch, seq_len, hidden_dim*3)

        # Fuse from 3*hidden_dim => hidden_dim via 1x1 conv
        fused = gated_cat.transpose(1, 2)       # => (batch, 3*hidden_dim, seq_len)
        fused = self.conv_fuse(fused)          # => (batch, hidden_dim, seq_len)
        fused = fused.transpose(1, 2)          # => (batch, seq_len, hidden_dim)

        fused = self.conv_ln(fused)            # LN after conv fusion
        fused = self.dropout(fused)

        # ---------------------------------------------------------------------
        # LSTM => (batch, seq_len, hidden_dim)
        # ---------------------------------------------------------------------
        lstm_out, _ = self.lstm(fused)  # => (batch, seq_len, hidden_dim)
        lstm_out = self.lstm_ln(lstm_out)
        lstm_out = self.dropout(lstm_out)

        # ---------------------------------------------------------------------
        # Multi-Head Attention
        #   - Must transpose to (seq_len, batch, hidden_dim)
        #   - Then revert back to (batch, seq_len, hidden_dim)
        # ---------------------------------------------------------------------
        lstm_out_t = lstm_out.transpose(0, 1)  # => (seq_len, batch, hidden_dim)
        attn_out, _ = self.attn(lstm_out_t, lstm_out_t, lstm_out_t)
        # Residual connection
        attn_out = attn_out + lstm_out_t
        # Transpose back to (batch, seq_len, hidden_dim)
        attn_out = attn_out.transpose(0, 1)
        attn_out = self.attn_ln(attn_out)
        attn_out = self.dropout(attn_out)

        # For final prediction, we can simply take the last time step
        # or do an average / other pooling as needed.
        out = attn_out[:, -1, :]  # => (batch, hidden_dim)

        # ---------------------------------------------------------------------
        # Final FC => (batch, output_dim)
        # ---------------------------------------------------------------------
        predictions = self.final_fc(out)
        return predictions

# --- LogSparse Transformer ---
class LogSparseTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, n_heads, ff_dim, n_layers, dropout=0.1, device='cuda'):
        super(LogSparseTransformer, self).__init__()
        self.device = device  # Adding device attribute for CPU/GPU support
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_blocks = nn.ModuleList([
            LogSparseTransformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        x = self.embedding(x)           # Shape: [batch_size, seq_len, d_model]
        for block in self.transformer_blocks:
            x = block(x)                # Shape remains [batch_size, seq_len, d_model]
        # Select the output from the last time step
        x = x[:, -1, :]                 # Shape: [batch_size, d_model]
        # Final output layer to match output_dim
        x = self.fc_out(x)              # Shape: [batch_size, output_dim]
        return x

class LogSparseTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super(LogSparseTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention mechanism
        attn_output, _ = self.self_attn(x, x, x)
        # Layer normalization and residual connection
        x = self.layer_norm1(x + self.dropout(attn_output))
        # Feed-forward layer and residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x
    
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint  # Ensure you have installed torchdiffeq (pip install torchdiffeq)

class NeuralODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        A neural network representing the derivative function of the ODE.
        Increased capacity by adding an extra hidden layer.
        """
        super(NeuralODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, t, x):
        return self.net(x)

class NeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, device='cuda'):
        """
        NeuralODE model that solves an ODE defined by NeuralODEFunc and then maps
        the final state to the target output using a fully connected layer.
        
        Args:
            input_dim (int): Dimensionality of the input (and ODE state).
            hidden_dim (int): Hidden dimension for the ODE function.
            output_dim (int): Dimensionality of the final prediction.
            device (str): 'cuda' or 'cpu'.
        """
        super(NeuralODE, self).__init__()
        self.func = NeuralODEFunc(input_dim, hidden_dim)
        # Use a learnable mapping to produce the final output.
        self.fc = nn.Linear(input_dim, output_dim)
        self.device = device
        self.to(device)

    def forward(self, x, t_span=None):
        """
        Solves the ODE defined by self.func starting from initial condition x, then
        applies a fully connected layer to produce the final output.
        
        Args:
            x (torch.Tensor): Initial condition tensor of shape [batch_size, input_dim].
            t_span (torch.Tensor, optional): A 1D tensor of time points. If not provided,
                                               a default time span with 10 points between 0 and 1 is used.
        
        Returns:
            torch.Tensor: The final prediction of shape [batch_size, output_dim].
        """
        x = x.to(self.device)
        # If no time span is provided, use a default with more time steps for better accuracy.
        if t_span is None:
            t_span = torch.linspace(0.0, 1.0, steps=10, device=self.device)
        else:
            t_span = t_span.to(self.device)
        
        # Solve the ODE; out has shape [T, batch_size, input_dim].
        out = odeint(self.func, x, t_span, rtol=1e-4, atol=1e-5, method='dopri5')
        final = out[-1]  # Select the final time point: shape [batch_size, input_dim].
        
        # Map the final state to the desired output dimension.
        final = self.fc(final)  # Shape: [batch_size, output_dim]
        return final

# --- WaveNet ---
class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, residual_channels, dilation):
        super(WaveNetBlock, self).__init__()
        self.dilated_conv = nn.Conv1d(
            in_channels,
            residual_channels,
            kernel_size=2,
            dilation=dilation,
            padding=dilation  # This leads to output length = L + dilation
        )
        self.relu = nn.ReLU()
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)

    def forward(self, x):
        # x has shape [B, residual_channels, L]
        out = self.dilated_conv(x)  # Output shape: [B, residual_channels, L + dilation]
        out = self.relu(out)
        
        # Crop the output to match the input's length:
        L = x.size(2)
        out = out[:, :, :L]  # Now out has shape [B, residual_channels, L]
        
        residual = self.residual_conv(out)  # [B, residual_channels, L]
        # skip is computed but not used in the residual connection here;
        # you might use it in a more complete WaveNet architecture.
        skip = self.skip_conv(out)  # [B, residual_channels, L]
        
        # Add the residual connection: shapes match now.
        return residual + x

class WaveNet(nn.Module):
    def __init__(self, input_dim, residual_channels, output_dim, dilation_depth, device='cuda'):
        super(WaveNet, self).__init__()
        self.device = device
        # The input layer transforms input_dim to residual_channels.
        self.input_layer = nn.Conv1d(input_dim, residual_channels, kernel_size=1)
        # Create a stack of WaveNetBlocks with increasing dilation.
        self.blocks = nn.ModuleList([
            WaveNetBlock(residual_channels, residual_channels, dilation=2 ** i) 
            for i in range(dilation_depth)
        ])
        # The output layer transforms residual_channels to output_dim.
        self.output_layer = nn.Conv1d(residual_channels, output_dim, kernel_size=1)

    def forward(self, x):
        # Ensure the input is on the correct device.
        x = x.to(self.device)
        # If input is [B, seq_len], add a channel dimension to become [B, 1, seq_len].
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # If input is [B, seq_len, input_dim], transpose to [B, input_dim, seq_len].
        elif x.dim() == 3 and x.size(1) != self.input_layer.in_channels:
            x = x.permute(0, 2, 1)
        
        # Apply the input layer.
        x = self.input_layer(x)
        # Pass through each WaveNetBlock.
        for block in self.blocks:
            x = block(x)
        # Apply the output layer.
        x = self.output_layer(x)
        # Select the last time step from the output.
        x = x[:, :, -1]  # Output shape: [B, output_dim]
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvNetModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, kernel_size=2, dropout=0.2, device='cuda'):
        super(TemporalConvNetModel, self).__init__()
        self.device = device
        self.input_dim = input_dim  # expected number of input channels
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            layers += [
                nn.Conv1d(in_channels,
                          num_channels[i],
                          kernel_size,
                          stride=1,
                          padding=(kernel_size - 1) * dilation_size,
                          dilation=dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_dim)
        self.to(device)
    
    def forward(self, x):
        # x should be either [B, input_dim, seq_len] or [B, seq_len, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            if x.size(1) == self.input_dim:
                # Already [B, channels, seq_len]
                pass
            elif x.size(2) == self.input_dim:
                # Permute [B, seq_len, channels] to [B, channels, seq_len]
                x = x.permute(0, 2, 1)
            else:
                raise ValueError(f"Input tensor shape {x.shape} does not contain the expected channel dimension {self.input_dim}.")
        else:
            raise ValueError("Input tensor must be 2D or 3D.")
        
        out = self.network(x)
        out = self.fc(out[:, :, -1])
        return out

###############################################################################
# 2. CREATE SEQUENCES FUNCTION (REUSED)
###############################################################################
def create_sequences(features: np.ndarray, targets: np.ndarray, sequence_length: int):
    """
    Standard sequence creation for time-series approach:
      features[i:i+seq_len], target[i+seq_len-1]
    """
    seqs = []
    labels = []
    for i in range(len(features) - sequence_length):
        seqs.append(features[i : i + sequence_length])
        labels.append(targets[i + sequence_length - 1])
    return np.array(seqs), np.array(labels)

###############################################################################
# 3. BUILD DATALOADERS (REUSED)
###############################################################################
def build_dataloaders(dataset: TensorDataset, seed: int, batch_size: int):
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

###############################################################################
# 4. PREPARE DATASET: GLOBAL
###############################################################################
def prepare_global_dataset(df: pd.DataFrame, sequence_length: int):
    """
    Prepare dataset for the global analysis:
     - Features = [Year] + [All region columns except 'Year' & 'Total Yearly Capacity']
     - Target   = 'Total Yearly Capacity'
    Returns:
      TensorDataset of sequential samples.
    """
    # Identify region columns by excluding 'Year' and 'Total Yearly Capacity'
    exclude_cols = ['Year', 'Total Yearly Capacity']
    region_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Features: Year + region columns
    X = df[['Year'] + region_cols].values  # shape: (N, 1 + #regions)
    
    # Target: 'Total Yearly Capacity'
    y = df['Total Yearly Capacity'].values.reshape(-1, 1)
    
    # Convert to sequences
    X_seq, y_seq = create_sequences(X, y, sequence_length)
    
    # Build dataset
    dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32),
                            torch.tensor(y_seq, dtype=torch.float32))
    return dataset

###############################################################################
# 5. PREPARE DATASET: REGIONAL (for each region)
###############################################################################
def prepare_regional_dataset(df: pd.DataFrame, region_name: str, sequence_length: int):
    """
    Prepare dataset for a specific region analysis:
      - Features: [Year] + [Total Yearly Capacity]
      - Target: region_name's capacity
    """
    if region_name not in df.columns:
        raise ValueError(f"Region '{region_name}' not found in DataFrame.")
    
    # Features: Year + Total Yearly Capacity
    X = df[['Year', 'Total Yearly Capacity']].values  # shape: (N, 2)
    
    # Target: region's capacity
    y = df[region_name].values.reshape(-1, 1)  # shape: (N, 1)
    
    # Convert to sequences
    X_seq, y_seq = create_sequences(X, y, sequence_length)
    
    dataset = TensorDataset(torch.tensor(X_seq, dtype=torch.float32),
                            torch.tensor(y_seq, dtype=torch.float32))
    return dataset

###############################################################################
# 6. PLACEHOLDER TRAIN & EVAL (You already have actual functions)
###############################################################################

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=200,
    patience=20,
    grad_clip_value=1.0,
    grad_accum_steps=1,
    warmup_steps=20,
):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    current_step = 0

    for epoch in range(num_epochs):
        # -----------------
        # 1. Training Phase
        # -----------------
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            # Forward pass depends on model type
            outputs = model(inputs)

            if isinstance(model, Transformer):
                # Typical shape: (B, seq_len, output_dim) or (B, output_dim)
                loss = criterion(outputs, labels)

            elif isinstance(model, NBeats):
                # We expect outputs to have shape (B, forecast_length, num_features)
                # Check if outputs and labels shape match
                if outputs.shape != labels.shape:
                    # Attempt minimal fix or raise a warning
                    # For example, if your label is 2D and output is 3D with last dim=1
                    if (outputs.dim() == 3 and outputs.size(-1) == 1 and labels.dim() == 2):
                        # shape fix for single-feature
                        # e.g. outputs => (B, forecast_len, 1), labels => (B, forecast_len)
                        outputs = outputs.squeeze(-1)
                    else:
                        print(f"[Warning] NBeats output shape {outputs.shape} != label shape {labels.shape}")
                        # Either raise an error or handle shape logic as needed

                loss = criterion(outputs, labels)

            elif isinstance(model, DeepAR):
                # DeepAR outputs tuple (mu, sigma)
                mu, sigma = outputs
                loss = model.loss_fn(labels, mu, sigma)

            else:
                # Generic handling for other models
                loss = criterion(outputs, labels)

            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * inputs.size(0)
            current_step += 1

            # LR warmup
            if current_step < warmup_steps:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * (current_step / warmup_steps)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # -------------------
        # 2. Validation Phase
        # -------------------
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(model.device)
                val_labels = val_labels.to(model.device)

                val_outputs = model(val_inputs)

                if isinstance(model, Transformer):
                    val_loss = criterion(val_outputs, val_labels)

                elif isinstance(model, NBeats):
                    # Again, ensure shape alignment
                    if val_outputs.shape != val_labels.shape:
                        # Attempt minimal fix or handle accordingly
                        if (val_outputs.dim() == 3 and val_outputs.size(-1) == 1 and val_labels.dim() == 2):
                            val_outputs = val_outputs.squeeze(-1)
                        else:
                            print(f"[Warning] NBeats val output {val_outputs.shape} != label {val_labels.shape}")

                    val_loss = criterion(val_outputs, val_labels)

                elif isinstance(model, DeepAR):
                    mu_val, sigma_val = val_outputs
                    val_loss = model.loss_fn(val_labels, mu_val, sigma_val)

                else:
                    val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item() * val_inputs.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Update scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)
        else:
            scheduler.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Training Loss: {epoch_train_loss:.6f}, '
                  f'Validation Loss: {epoch_val_loss:.6f}')

        # Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'best_model_{model.__class__.__name__}.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}. No improvement in validation loss for {patience} epochs.')
            break

    # Load best model
    model.load_state_dict(torch.load(f'best_model_{model.__class__.__name__}.pth'))

    return model, train_losses, val_losses

def regional_train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=200,
    patience=20,
    grad_clip_value=1.0,
    grad_accum_steps=1,
    warmup_steps=20,
):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    current_step = 0

    for epoch in range(num_epochs):
        # -----------------
        # 1. Training Phase
        # -----------------
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            # Forward pass depends on model type
            outputs = model(inputs)

            if isinstance(model, Transformer):
                # Typical shape: (B, seq_len, output_dim) or (B, output_dim)
                loss = criterion(outputs, labels)

            elif isinstance(model, NBeats):
                # We expect outputs to have shape (B, forecast_length, num_features)
                # Check if outputs and labels shape match
                if outputs.shape != labels.shape:
                    # Attempt minimal fix or raise a warning
                    # For example, if your label is 2D and output is 3D with last dim=1
                    if (outputs.dim() == 3 and outputs.size(-1) == 1 and labels.dim() == 2):
                        # shape fix for single-feature
                        # e.g. outputs => (B, forecast_len, 1), labels => (B, forecast_len)
                        outputs = outputs.squeeze(-1)
                    else:
                        print(f"[Warning] NBeats output shape {outputs.shape} != label shape {labels.shape}")
                        # Either raise an error or handle shape logic as needed

                loss = criterion(outputs, labels)

            elif isinstance(model, DeepAR):
                # DeepAR outputs tuple (mu, sigma)
                mu, sigma = outputs
                loss = model.loss_fn(labels, mu, sigma)

            else:
                # Generic handling for other models
                loss = criterion(outputs, labels)

            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * inputs.size(0)
            current_step += 1

            # LR warmup
            if current_step < warmup_steps:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * (current_step / warmup_steps)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # -------------------
        # 2. Validation Phase
        # -------------------
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(model.device)
                val_labels = val_labels.to(model.device)

                val_outputs = model(val_inputs)

                if isinstance(model, Transformer):
                    val_loss = criterion(val_outputs, val_labels)

                elif isinstance(model, NBeats):
                    # Again, ensure shape alignment
                    if val_outputs.shape != val_labels.shape:
                        # Attempt minimal fix or handle accordingly
                        if (val_outputs.dim() == 3 and val_outputs.size(-1) == 1 and val_labels.dim() == 2):
                            val_outputs = val_outputs.squeeze(-1)
                        else:
                            print(f"[Warning] NBeats val output {val_outputs.shape} != label {val_labels.shape}")

                    val_loss = criterion(val_outputs, val_labels)

                elif isinstance(model, DeepAR):
                    mu_val, sigma_val = val_outputs
                    val_loss = model.loss_fn(val_labels, mu_val, sigma_val)

                else:
                    val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item() * val_inputs.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Update scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)
        else:
            scheduler.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Training Loss: {epoch_train_loss:.6f}, '
                  f'Validation Loss: {epoch_val_loss:.6f}')

        # Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'best_model_{model.__class__.__name__}.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}. No improvement in validation loss for {patience} epochs.')
            break

    # Load best model
    model.load_state_dict(torch.load(f'best_model_{model.__class__.__name__}.pth'))

    return model, train_losses, val_losses

def evaluate_model(model, test_loader, output_dim, target_scaler=None):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            # Forward pass
            if isinstance(model, DeepAR):
                mu, sigma = model(inputs)
                outputs = mu  # Use the mean for evaluation
            else:
                outputs = model(inputs)

                # Conditional adjustment for NBeats
                if isinstance(model, NBeats):
                    # Aggregate outputs if labels are summary values
                    outputs = outputs.mean(dim=[1, 2]).unsqueeze(1)  # Shape: [batch_size, 1]

            # Detach and collect predictions
            predictions.append(outputs.cpu().numpy())
            actuals.append(labels.cpu().numpy())

    # Concatenate all predictions and actuals
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # Reshape or scale data if necessary
    if target_scaler is not None:
        predictions = target_scaler.inverse_transform(predictions)
        actuals = target_scaler.inverse_transform(actuals)

    # Compute metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)

    print(f"Evaluation Results - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return mse, mae, r2, rmse

def plot_all_training_curves(train_histories, val_histories, model_names):
    sns.set_style("whitegrid", {'axes.grid': True,
                               'grid.color': '.9',
                               'grid.linestyle': '--'})
    num_models = len(model_names)
    
    # Three subplots per row
    cols = 3
    rows = (num_models + cols - 1) // cols
    
    # Create figure with extra space at bottom for legend
    fig = plt.figure(figsize=(15, 4.5*rows + 1))  # Added height for legend
    
    # Create GridSpec with better spacing, leaving room for legend
    gs = fig.add_gridspec(rows, cols, hspace=0.4, wspace=0.3)
    axes = []
    for i in range(rows):
        for j in range(cols):
            if i * cols + j < num_models:
                axes.append(fig.add_subplot(gs[i, j]))
    
    # Professional color palette
    colors = {'train': '#0071BC', 'val': '#F05023'}
    
    for i, name in enumerate(model_names):
        if i < len(train_histories):
            ax = axes[i]
            
            sns.lineplot(x=np.arange(1, len(train_histories[name]) + 1),
                        y=train_histories[name],
                        label='Training Loss',
                        color=colors['train'],
                        linewidth=1.5,
                        ax=ax)
            
            sns.lineplot(x=np.arange(1, len(val_histories[name]) + 1),
                        y=val_histories[name],
                        label='Validation Loss',
                        color=colors['val'],
                        linewidth=1.5,
                        ax=ax)
            
            ax.set_title(name, 
                        fontsize=11, 
                        fontweight='bold',
                        pad=10)
            ax.set_xlabel('Epochs', fontsize=9, labelpad=7)
            ax.set_ylabel('Loss', fontsize=9, labelpad=7)
            
            sns.despine(ax=ax, top=True, right=True)
            ax.grid(True, linestyle='--', alpha=0.7, color='#E6E6E6')
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8)
            
            # Remove individual legends
            ax.get_legend().remove()
    
    # Clean up any unused subplots
    for j in range(len(model_names), len(axes)):
        axes[j].set_visible(False)
    
    # First adjust the subplots
    plt.tight_layout()
    
    # Then add the legend below all subplots
    fig.legend(['Training Loss', 'Validation Loss'],
              loc='center',
              bbox_to_anchor=(0.5, 0.02),
              ncol=2,
              fontsize=10,
              frameon=False)
    
    # Finally adjust the bottom margin for the legend
    plt.subplots_adjust(bottom=0.1, hspace=0.4)
    plt.show()

def plot_evaluation_metrics(results):
    sns.set_style("whitegrid", {'axes.grid': True,
                               'grid.color': '.9',
                               'grid.linestyle': '--'})
    metrics = ['MSE', 'MAE', 'R2', 'RMSE']
    
    # Create figure with adjusted dimensions for 2x2 layout
    fig = plt.figure(figsize=(15, 10))
    
    # Create GridSpec with better spacing
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    
    # Professional color palette
    colors = sns.color_palette("husl", len(results))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [results[model][metric] for model in results.keys()]
        
        # Enhanced bar plot
        sns.barplot(x=list(results.keys()), 
                   y=values, 
                   ax=ax, 
                   palette=colors)
        
        # Enhanced styling
        ax.set_title(f'{metric} Comparison', 
                    fontsize=12, 
                    fontweight='bold',
                    pad=15)
        ax.set_xlabel('Model', fontsize=10, labelpad=10)
        ax.set_ylabel(metric, fontsize=10, labelpad=10)
        
        # Improved tick labels
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # Add value labels on bars with enhanced styling
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.3f}', 
                       (p.get_x() + p.get_width()/2., height),
                       ha='center', 
                       va='bottom',
                       fontsize=8,
                       fontweight='regular')
        
        # Remove top and right spines
        sns.despine(ax=ax, top=True, right=True)
        
        # Enhanced grid styling
        ax.grid(True, linestyle='--', alpha=0.7, color='#E6E6E6')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actuals(models, test_loader, output_dim, target_scaler):
    sns.set_style("whitegrid", {'axes.grid': True,
                               'grid.color': '.9',
                               'grid.linestyle': '--'})
    num_models = len(models)
    
    # Three subplots per row
    cols = 3
    rows = (num_models + cols - 1) // cols
    
    # Create figure with constrained_layout
    fig = plt.figure(figsize=(15, 4.5*rows + 1), constrained_layout=True)
    
    # Create GridSpec
    gs = fig.add_gridspec(rows, cols)
    axes = []
    for i in range(rows):
        for j in range(cols):
            if i * cols + j < num_models:
                axes.append(fig.add_subplot(gs[i, j]))
    
    colors = {'actuals': '#0071BC', 'predictions': '#F05023'}
    
    for i, (name, model) in enumerate(models.items()):
        model.eval()
        
        # Initialize lists to collect predictions and actuals
        predictions_list = []
        actuals_list = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(next(model.parameters()).device)
                labels = labels.to(next(model.parameters()).device)
                
                # Forward pass
                if isinstance(model, DeepAR):
                    mu, sigma = model(inputs)
                    outputs = mu.cpu()
                else:
                    outputs = model(inputs).cpu()
                    
                    # Conditional adjustment for NBeats
                    if isinstance(model, NBeats):
                        if labels.shape != outputs.shape:
                            outputs = outputs.mean(dim=[1, 2]).unsqueeze(1)
                
                labels = labels.cpu()
                
                # Adjust outputs and labels to have matching shapes
                outputs, labels = adjust_shapes(outputs, labels)
                
                # Reshape outputs and labels to 2D arrays
                outputs_flat = outputs.numpy().reshape(-1, outputs.shape[-1])
                labels_flat = labels.numpy().reshape(-1, labels.shape[-1])
                
                # Inverse transform
                outputs_rescaled = target_scaler.inverse_transform(outputs_flat)
                labels_rescaled = target_scaler.inverse_transform(labels_flat)
                
                # Collect predictions and actuals
                predictions_list.append(outputs_rescaled)
                actuals_list.append(labels_rescaled)
        
        # Concatenate all batches
        predictions = np.concatenate(predictions_list, axis=0).ravel()
        actuals = np.concatenate(actuals_list, axis=0).ravel()
        
        # Compute correlation coefficient
        corr = np.corrcoef(predictions, actuals)[0, 1]
        
        if i < len(axes):
            ax = axes[i]
            
            sns.lineplot(x=np.arange(len(actuals)), 
                         y=actuals, 
                         label='Actuals',
                         color=colors['actuals'],
                         linewidth=1.5,
                         ax=ax)
            
            sns.lineplot(x=np.arange(len(predictions)), 
                         y=predictions, 
                         label='Predictions',
                         color=colors['predictions'],
                         linewidth=1.5,
                         ax=ax)
            
            ax.set_title(name, 
                         fontsize=11, 
                         fontweight='bold',
                         pad=10)
            ax.set_xlabel('Sample Index', fontsize=9, labelpad=7)
            ax.set_ylabel('Capacity (kt)', fontsize=9, labelpad=7)
            
            sns.despine(ax=ax, top=True, right=True)
            ax.grid(True, linestyle='--', alpha=0.7, color='#E6E6E6')
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8)
            
            # Add correlation coefficient
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                    transform=ax.transAxes,
                    fontsize=9,
                    bbox=dict(facecolor='white', 
                              alpha=0.9,
                              edgecolor='#E6E6E6',
                              boxstyle='round,pad=0.5'))
            
            # Remove individual legends
            ax.get_legend().remove()
        
    # Clean up any unused subplots
    for j in range(len(models), len(axes)):
        axes[j].set_visible(False)
    
    # Add legend using constrained layout
    fig.legend(['Actuals', 'Predictions'],
               loc='lower center',
               ncol=2,
               fontsize=10,
               frameon=False)
    
    plt.show()

def adjust_shapes(outputs, labels):
    # If outputs and labels have different shapes, adjust them
    if outputs.shape != labels.shape:
        # If outputs have extra dimensions, squeeze them
        if outputs.dim() > labels.dim():
            outputs = outputs.squeeze()
        elif labels.dim() > outputs.dim():
            labels = labels.squeeze()
        
        # If outputs and labels still have different shapes, select last time step
        if outputs.shape != labels.shape:
            if outputs.shape[1:] == labels.shape[1:]:
                # Flatten over batch dimension
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1, labels.shape[-1])
            else:
                # Select the last time step
                outputs = outputs[:, -1, :] if outputs.dim() > 2 else outputs
                labels = labels[:, -1, :] if labels.dim() > 2 else labels
    
    return outputs, labels

# Add padding to input sequence to match the expected sequence length
def pad_input_sequence(input_sequence, target_length):
    current_length = input_sequence.shape[1]
    if current_length < target_length:
        padding = torch.zeros((input_sequence.shape[0], target_length - current_length, input_sequence.shape[2])).to(input_sequence.device)
        input_sequence = torch.cat([input_sequence, padding], dim=1)
    return input_sequence

def get_base_model(m):
    """Unwraps DataParallel if needed."""
    while hasattr(m, "module"):
        m = m.module
    return m

def predict_future_total_capacity(models, df, sequence_length):
    """
    Uses a sliding-window approach to forecast future total capacity for each model.
    
    The function uses the same features as in global training:
      - Features = [Year] + [all region columns except 'Year' and 'Total Yearly Capacity']
      - Target = 'Total Yearly Capacity'
    
    The last `sequence_length` rows of df (using those features) form the initial input.
    Forecasts are generated for future years from 2025 to 2050 (using freq="YE").
    
    Only yearly total capacity forecasts are computed (no cumulative values).
    The function plots the yearly forecasts with a professional aesthetic and saves the plot and CSV.
    
    Parameters
    ----------
    models : dict
        Dictionary mapping model names to trained model instances.
    df : pd.DataFrame
        The augmented DataFrame (in final/original scale) used for training/inference.
    sequence_length : int
        Number of rows (time steps) to use as the input window.
    
    Returns
    -------
    forecasts : dict
        Dictionary mapping each model name to {"yearly": np.array}.
        (These yearly forecasts can be used later to compute cumulative totals.)
    """
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Determine features as in training:
    exclude_cols = ['Year', 'Total Yearly Capacity']
    region_cols = [c for c in df.columns if c not in exclude_cols]
    features = ['Year'] + region_cols

    # Future years: using freq "YE" so the x-axis shows proper year-end values.
    future_years = pd.date_range(start="2025", end="2051", freq="Y")
    num_future = len(future_years)
    data_to_save = pd.DataFrame({"Year": future_years.year})

    # Set up plot with enhanced aesthetics
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("deep", len(models))
    base_filename = "forecasted_capacity_professional"

    forecasts = {}
    any_success = False

    # Prepare the initial input window using the same features as in training.
    init_data = df[features].iloc[-sequence_length:].values
    # (Assuming training and inference use the same column order.)
    
    for i, (model_name, model) in enumerate(models.items()):
        model.eval()
        predictions = []
        device = next(model.parameters()).device

        # Determine if the model is an NBeats (by class name) or a univariate model.
        def get_base_model(m):
            while hasattr(m, "module"):
                m = m.module
            return m

        base_model = get_base_model(model)
        mclass = type(base_model).__name__.lower()
        is_nbeats = ("nbeats" in mclass) or ("n-beats" in mclass)
        # (For this version we do not collapse features for univariate models.)
        
        if is_nbeats:
            # For NBeats, flatten the input to a 2D tensor.
            # Ensure that sequence_length equals the model's backcast length.
            backcast_len = model.backcast_length  # e.g., 15
            if sequence_length != backcast_len:
                raise ValueError(
                    f"[{model_name}] Expected sequence_length {backcast_len} for NBeats, got {sequence_length}"
                )
            flat_input = torch.tensor(init_data, dtype=torch.float32, device=device).reshape(1, -1)
            input_sequence = flat_input
            print(f"[DEBUG] {model_name} (NBeats) initial input shape: {list(input_sequence.shape)}")
        else:
            # For other models, use 3D input: [1, sequence_length, input_dim]
            input_sequence = torch.tensor(init_data, dtype=torch.float32, device=device).unsqueeze(0)
            print(f"[DEBUG] {model_name} initial input shape: {list(input_sequence.shape)}")
        
        produced_forecasts = True
        with torch.no_grad():
            for year in future_years:
                try:
                    output = base_model(input_sequence)
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    if is_nbeats:
                        # Expect output shape: [1, forecast_length * num_features]
                        num_feats = base_model.num_features
                        # For yearly forecast, we sum the forecast block (assumes forecast_length >= 1)
                        forecast_step = output[:, -num_feats:]
                        step_sum = forecast_step.sum(dim=1).item()
                        predictions.append(step_sum)
                        # Update the flattened window by sliding: remove leftmost num_feats values.
                        left_slice = input_sequence[:, num_feats:]
                        input_sequence = torch.cat([left_slice, forecast_step], dim=1)
                    else:
                        # Expect output shape: [1, sequence_length, feature_dim]
                        if output.dim() == 3:
                            pred = output[:, -1, :].squeeze(0).cpu().numpy()  # shape: (feature_dim,)
                        elif output.dim() == 2:
                            pred = output.squeeze(0).cpu().numpy()
                        elif output.dim() == 1:
                            pred = output.cpu().numpy()
                        else:
                            raise ValueError(
                                f"[{model_name}] Unexpected output shape: {list(output.shape)}"
                            )
                        predictions.append(pred)
                        # Slide the 3D window: remove first time step, append new prediction.
                        arr = input_sequence.cpu().numpy()  # shape: [1, seq_len, feature_dim]
                        arr = arr[:, 1:, :]  # new shape: [1, seq_len-1, feature_dim]
                        n_feat = arr.shape[-1]
                        if pred.ndim == 1:
                            # Ensure pred is of length n_feat
                            if pred.shape[0] < n_feat:
                                pred = np.pad(pred, (0, n_feat - pred.shape[0]), 'constant')
                            elif pred.shape[0] > n_feat:
                                pred = pred[:n_feat]
                            pred_3d = pred.reshape(1, 1, n_feat)
                        else:
                            raise ValueError(f"[{model_name}] Unexpected prediction shape: {pred.shape}")
                        new_arr = np.concatenate([arr, pred_3d], axis=1)
                        input_sequence = torch.tensor(new_arr, dtype=torch.float32, device=device)
                except Exception as e:
                    print(f"Error forecasting with {model_name} for year {year.year}: {e}")
                    produced_forecasts = False
                    break

        if not produced_forecasts or len(predictions) == 0:
            print(f"No forecasts produced by {model_name}; skipping.")
            continue

        any_success = True
        
        # 1) Convert predictions to float array
        predictions = np.array(predictions, dtype=float)
        # 2) Take the absolute value to avoid negative capacity forecasts
        predictions = np.abs(predictions) * 10
        
        # For multivariate outputs, sum across features to obtain a scalar per year.
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            total_yearly = predictions.sum(axis=1)
        else:
            total_yearly = predictions.flatten()
        
        # (Do not compute cumulative here; only yearly forecasts are needed.)
        forecasts[model_name] = {"yearly": total_yearly}
        data_to_save[f"{model_name}_Yearly"] = total_yearly
        
        xvals = future_years.year.astype(int)
        sns.lineplot(
            x=xvals,
            y=total_yearly,
            label=f"{model_name} Yearly",
            color=palette[i],
            lw=2
        )

    if not any_success:
        print("No models produced forecasts. Exiting without saving plot/CSV.")
        return {}

    # Enhance plot aesthetics
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Capacity (kt)", fontsize=14, fontweight="bold")
    plt.title("Yearly Total Capacity Forecast", fontsize=16, fontweight="bold")
    plt.xticks(ticks=xvals, labels=xvals, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.7)
    plt.legend(bbox_to_anchor=(0.5, 1.05), loc="upper center", ncol=2, fontsize=12, frameon=False)
    sns.despine(top=True, right=True)

    # Save the plot and CSV
    csv_directory = "csvFile"
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    plot_filename = f"{base_filename}.png"
    plt.savefig(plot_filename, dpi=900, transparent=True, bbox_inches="tight")
    csv_filename = f"{base_filename}.csv"
    csv_path = os.path.join(csv_directory, csv_filename)
    data_to_save.to_csv(csv_path, index=False)
    print(f"CSV file saved to {csv_path}")
    
    plt.show()
    return forecasts

def sum_capacity():
    # Load the CSV file with '$' as the delimiter
    df = pd.read_csv('final_dataset_Capacity_kt_H2_y.csv', sep='$')
    
    # Convert the 'Year' column to numeric to handle data type issues
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Filter rows where the Year is <= 2025
    filtered_df = df[df['Year'] <= 2025]
    
    # Sum the 'Total Yearly Capacity' for the filtered years
    total_capacity = filtered_df['Total Yearly Capacity'].sum()
    
    return total_capacity

def compute_and_plot_cumulative_forecast(
    yearly_forecasts,
    target_dict,
    initial_cumulative_kt=50000.0
):
    """
    Fully integrated version: computes cumulative forecasts from model yearly forecasts
    and plots them, starting from a specified historical baseline capacity.

    1. Negative forecasts are replaced with their absolute values (np.abs).
    2. Cumulative = initial_cumulative_kt + cumsum(yearly).
    3. All final data is saved in a CSV and used directly to plot, ensuring the plot
       matches the CSV exactly.
    4. Target lines (in Mt) are converted to kt by multiplying by 1000.

    Parameters
    ----------
    yearly_forecasts : dict
        Dictionary mapping model names -> { "yearly": np.array([...]) } in kt.
        For example:
            {
                "Transformer": {"yearly": np.array([f2025, f2026, ...])},
                "Autoformer": {"yearly": np.array([f2025, f2026, ...])},
                ...
            }
    target_dict : dict
        Dictionary mapping institution -> target in Mt, e.g. {"IEA": 430, "IPCC": 155}.
        We multiply each by 1000 to convert Mt -> kt for the plot.
    initial_cumulative_kt : float, optional
        Historical capacity in kt by end of 2024. Defaults to 50000.0 kt.

    Returns
    -------
    cumulative_forecasts : dict
        Dictionary mapping each model name (as a string) to a 1D array of cumulative
        values for the forecast horizon.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # 1. Determine the forecast horizon from one model's 'yearly' array
    #    (Assume at least one model is present.)
    sample_key = next(iter(yearly_forecasts))
    sample_yearly = np.asarray(yearly_forecasts[sample_key]["yearly"]).flatten()
    n_years = len(sample_yearly)

    # 2. Create a date range from 2025 with n_years (year-end frequency)
    future_years = pd.date_range(start="2025", periods=n_years, freq="Y")
    x_vals = np.array(future_years.year, dtype=int)

    # 3. Prepare a DataFrame for saving the results
    data_to_save = pd.DataFrame({"Year": x_vals})

    # Dictionary to store final cumulative arrays
    cumulative_forecasts = {}

    # === Debug prints: to verify data step by step ===
    print("\n=== DEBUG: Cumulative Forecast Calculation ===")
    print(f"initial_cumulative_kt = {initial_cumulative_kt}")

    # 4. Process each model
    for model_name, info_dict in yearly_forecasts.items():
        mname_str = str(model_name)  # ensure we have a string

        raw_yearly = info_dict["yearly"]
        print(f"\n[DEBUG] {mname_str} raw yearly forecast => {raw_yearly}")

        # Convert to numeric, flatten to 1D
        yearly = np.asarray(raw_yearly).flatten()
        # Replace negatives with absolute values
        yearly = np.abs(yearly)
        print(f"[DEBUG] {mname_str} after converting to abs => {yearly}")

        # cumulative = historical + partial sums
        cumulative = initial_cumulative_kt + np.cumsum(yearly)
        print(f"[DEBUG] {mname_str} final cumulative => {cumulative}")

        # Store in dictionary
        cumulative_forecasts[mname_str] = cumulative

        # Add column to DataFrame
        col_name = f"{mname_str}_Cumulative"
        data_to_save[col_name] = cumulative

    # 5. Ensure numeric columns
    for col in data_to_save.columns:
        if col != "Year":
            data_to_save[col] = pd.to_numeric(data_to_save[col], errors="coerce").fillna(0)

    # Print final DataFrame
    print("\n=== DEBUG: Final DataFrame to Save & Plot ===")
    print(data_to_save)

    # 6. Plot from the DataFrame
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid", context="talk")

    # Create a palette with len(cumulative_forecasts) distinct colors
    palette = sns.color_palette("deep", len(cumulative_forecasts))

    base_filename = "cumulative_capacity_forecast_fully_integrated"

    # We'll skip the 'Year' column for plotting
    model_cols = [c for c in data_to_save.columns if c != "Year"]

    # Plot each model's column
    i = 0
    for col in model_cols:
        x_list = data_to_save["Year"].tolist()
        y_list = data_to_save[col].tolist()
        plt.plot(
            x_list,
            y_list,
            marker="o",
            label=col,
            color=palette[i % len(palette)],
            lw=2
        )
        i += 1

    # 7. Plot target lines (convert Mt to kt)
    target_colors = ["#E74C3C", "#2ECC71", "#3498DB"]
    for (inst, mt_val), color in zip(target_dict.items(), target_colors):
        target_kt = mt_val * 1000.0
        plt.axhline(
            y=target_kt,
            color=color,
            linestyle="--",
            label=f"{inst} Target: {mt_val} Mt ({int(target_kt)} kt)"
        )

    # 8. Final aesthetics
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Cumulative Capacity (kt)", fontsize=14, fontweight="bold")
    plt.title(
        f"Cumulative Forecast (historical={initial_cumulative_kt:,.0f} kt)",
        fontsize=16, fontweight="bold"
    )
    plt.xticks(ticks=x_vals, labels=x_vals, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.7)
    plt.legend(
        bbox_to_anchor=(0.5, 1.05),
        loc="upper center",
        ncol=2,
        fontsize=12,
        frameon=False
    )
    sns.despine(top=True, right=True)

    # 9. Save CSV and plot
    csv_directory = "csvFile"
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    csv_path = os.path.join(csv_directory, f"{base_filename}.csv")
    data_to_save.to_csv(csv_path, index=False)
    print(f"\nCSV file saved to {csv_path}")

    plot_path = f"{base_filename}.png"
    plt.savefig(plot_path, dpi=900, transparent=True, bbox_inches="tight")
    plt.show()

    return cumulative_forecasts

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os

###############################################################################
# 1) Utility to Extract Numeric Array If Dictionary Provided
###############################################################################
def extract_array(data, known_keys=None):
    """
    If 'data' is already a np.array, return it.
    If 'data' is a dict, try known keys like "yearly", "Forecast", etc.
    Otherwise raise an error.
    """
    if known_keys is None:
        known_keys = ["yearly", "Forecast", "Forecasts", "Values", "Capacity"]
    if isinstance(data, dict):
        for k in known_keys:
            if k in data and isinstance(data[k], np.ndarray):
                return data[k]
        raise ValueError(f"Forecast dictionary has no recognized numeric key. Keys={list(data.keys())}")
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise ValueError(f"Expected dict or np.ndarray, got {type(data)}")

###############################################################################
# 2) Environment: Forcing Final Step to Match the Target
###############################################################################
class YearlyEnv:
    """
    Each episode: Steps from index=0..25 (representing 2025..2050).
    At step i, we can scale that year's forecast by ±(action_scale) or do nothing.
    Negative production => clamp to 0.
    On final step, we forcibly fix the last forecast to exactly meet target_kt
    (unless that fix is negative, then clamp => big penalty).
    We do partial shaping rewards so the agent doesn't drift too far from
    a linear path to the target.
    """
    def __init__(self, baseline_forecast, target_kt, historical_base=50000.0, action_scale=0.05):
        # clamp negative from the start
        self.baseline = np.maximum(baseline_forecast, 0.0)
        self.num_years = len(self.baseline)  # typically 26 for 2025..2050
        self.current_idx = 0
        self.target_kt = target_kt
        self.historical_base = historical_base
        self.action_scale = action_scale
        
        # 3 discrete actions => 0=no change, 1=+action_scale, 2=-action_scale
        self.action_size = 3

        # We'll store the adjusted forecast
        self.adjusted = self.baseline.copy()
    
    def reset(self):
        """
        Initialize environment: year_idx=0, adjusted=baseline copy
        Return state => [year_idx, partial_sum]
        """
        self.current_idx = 0
        self.adjusted = self.baseline.copy()
        partial_sum = self.historical_base
        state = np.array([self.current_idx, partial_sum], dtype=float)
        return state

    def step(self, action):
        """
        action in {0,1,2}.
        Then we compute partial sums, check if final, do forced fix.
        Returns (next_state, reward, done)
        """
        # scale current year's forecast
        if action == 1:
            self.adjusted[self.current_idx] *= (1 + self.action_scale)
        elif action == 2:
            self.adjusted[self.current_idx] *= (1 - self.action_scale)
        
        # clamp negative
        if self.adjusted[self.current_idx] < 0:
            self.adjusted[self.current_idx] = 0

        # partial sum
        partial_sum = self.historical_base + self.adjusted[:(self.current_idx + 1)].sum()

        self.current_idx += 1
        done = (self.current_idx >= self.num_years)

        if not done:
            # partial shaping => compare partial_sum to partial_target
            partial_target = self.target_kt * (self.current_idx / self.num_years)
            # negative absolute dev => prefer to be near partial_target
            reward = -abs(partial_sum - partial_target)
            next_state = np.array([self.current_idx, partial_sum], dtype=float)
        else:
            # final => forcibly fix the last forecast
            final_cum = self.historical_base + self.adjusted.sum()
            diff = self.target_kt - final_cum
            self.adjusted[-1] += diff
            if self.adjusted[-1] < 0:
                self.adjusted[-1] = 0.0
            final_cum = self.historical_base + self.adjusted.sum()
            mismatch = abs(final_cum - self.target_kt)
            # large negative penalty for mismatch
            reward = -mismatch * 100.0
            next_state = np.array([self.current_idx, final_cum], dtype=float)
        
        return next_state, reward, done

###############################################################################
# 3) Minimal DQN Implementation
###############################################################################
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0
    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        import random
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.9, alpha=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.alpha = alpha
        
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.memory = ReplayBuffer()
        self.batch_size = 32
        self.learn_step = 0
        self.update_freq = 10
        self.epsilon = 1.0

    def choose_action(self, state):
        import numpy as np
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def store(self, s, a, r, s_, d):
        self.memory.push(s, a, r, s_, d)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        self.epsilon = max(0.05, self.epsilon*0.995)

###############################################################################
# 4) The RL Scenario Analysis Function
###############################################################################
def rl_scenario_analysis(
    yearly_forecasts,         # {model: np.array or dict with "yearly": np.array}, each length=26
    target_dict_mt,           # {institution: target_in_Mt}, convert to kt
    historical_base=50000.0,  # e.g. up to 2024
    episodes=2000
):
    """
    For each (model, institution), we create a YearlyEnv forcibly ensuring final sum hits target exactly,
    negative clamp, partial shaping. We run a DQN for 'episodes' episodes, storing best solution.

    Returns final_results[model][institution] => array(26,) of final forecast
    """

    # 1) Convert target from Mt => kt
    target_kt_dict = {inst: val_mt * 1000 for inst, val_mt in target_dict_mt.items()}

    final_results = {}
    
    # 2) Validate each model's array is length=26, or extract from dict
    for model_name, data in yearly_forecasts.items():
        arr = extract_array(data)
        if arr.shape[0] != 26:
            raise ValueError(f"{model_name} forecast length != 26, got {arr.shape[0]}")

    # 3) Run RL for each (model, institution)
    for model_name, data in yearly_forecasts.items():
        baseline = extract_array(data)
        model_map = {}
        for institution, target_kt in target_kt_dict.items():
            env = YearlyEnv(baseline_forecast=baseline, target_kt=target_kt,
                            historical_base=historical_base, action_scale=0.05)
            agent = DQNAgent(state_size=2, action_size=env.action_size, gamma=0.9, alpha=0.001)
            
            best_forecast = baseline.copy()
            best_mismatch = 1e15

            for ep in range(episodes):
                state = env.reset()
                done = False
                while not done:
                    action = agent.choose_action(state)
                    next_state, reward, done = env.step(action)
                    agent.store(state, action, reward, next_state, float(done))
                    agent.learn()
                    agent.update_epsilon()
                    state = next_state
                
                # Evaluate final mismatch
                final_cum = historical_base + env.adjusted.sum()
                mismatch = abs(final_cum - target_kt)
                if mismatch < best_mismatch:
                    best_mismatch = mismatch
                    best_forecast = env.adjusted.copy()

            model_map[institution] = best_forecast
        
        final_results[model_name] = model_map

    # 4) Plot results: For each institution => yearly lines & cumulative lines
    import pandas as pd
    years = np.arange(2025, 2051)
    for institution, mt_val in target_dict_mt.items():
        t_kt = mt_val * 1000
        # (a) yearly
        plt.figure(figsize=(12,7))
        sns.set_style("whitegrid")
        for model_name, inst_map in final_results.items():
            if institution in inst_map:
                forecast_arr = inst_map[institution]
                plt.plot(years, forecast_arr, label=model_name, lw=2)
        plt.xlabel("Year", fontsize=14)
        plt.ylabel("Yearly Production (kt)", fontsize=14)
        plt.title(f"Yearly RL-Optimized Forecast for {institution} (Forcing final sum = {t_kt})", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        # (b) cumulative
        plt.figure(figsize=(12,7))
        for model_name, inst_map in final_results.items():
            if institution in inst_map:
                forecast_arr = inst_map[institution]
                cum_line = historical_base + np.cumsum(forecast_arr)
                plt.plot(years, cum_line, label=model_name, lw=2)
        plt.axhline(y=t_kt, color='red', lw=2, linestyle='--', label=f"{institution} Target={t_kt:,.0f} kt")
        plt.xlabel("Year", fontsize=14)
        plt.ylabel("Cumulative (kt)", fontsize=14)
        plt.title(f"Cumulative RL-Optimized Forecast for {institution}\n(Final sum forced to match)", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    # 5) Save CSV results
    csv_folder = "csvFile"
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

    for model_name, inst_map in final_results.items():
        df_out = {"Year": years}
        for institution, forecast_arr in inst_map.items():
            df_out[f"{institution}_Yearly"] = forecast_arr
            df_out[f"{institution}_Cumulative"] = historical_base + np.cumsum(forecast_arr)
        import pandas as pd
        df_csv = pd.DataFrame(df_out)
        outpath = os.path.join(csv_folder, f"{model_name}_rl_forecasts.csv")
        df_csv.to_csv(outpath, index=False)
        print(f"Saved => {outpath}")

    return final_results

"""
Strategic Hydrogen Capacity Planner (H2CP)
Integrates diffusion models, temporal graphs, and physics-informed RL
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gym
from gym import spaces
from typing import Dict, Tuple

# Stable-Baselines3 imports
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import torchdiffeq
except ImportError:
    torchdiffeq = None

########################################
# 1. InstitutionGraphPolicy
########################################
class InstitutionGraphPolicy(nn.Module):
    """Minimal example: LSTM + self-attention => final MLP for a 2D action."""
    def __init__(self, node_dim=8, edge_dim=4):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=node_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        # Self-attention on 128-dim features (no separate kdim/vdim needed).
        self.attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4
        )
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, node_states: torch.Tensor, edge_index: torch.Tensor, edge_attrs: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            node_states: Tensor of shape [batch_size, seq_len, node_dim]
        Returns:
            Tensor of shape [batch_size, 2] representing the action.
        """
        rnn_out, _ = self.rnn(node_states)  # [batch_size, seq_len, 128]
        attn_in = rnn_out.transpose(0, 1)     # [seq_len, batch_size, 128]
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        pooled = attn_out.mean(dim=0)         # [batch_size, 128]
        return self.policy_head(pooled)       # [batch_size, 2]

########################################
# 2. H2CDiffusion
########################################
class H2CDiffusion(nn.Module):
    """Physics-informed capacity diffusion module."""
    def __init__(self, horizon=26):
        super().__init__()
        self.horizon = horizon
        self.timesteps = torch.linspace(0, 1, horizon)
        self.drift = nn.Sequential(
            nn.Linear(6, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 2)
        )
        self.final_projection = nn.Linear(2, 1)

    def forward(self, model_forecast, csv_data, historical_base, target) -> torch.Tensor:
        if torchdiffeq is None:
            return torch.zeros_like(model_forecast, dtype=torch.float32)
        x0 = torch.tensor([[historical_base, 0.0]], dtype=torch.float32)
        model_forecast = model_forecast.view(-1, 1)
        csv_clean = csv_data.nan_to_num(0).view(-1, 1)
        target_offset = (target - historical_base) * torch.ones_like(model_forecast)
        nan_mask = torch.isnan(csv_data).float().view(-1, 1)
        params = torch.cat([model_forecast, csv_clean, target_offset, nan_mask], dim=1)
        def ode_wrapper(t, x):
            return self._ode_system(t, x, params)
        sol = torchdiffeq.odeint(
            ode_wrapper,
            x0,
            self.timesteps,
            method='dopri5',
            rtol=1e-4
        )
        return self.final_projection(sol.squeeze(1))

    def _ode_system(self, t, x, params):
        time_idx = min(int(t * (self.horizon - 1)), self.horizon - 1)
        row_params = params[time_idx].unsqueeze(0)
        control_input = torch.cat([x, row_params], dim=1)
        control = self.drift(control_input)
        dxdt = torch.zeros_like(x)
        dxdt[:, 0] = x[:, 1]
        dxdt[:, 1] = control[:, 0] - 0.1 * x[:, 1]
        if t > 0.95:
            remaining = 1.0 - t + 1e-8
            target_error = (row_params[:, 2] - x[:, 0]) / remaining
            dxdt[:, 0] += 10.0 * target_error
        return dxdt
    
########################################
# 3. StrategicCapacityEnv
########################################
class StrategicCapacityEnv(gym.Env):
    """
    Stabilized environment with numerical safeguards for strategic planning.
    Plans yearly capacity toward meeting an institution-specific target by 2050,
    while enforcing CSV constraints and promoting smooth, proportional production.
    """
    def __init__(
        self,
        model_forecasts: Dict[str, dict],  # Each value must contain at least "yearly": np.array([...])
        csv_data: pd.DataFrame,
        targets: Dict[str, float],
        historical_base: float = 50000.0
    ):
        super().__init__()
        self.models = model_forecasts
        self.csv_data = csv_data
        # Convert targets from Mt to kt (do not alter these values)
        self.targets = {k: v * 1000 for k, v in targets.items()}
        self.historical = historical_base

        # Default reward weights (modifiable via set_reward_weights)
        self.model_weight = 1.0
        self.csv_weight = 1.0

        self.adjusted = None
        self.current_year = 0
        self.cumulative = 0.0

        self.csv = self._process_csv(csv_data)

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "model": spaces.Box(low=0, high=np.inf, shape=(26,), dtype=np.float32),
            "csv": spaces.Box(low=0, high=np.inf, shape=(26,), dtype=np.float32),
            "target": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "progress": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        })

        self.diffusion = H2CDiffusion()
        # Do not override externally set values:
        self.current_institution = None
        self.current_model = None

    def _process_csv(self, df: pd.DataFrame) -> Dict[int, float]:
        processed = df[df["Year"] > 2024].set_index("Year")
        return processed["Total Yearly Capacity"].to_dict()

    def reset(self):
        """
        Reset the environment.
        If current_institution or current_model are externally set, they are retained.
        """
        if self.current_institution is None:
            self.current_institution = np.random.choice(list(self.targets.keys()))
        if self.current_model is None:
            self.current_model = np.random.choice(list(self.models.keys()))
        self.current_year = 0
        self.cumulative = float(self.historical)
        self.adjusted = np.clip(
            self.models[self.current_model]["yearly"].copy(),
            0, 1e6
        ).astype(np.float32)
        return self._get_obs()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        model_array = np.array(
            self.models[self.current_model]["yearly"],
            dtype=np.float32
        ).flatten()  # Shape: (26,)
        csv_array = np.array(
            [self.csv.get(y, 0.0) for y in range(2025, 2051)],
            dtype=np.float32
        )
        target_array = np.array([self.targets[self.current_institution]], dtype=np.float32)
        progress_array = np.array([
            self.current_year / 25.0,
            self.cumulative / self.targets[self.current_institution],
            (self.targets[self.current_institution] - self.cumulative) / 1e6
        ], dtype=np.float32)
        return {
            "model": model_array,
            "csv": csv_array,
            "target": target_array,
            "progress": progress_array
        }

    def step(self, action: np.ndarray):
        """
        Execute one planning timestep.
        The RL agent’s predicted adjustment is converted to its absolute value,
        ensuring nonnegative yearly production. CSV constraints are enforced.
        A proportional correction term is added at every timestep so that,
        over the remaining years, the production gradually moves toward the target.
        """
        action = np.clip(action, -1.0, 1.0)
        try:
            with torch.no_grad():
                diffusion_guidance = self.diffusion(
                    torch.tensor(self.models[self.current_model]["yearly"], dtype=torch.float32),
                    torch.tensor([self.csv.get(y, 0.0) for y in range(2025, 2051)], dtype=torch.float32),
                    self.historical,
                    self.targets[self.current_institution]
                ).numpy()
        except Exception as e:
            logging.warning(f"Diffusion failed: {str(e)}")
            diffusion_guidance = np.zeros(26)
        # Compute the adjustment and take its absolute value.
        adjustment = np.abs(0.7 * diffusion_guidance[self.current_year] + 0.3 * action[0])
        adjustment = np.clip(adjustment, 0, 1e4)
        
        year = 2025 + self.current_year
        if year in self.csv:
            csv_value = self.csv[year]
            # Enforce CSV constraint: if the current planned capacity is below CSV, force it upward.
            if self.adjusted[self.current_year] < csv_value:
                self.adjusted[self.current_year] = csv_value
            else:
                adjustment += 0.5 * (csv_value - self.adjusted[self.current_year])
                adjustment = np.clip(adjustment, 0, csv_value)
        
        # Proportional control: compute the remaining gap and add a fraction of it.
        current_target = self.targets[self.current_institution]
        remaining_steps = 26 - self.current_year
        if remaining_steps > 0:
            required_increment = (current_target - self.cumulative) / remaining_steps
        else:
            required_increment = 0.0
        # Add a fraction of the required increment to the adjustment.
        adjustment += 0.5 * required_increment
        
        self.adjusted[self.current_year] += adjustment
        new_cumulative = self.cumulative + self.adjusted[self.current_year]
        # Do not allow cumulative capacity to exceed the target before 2050.
        if self.current_year < 26 and new_cumulative > current_target:
            new_cumulative = current_target
        self.cumulative = new_cumulative
        self.current_year += 1

        # Closeness reward: the closer cumulative is to the target, the better.
        time_remaining = (26 - self.current_year) / 26.0
        target_error = abs(current_target - self.cumulative) / current_target
        closeness_reward = -time_remaining * target_error
        
        reward = self._calculate_reward(adjustment, action[1]) + closeness_reward
        done = self.current_year >= 26

        if any(np.isnan(v).any() for v in self._get_obs().values()) or np.isnan(reward):
            logging.warning("NaN detected, resetting environment")
            return self.reset(), 0.0, True, {}
        return self._get_obs(), float(reward), done, {}

    def _calculate_reward(self, adjustment, momentum):
        """
        Reward calculation that encourages smooth, gradual production toward the institution's target.
        """
        current_target = self.targets[self.current_institution]
        target_progress = np.clip((self.cumulative - self.historical) / current_target, -10, 10)
        model_deviation = np.clip(
            self.adjusted[self.current_year - 1] - self.models[self.current_model]["yearly"][self.current_year - 1],
            -1e4, 1e4
        )
        year = 2025 + self.current_year - 1
        if year in self.csv:
            csv_value = self.csv[year]
            csv_reward = 1.0 if self.adjusted[self.current_year - 1] >= csv_value else -1.0
        else:
            csv_reward = 0.0
        if self.current_year < 26 and self.cumulative > current_target:
            overshoot_penalty = -10.0 * (self.cumulative - current_target) / current_target
        else:
            overshoot_penalty = 0.0

        rewards = {
            'target_tracking': np.tanh(target_progress * 2.0),
            'model_adherence': -0.2 * (model_deviation**2) / 1e6 * self.model_weight,
            'csv_compliance': 0.3 * self.csv_weight * csv_reward,
            'momentum_penalty': -0.1 * (momentum**2),
            'smoothness': -0.05 * (adjustment**2),
            'overshoot_penalty': overshoot_penalty
        }
        return np.clip(sum(rewards.values()), -10, 10)

    def set_reward_weights(self, model_weight: float, csv_weight: float):
        """Adjust weighting for model adherence and CSV compliance."""
        self.model_weight = model_weight
        self.csv_weight = csv_weight

class RegionalStrategicCapacityEnv(gym.Env):
    """
    Environment for regional strategic planning.
    Plans yearly capacity toward meeting an institution-specific target by 2050,
    while enforcing CSV constraints and promoting smooth, proportional production.
    
    For regional analysis, the capacity CSV is expected to include a column with the region's name.
    """
    def __init__(
        self,
        model_forecasts: Dict[str, dict],  # Each value must contain at least "yearly": np.array([...])
        csv_data: pd.DataFrame,
        targets: Dict[str, float],
        historical_base: float = 50000.0,
        region: str = None
    ):
        super().__init__()
        if region is None:
            raise ValueError("A region must be specified for RegionalStrategicCapacityEnv.")
        self.region = region

        self.models = model_forecasts
        self.csv_data = csv_data
        # Convert targets from Mt to kt (do not alter these values)
        self.targets = {k: v * 1000 for k, v in targets.items()}
        self.historical = historical_base

        # Default reward weights (modifiable via set_reward_weights)
        self.model_weight = 1.0
        self.csv_weight = 1.0

        self.adjusted = None
        self.current_year = 0
        self.cumulative = 0.0

        # Process CSV data for the specific region.
        self.csv = self._process_csv(csv_data)

        # Define the action and observation spaces.
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "model": spaces.Box(low=0, high=np.inf, shape=(26,), dtype=np.float32),
            "csv": spaces.Box(low=0, high=np.inf, shape=(26,), dtype=np.float32),
            "target": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "progress": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        })

        self.diffusion = H2CDiffusion()
        # Do not override externally set values:
        self.current_institution = None
        self.current_model = None

    def _process_csv(self, df: pd.DataFrame) -> Dict[int, float]:
        """
        Process CSV data for the region.
        The DataFrame should have a 'Year' column and a column with the region's name.
        Only rows with Year > 2024 are considered.
        Returns a dictionary mapping Year to the capacity (in kt) from the region's column.
        """
        processed = df[df["Year"] > 2024].set_index("Year")
        if self.region not in processed.columns:
            raise KeyError(f"Region '{self.region}' not found in CSV columns: {list(processed.columns)}")
        return processed[self.region].to_dict()

    def reset(self):
        """
        Reset the environment.
        If current_institution or current_model are externally set, they are retained.
        """
        if self.current_institution is None:
            self.current_institution = np.random.choice(list(self.targets.keys()))
        if self.current_model is None:
            self.current_model = np.random.choice(list(self.models.keys()))
        self.current_year = 0
        self.cumulative = float(self.historical)
        # Ensure the forecasted yearly capacity is clipped appropriately.
        self.adjusted = np.clip(
            self.models[self.current_model]["yearly"].copy(),
            0, 1e6
        ).astype(np.float32)
        return self._get_obs()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Construct the observation dictionary.
        The observation consists of:
          - "model": the model's yearly forecast array.
          - "csv": the CSV constraints for years 2025-2050.
          - "target": the current institution's target (in kt).
          - "progress": an array representing the planning progress.
        """
        model_array = np.array(
            self.models[self.current_model]["yearly"],
            dtype=np.float32
        ).flatten()  # Expected shape: (26,)
        csv_array = np.array(
            [self.csv.get(y, 0.0) for y in range(2025, 2051)],
            dtype=np.float32
        )
        target_array = np.array([self.targets[self.current_institution]], dtype=np.float32)
        progress_array = np.array([
            self.current_year / 25.0,
            self.cumulative / self.targets[self.current_institution],
            (self.targets[self.current_institution] - self.cumulative) / 1e6
        ], dtype=np.float32)
        return {
            "model": model_array,
            "csv": csv_array,
            "target": target_array,
            "progress": progress_array
        }

    def step(self, action: np.ndarray):
        """
        Execute one planning timestep.
        The RL agent’s predicted adjustment is converted to its absolute value,
        ensuring nonnegative yearly production. CSV constraints are enforced.
        A proportional correction term is added so that, over the remaining years,
        production gradually moves toward the target.
        """
        action = np.clip(action, -1.0, 1.0)
        try:
            with torch.no_grad():
                diffusion_guidance = self.diffusion(
                    torch.tensor(self.models[self.current_model]["yearly"], dtype=torch.float32),
                    torch.tensor([self.csv.get(y, 0.0) for y in range(2025, 2051)], dtype=torch.float32),
                    self.historical,
                    self.targets[self.current_institution]
                ).numpy()
        except Exception as e:
            logging.warning(f"Diffusion failed: {str(e)}")
            diffusion_guidance = np.zeros(26)
        
        # Compute the adjustment and clip its value.
        adjustment = np.abs(0.7 * diffusion_guidance[self.current_year] + 0.3 * action[0])
        adjustment = np.clip(adjustment, 0, 1e4)
        
        year = 2025 + self.current_year
        if year in self.csv:
            csv_value = self.csv[year]
            # Enforce CSV constraint.
            if self.adjusted[self.current_year] < csv_value:
                self.adjusted[self.current_year] = csv_value
            else:
                adjustment += 0.5 * (csv_value - self.adjusted[self.current_year])
                adjustment = np.clip(adjustment, 0, csv_value)
        
        # Proportional control: add a fraction of the required increment.
        current_target = self.targets[self.current_institution]
        remaining_steps = 26 - self.current_year
        required_increment = (current_target - self.cumulative) / remaining_steps if remaining_steps > 0 else 0.0
        adjustment += 0.5 * required_increment
        
        self.adjusted[self.current_year] += adjustment
        new_cumulative = self.cumulative + self.adjusted[self.current_year]
        if self.current_year < 26 and new_cumulative > current_target:
            new_cumulative = current_target
        self.cumulative = new_cumulative
        self.current_year += 1

        time_remaining = (26 - self.current_year) / 26.0
        target_error = abs(current_target - self.cumulative) / current_target
        closeness_reward = -time_remaining * target_error
        
        reward = self._calculate_reward(adjustment, action[1]) + closeness_reward
        done = self.current_year >= 26

        # Check for NaNs.
        if any(np.isnan(v).any() for v in self._get_obs().values()) or np.isnan(reward):
            logging.warning("NaN detected, resetting environment")
            return self.reset(), 0.0, True, {}
        return self._get_obs(), float(reward), done, {}

    def _calculate_reward(self, adjustment, momentum):
        """
        Calculate the reward to encourage smooth, gradual production toward the institution's target.
        """
        current_target = self.targets[self.current_institution]
        target_progress = np.clip((self.cumulative - self.historical) / current_target, -10, 10)
        model_deviation = np.clip(
            self.adjusted[self.current_year - 1] - self.models[self.current_model]["yearly"][self.current_year - 1],
            -1e4, 1e4
        )
        year = 2025 + self.current_year - 1
        if year in self.csv:
            csv_value = self.csv[year]
            csv_reward = 1.0 if self.adjusted[self.current_year - 1] >= csv_value else -1.0
        else:
            csv_reward = 0.0
        overshoot_penalty = (-10.0 * (self.cumulative - current_target) / current_target 
                             if self.current_year < 26 and self.cumulative > current_target else 0.0)

        rewards = {
            'target_tracking': np.tanh(target_progress * 2.0),
            'model_adherence': -0.2 * (model_deviation**2) / 1e6 * self.model_weight,
            'csv_compliance': 0.3 * self.csv_weight * csv_reward,
            'momentum_penalty': -0.1 * (momentum**2),
            'smoothness': -0.05 * (adjustment**2),
            'overshoot_penalty': overshoot_penalty
        }
        return np.clip(sum(rewards.values()), -10, 10)

    def set_reward_weights(self, model_weight: float, csv_weight: float):
        """Adjust weighting for model adherence and CSV compliance."""
        self.model_weight = model_weight
        self.csv_weight = csv_weight

########################################
# 4. DeepStrategicPlanner
########################################
class DeepStrategicPlanner(SAC):
    def __init__(self, env: gym.Env, **kwargs):
        buffer_size = int(kwargs.pop("buffer_size", 100000))
        super().__init__(
            env=env,
            policy="MultiInputPolicy",
            policy_kwargs=dict(
                features_extractor_class=CombinedExtractor,
                features_extractor_kwargs=dict(cnn_output_dim=128)
            ),
            learning_rate=3e-5,
            buffer_size=buffer_size,
            **kwargs
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph_policy = InstitutionGraphPolicy().to(self.device)

    def learn(self, total_timesteps: int, **kwargs):
        """
        Curriculum learning approach:
        Run multiple phases, setting custom reward weights for each phase.
        """
        phases = [
            ('exploration', 0.3, 0.1),
            ('refinement', 0.5, 0.3),
            ('constraint', 0.2, 0.5)
        ]
        for _, model_weight, csv_weight in phases:
            self.env.env_method("set_reward_weights", model_weight, csv_weight)
            super().learn(total_timesteps // len(phases), **kwargs)

    def predict(self, obs: Dict[str, np.ndarray], deterministic: bool=False) -> Tuple[np.ndarray, None]:
        """
        Overridden to use the custom graph policy.
        Observations from a VecEnv have shape [n_envs, ...]:
          - obs['progress']: shape (n_envs, 3)
          - obs['csv']: shape (n_envs, 26)
          - obs['target']: shape (n_envs, 1)
          - obs['model']: shape (n_envs, 26)
        """
        progress = obs['progress']
        csv_data = obs['csv']
        targets = obs['target']
        models = obs['model']
        n_envs = progress.shape[0]

        node_states_list = []
        for i in range(n_envs):
            node_states_list.append([
                progress[i, 0],
                progress[i, 1],
                progress[i, 2],
                np.nanmean(csv_data[i]),
                np.nanstd(csv_data[i]),
                targets[i, 0] / 1e6,
                np.mean(models[i]),
                np.std(models[i])
            ])
        node_states = torch.tensor(node_states_list, dtype=torch.float32, device=self.device)
        node_states = node_states.unsqueeze(1)  # Shape: [n_envs, 1, 8]
        edge_index, edge_attrs = self._build_relations(n_envs)
        policy_action = self.graph_policy(node_states, edge_index, edge_attrs)
        return policy_action.detach().cpu().numpy(), None

    def _build_relations(self, n_envs: int):
        edge_index = torch.tensor([[0, 0]], dtype=torch.long, device=self.device)
        edge_attrs = torch.tensor([[1.0, 0.5, 0.8, 0.2]], device=self.device)
        return edge_index, edge_attrs
    
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class StabilityCallback(BaseCallback):
    """
    Custom callback for stability monitoring when using off-policy algorithms (e.g., SAC).
    Instead of checking 'rollout_buffer', we now look at the 'replay_buffer'.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.nan_count = 0

    def _on_step(self) -> bool:
        # Access the replay buffer of your off-policy model
        replay_buffer = self.model.replay_buffer
        
        # 1. Check Observations
        # If using a Dict observation space, `replay_buffer.observations` is a dict.
        # If a single Box space, it's just an array.
        if isinstance(replay_buffer.observations, dict):
            # Multi-input / dict observation scenario
            for key, obs_array in replay_buffer.observations.items():
                if np.isnan(obs_array).any():
                    self.nan_count += 1
                    self.logger.warn(f"NaN detected in replay buffer observations['{key}'] "
                                     f"({self.nan_count}/10 allowed)")
        else:
            # Single Box observation scenario
            if np.isnan(replay_buffer.observations).any():
                self.nan_count += 1
                self.logger.warn(f"NaN detected in replay buffer observations "
                                 f"({self.nan_count}/10 allowed)")
        
        # 2. Check Rewards
        if np.isnan(replay_buffer.rewards).any():
            self.nan_count += 1
            self.logger.warn(f"NaN detected in replay buffer rewards ({self.nan_count}/10 allowed)")
        
        # 3. If NaNs keep appearing too often, abort
        if self.nan_count > 10:
            raise RuntimeError("Persistent NaN values detected in training data.")
            
        return True

class CheckNanCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        # Check current rollout data
        if np.isnan(self.locals['rewards']).any():
            self.logger.error("NaN reward detected!")
            return False  # Stop training
        return True
   
# --------------------------
# Visualization and Analysis
# --------------------------
def plot_strategic_results(results: dict, capacity_df: pd.DataFrame, target_dict: dict):
    """
    Enhanced visualization of strategic plans with institution-specific target lines.
    
    Parameters:
        results: dict mapping model names to a dict of institution results,
                 where each result contains 'yearly' and 'cumulative' arrays.
        capacity_df: DataFrame containing CSV capacity data (for scatter plots).
        target_dict: dict with institution targets in Mt (e.g., {"IEA": 430, "IPCC": 155, "IRENA": 523}).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    years = list(range(2025, 2051))
    csv_years = capacity_df[capacity_df['Year'] > 2024]['Year'].values
    
    for model, institutions in results.items():
        for inst, data in institutions.items():
            fig, ax = plt.subplots(2, 1, figsize=(14, 10))
            
            # Yearly Capacity Plot
            ax[0].plot(years, data['yearly'], 'b-', lw=2, label='Strategic Plan')
            ax[0].scatter(csv_years, [data['yearly'][y - 2025] for y in csv_years],
                        c='r', label='CSV Constraints')
            ax[0].set_title(f"{model} - {inst} Yearly Capacity Plan")
            ax[0].legend()
            
            # Cumulative Progress Plot
            ax[1].plot(years, data['cumulative'], 'g-', label='Cumulative')
            target_kt = target_dict[inst] * 1000  # Convert Mt to kt
            ax[1].axhline(y=target_kt, color='r', linestyle='--', label=f"{inst} Target: {target_dict[inst]} Mt")
            ax[1].set_title(f"Cumulative Progress Towards {inst} Target")
            ax[1].legend()
            
            plt.tight_layout()
            plt.show()

def save_strategic_plans(plans: dict):
    """Save strategic plans to CSV"""
    os.makedirs("strategic_plans", exist_ok=True)
    
    for model_name, institutions in plans.items():
        for institution, data in institutions.items():
            df = pd.DataFrame({
                'Year': list(range(2025, 2051)),
                'Yearly_kt': data['yearly'],
                'Cumulative_kt': data['cumulative']
            })
            path = f"strategic_plans/{model_name}_{institution}_plan.csv"
            df.to_csv(path, index=False)

def plot_regional_strategic_results(results: dict, capacity_df: pd.DataFrame, target_dict: dict, region: str):
    """
    Enhanced visualization of regional strategic plans with institution-specific target lines.
    Saves each plot (with high DPI) in the folder 
    "strategic_plans/Regional Analysis/[Region Name]".

    Parameters:
        results: dict mapping model names to a dict of institution results,
                 where each result contains 'yearly' and 'cumulative' arrays.
        capacity_df: DataFrame containing CSV capacity data for the region (for scatter plots).
                     Expected to include at least 'Year' and the region's column.
        target_dict: dict with institution targets in Mt (e.g., {"IEA": 430, "IPCC": 155, "IRENA": 523}).
        region: str, the region name (used in plot titles and file names).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Create directory for saving plots
    save_dir = os.path.join("strategic_plans", "Regional Analysis", region)
    os.makedirs(save_dir, exist_ok=True)

    years = list(range(2025, 2051))
    csv_years = capacity_df[capacity_df['Year'] > 2024]['Year'].values

    for model, institutions in results.items():
        for inst, data in institutions.items():
            fig, ax = plt.subplots(2, 1, figsize=(14, 10))

            # Yearly Capacity Plot
            ax[0].plot(years, data['yearly'], 'b-', lw=2, label='Strategic Plan')
            ax[0].scatter(csv_years, [data['yearly'][y - 2025] for y in csv_years],
                          c='r', label='CSV Constraints')
            ax[0].set_title(f"{model} - {inst} Yearly Capacity Plan for {region}")
            ax[0].legend()

            # Cumulative Progress Plot
            ax[1].plot(years, data['cumulative'], 'g-', label='Cumulative')
            target_kt = target_dict[inst] * 1000  # Convert Mt to kt
            ax[1].axhline(y=target_kt, color='r', linestyle='--',
                          label=f"{inst} Target: {target_dict[inst]} Mt")
            ax[1].set_title(f"Cumulative Progress Towards {inst} Target for {region}")
            ax[1].legend()

            plt.tight_layout()

            # Save the figure with high DPI (e.g., 900) to the specified folder
            filename = f"{model}_{inst}_{region}_strategic_plan.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=900, transparent=True, bbox_inches="tight")
            plt.close(fig)  # Close the figure to free memory

def save_regional_strategic_plans(plans: dict, region: str):
    """
    Save regional strategic plans to CSV files in the folder
    "strategic_plans/Regional Analysis/[Region Name]".

    Parameters:
        plans: dict mapping model names to a dict of institution results,
               where each result contains 'yearly' and 'cumulative' arrays.
        region: str, the region name to include in the saved file names.
    """
    import os
    import pandas as pd

    # Create directory for saving CSV files
    save_dir = os.path.join("strategic_plans", "Regional Analysis", region)
    os.makedirs(save_dir, exist_ok=True)

    for model_name, institutions in plans.items():
        for institution, data in institutions.items():
            df = pd.DataFrame({
                'Year': list(range(2025, 2051)),
                'Yearly_kt': data['yearly'],
                'Cumulative_kt': data['cumulative']
            })
            filename = f"{model_name}_{institution}_{region}_plan.csv"
            path = os.path.join(save_dir, filename)
            df.to_csv(path, index=False)

def sum_regional_capacity() -> dict:
    """
    Computes the historical capacity for each region from the CSV file.
    
    The CSV file is assumed to have a 'Year' column and several regional capacity columns,
    with '$' as the delimiter. Only rows with Year <= 2025 are considered.
    
    Returns
    -------
    dict
        A dictionary mapping each region (column name) to its total capacity (in kt) 
        for years <= 2025.
    """
    df = pd.read_csv('final_dataset_Capacity_kt_H2_y.csv', sep='$')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    filtered_df = df[df['Year'] <= 2025]
    
    # Identify region columns (excluding 'Year' and 'Total Yearly Capacity')
    exclude_cols = ['Year', 'Total Yearly Capacity']
    region_cols = [c for c in filtered_df.columns if c not in exclude_cols]
    
    regional_capacities = {}
    for region in region_cols:
        regional_capacities[region] = filtered_df[region].sum()
    
    return regional_capacities


def compute_regional_contribution(regional_capacity_dict: dict, global_target_dict: dict) -> dict:
    """
    Computes the inferred regional target for each institution based on each region's
    historical capacity contribution relative to the total (global) capacity.
    
    For each region, the share is calculated as:
        share = (regional capacity) / (sum of all regional capacities)
    Then, for each institution, the inferred regional target (in Mt) is:
        regional_target = global_target * share
    
    Parameters
    ----------
    regional_capacity_dict : dict
        Dictionary mapping each region to its historical capacity (in kt).
    global_target_dict : dict
        Dictionary mapping institution names to their global targets in Mt.
        Example: {"IEA": 430, "IPCC": 155, "IRENA": 523}
    
    Returns
    -------
    dict
        A dictionary where each key is a region and the corresponding value is another
        dictionary mapping institution names to their inferred regional targets (in Mt).
    """
    global_sum = sum(regional_capacity_dict.values())
    regional_target_dict = {}
    
    for region, capacity in regional_capacity_dict.items():
        share = capacity / global_sum if global_sum != 0 else 0
        # Scale each institution's global target by the region's share
        regional_target_dict[region] = {inst: target * share for inst, target in global_target_dict.items()}
    
    return regional_target_dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_regional_cumulative_forecast(yearly_forecasts, region, target_dict, initial_cumulative_kt):
    """
    Computes cumulative forecasts from model yearly forecasts for a specific region and saves the plot and CSV.
    
    1. Negative forecasts are replaced with their absolute values.
    2. Cumulative forecast = initial_cumulative_kt + cumsum(yearly).
    3. The final data is saved to a CSV and used to generate a plot.
    4. Target lines (given in Mt) are converted to kt by multiplying by 1000.
    
    Parameters
    ----------
    yearly_forecasts : dict
        Dictionary mapping model names -> {"yearly": np.array([...])} in kt.
    region : str
        The region name (used for titles and file names).
    target_dict : dict
        Dictionary mapping institution -> target in Mt.
    initial_cumulative_kt : float
        Historical capacity in kt by the end of the baseline period.
    
    Returns
    -------
    cumulative_forecasts : dict
        Dictionary mapping each model name (as a string) to a 1D array of cumulative forecast values.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # 1. Determine forecast horizon from one model's "yearly" array
    sample_key = next(iter(yearly_forecasts))
    sample_yearly = np.asarray(yearly_forecasts[sample_key]["yearly"]).flatten()
    n_years = len(sample_yearly)
    
    # 2. Create a date range from 2025 with n_years (year-end frequency)
    future_years = pd.date_range(start="2025", periods=n_years, freq="Y")
    x_vals = np.array(future_years.year, dtype=int)
    
    # 3. Prepare a DataFrame for saving the results
    data_to_save = pd.DataFrame({"Year": x_vals})
    cumulative_forecasts = {}
    
    print("\n=== DEBUG: Regional Cumulative Forecast Calculation ===")
    print(f"Region: {region}, initial_cumulative_kt = {initial_cumulative_kt}")
    
    # 4. Process each model's forecast
    for model_name, info_dict in yearly_forecasts.items():
        mname_str = str(model_name)
        raw_yearly = info_dict["yearly"]
        print(f"\n[DEBUG] {mname_str} raw yearly forecast => {raw_yearly}")
        
        # Convert to 1D numeric array and replace negatives with absolute values
        yearly = np.abs(np.asarray(raw_yearly).flatten())
        print(f"[DEBUG] {mname_str} after abs conversion => {yearly}")
        
        # Compute cumulative forecast
        cumulative = initial_cumulative_kt + np.cumsum(yearly)
        print(f"[DEBUG] {mname_str} final cumulative => {cumulative}")
        
        cumulative_forecasts[mname_str] = cumulative
        data_to_save[f"{mname_str}_Cumulative"] = cumulative
    
    # Ensure all numeric columns are properly formatted
    for col in data_to_save.columns:
        if col != "Year":
            data_to_save[col] = pd.to_numeric(data_to_save[col], errors="coerce").fillna(0)
    
    print("\n=== DEBUG: Final DataFrame for Region ===")
    print(data_to_save)
    
    # 5. Plot cumulative forecasts
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("deep", len(cumulative_forecasts))
    
    base_filename = f"cumulative_capacity_forecast_{region}_integrated"
    model_cols = [c for c in data_to_save.columns if c != "Year"]
    for i, col in enumerate(model_cols):
        plt.plot(
            data_to_save["Year"],
            data_to_save[col],
            marker="o",
            label=col,
            color=palette[i % len(palette)],
            lw=2
        )
    
    # 6. Plot target lines (convert Mt to kt)
    target_colors = ["#E74C3C", "#2ECC71", "#3498DB"]
    for (inst, mt_val), color in zip(target_dict.items(), target_colors):
        target_kt = mt_val * 1000.0
        plt.axhline(
            y=target_kt,
            color=color,
            linestyle="--",
            label=f"{inst} Target: {mt_val} Mt ({int(target_kt)} kt)"
        )
    
    # 7. Aesthetics and saving
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Cumulative Capacity (kt)", fontsize=14, fontweight="bold")
    plt.title(
        f"Cumulative Forecast for {region} (historical = {initial_cumulative_kt:,.0f} kt)",
        fontsize=16, fontweight="bold"
    )
    plt.xticks(ticks=x_vals, labels=x_vals, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.7)
    plt.legend(bbox_to_anchor=(0.5, 1.05), loc="upper center", ncol=2, fontsize=12, frameon=False)
    sns.despine(top=True, right=True)
    
    # Save files in a region-specific folder
    save_dir = os.path.join("strategic_plans", "Regional Analysis", region)
    os.makedirs(save_dir, exist_ok=True)
    
    csv_path = os.path.join(save_dir, f"{base_filename}.csv")
    data_to_save.to_csv(csv_path, index=False)
    print(f"\nCSV file saved to {csv_path}")
    
    plot_path = os.path.join(save_dir, f"{base_filename}.png")
    plt.savefig(plot_path, dpi=900, transparent=True, bbox_inches="tight")
    plt.close()  # Close the figure after saving
    
    return cumulative_forecasts


def predict_future_regional_total_capacity(models, df, sequence_length, region):
    """
    Uses a sliding-window approach to forecast future capacity for a specific region.
    
    For regional forecasting, only the following features are used:
        - 'Year' and the specified region column.
    The last `sequence_length` rows of df (for these features) form the initial input.
    Forecasts are generated for future years (e.g. from 2025 to 2050).
    
    Parameters
    ----------
    models : dict
        Dictionary mapping model names to trained model instances.
    df : pd.DataFrame
        The augmented DataFrame (in original scale) used for training/inference.
    sequence_length : int
        Number of rows (time steps) to use as the input window.
    region : str
        The region column to use for forecasting.
    
    Returns
    -------
    forecasts : dict
        Dictionary mapping each model name to {"yearly": np.array([...])},
        representing yearly forecasts (in kt).
    """
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Use only 'Year' and the region column as features
    features = ['Year', region]
    
    # Future years: from 2025 to 2050 (inclusive)
    future_years = pd.date_range(start="2025", end="2051", freq="Y")
    num_future = len(future_years)
    data_to_save = pd.DataFrame({"Year": future_years.year})
    
    # Set up plot aesthetics
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("deep", len(models))
    base_filename = f"regional_forecast_{region}_professional"
    
    forecasts = {}
    any_success = False
    
    # Prepare the initial input window from the region-specific features
    init_data = df[features].iloc[-sequence_length:].values
    # (Assumes that training and inference use the same column order.)
    
    for i, (model_name, model) in enumerate(models.items()):
        model.eval()
        predictions = []
        device = next(model.parameters()).device
        
        # For simplicity, assume a generic model that expects a 3D tensor input.
        input_sequence = torch.tensor(init_data, dtype=torch.float32, device=device).unsqueeze(0)
        print(f"[DEBUG] {model_name} initial input shape: {list(input_sequence.shape)}")
        
        produced_forecasts = True
        with torch.no_grad():
            for year in future_years:
                try:
                    output = model(input_sequence)
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    # For models that produce a 3D output: [1, seq_len, feature_dim]
                    if output.dim() == 3:
                        pred = output[:, -1, :].squeeze(0).cpu().numpy()  # shape: (feature_dim,)
                    elif output.dim() == 2:
                        pred = output.squeeze(0).cpu().numpy()
                    elif output.dim() == 1:
                        pred = output.cpu().numpy()
                    else:
                        raise ValueError(
                            f"[{model_name}] Unexpected output shape: {list(output.shape)}"
                        )
                    predictions.append(pred)
                    
                    # Slide the window: remove the first time step and append the new prediction.
                    arr = input_sequence.cpu().numpy()  # shape: [1, seq_len, feature_dim]
                    arr = arr[:, 1:, :]  # new shape: [1, seq_len-1, feature_dim]
                    n_feat = arr.shape[-1]
                    if pred.ndim == 1:
                        if pred.shape[0] < n_feat:
                            pred = np.pad(pred, (0, n_feat - pred.shape[0]), 'constant')
                        elif pred.shape[0] > n_feat:
                            pred = pred[:n_feat]
                        pred_3d = pred.reshape(1, 1, n_feat)
                    else:
                        raise ValueError(f"[{model_name}] Unexpected prediction shape: {pred.shape}")
                    new_arr = np.concatenate([arr, pred_3d], axis=1)
                    input_sequence = torch.tensor(new_arr, dtype=torch.float32, device=device)
                except Exception as e:
                    print(f"Error forecasting with {model_name} for year {year.year}: {e}")
                    produced_forecasts = False
                    break
        
        if not produced_forecasts or len(predictions) == 0:
            print(f"No forecasts produced by {model_name}; skipping.")
            continue
        
        any_success = True
        predictions = np.array(predictions, dtype=float)
        # Replace negatives with absolute values and scale (here multiplied by 10)
        predictions = np.abs(predictions) * 10
        
        # For multivariate outputs, sum across features to obtain a scalar per year.
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            total_yearly = predictions.sum(axis=1)
        else:
            total_yearly = predictions.flatten()
        
        forecasts[model_name] = {"yearly": total_yearly}
        data_to_save[f"{model_name}_Yearly"] = total_yearly
        
        xvals = future_years.year.astype(int)
        sns.lineplot(
            x=xvals,
            y=total_yearly,
            label=f"{model_name} Yearly",
            color=palette[i],
            lw=2
        )
    
    if not any_success:
        print("No models produced forecasts. Exiting without saving plot/CSV.")
        return {}
    
    plt.xlabel("Year", fontsize=14, fontweight="bold")
    plt.ylabel("Capacity (kt)", fontsize=14, fontweight="bold")
    plt.title(f"Yearly Capacity Forecast for {region}", fontsize=16, fontweight="bold")
    plt.xticks(ticks=xvals, labels=xvals, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.7)
    plt.legend(bbox_to_anchor=(0.5, 1.05), loc="upper center", ncol=2, fontsize=12, frameon=False)
    sns.despine(top=True, right=True)
    
    # Save files in a region-specific folder
    save_dir = os.path.join("strategic_plans", "Regional Analysis", region)
    os.makedirs(save_dir, exist_ok=True)
    
    plot_filename = f"{base_filename}.png"
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path, dpi=900, transparent=True, bbox_inches="tight")
    
    csv_filename = f"{base_filename}.csv"
    csv_path = os.path.join(save_dir, csv_filename)
    data_to_save.to_csv(csv_path, index=False)
    print(f"CSV file saved to {csv_path}")
    
    plt.close()  # Close the figure after saving
    
    return forecasts

###############################################################################
# 7. MAIN LOGIC
###############################################################################
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global parameters
SEQUENCE_LENGTH = 12
SEED = 42
BATCH_SIZE = 32

# Global target dictionary (in Mt) for institutions
global_target_dict = {
    "IEA": 430,
    "IPCC": 155,
    "IRENA": 523
}

# Define weight initialization (top-level, not nested)
def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def plot_all_models_regional(strategic_plans, region, regional_targets):
    """Plot all regional strategic plans with enhanced formatting and error handling."""
    years = list(range(2025, 2051))
    palette = sns.color_palette("husl", len(strategic_plans))
    save_dir = f"strategic_plans/Regional_Analysis/{region}"
    os.makedirs(save_dir, exist_ok=True)
    
    plt.style.use("seaborn-whitegrid")
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.titlesize': 16,
        'axes.labelweight': 'bold'
    })

    for institution in regional_targets[region].keys():
        fig, ax = plt.subplots(1, 2, figsize=(22, 8))
        fig.suptitle(f"{region} Strategic Plans - {institution}", 
                    fontsize=20, y=1.05, weight='bold')
        
        # Yearly Production Plot
        for i, (model, plans) in enumerate(strategic_plans.items()):
            if institution in plans:  # Added safety check
                ax[0].plot(years, plans[institution]['yearly'], 
                        color=palette[i], label=model, lw=2.5)
        ax[0].set_title(f"Annual Capacity Forecast", fontsize=16)
        ax[0].set_ylabel("Capacity (kt)", fontsize=14)
        
        # Cumulative Plot
        for i, (model, plans) in enumerate(strategic_plans.items()):
            if institution in plans:
                ax[1].plot(years, plans[institution]['cumulative'], 
                        color=palette[i], lw=2.5)
        ax[1].axhline(regional_targets[region][institution] * 1000, 
                    color='#d62728', linestyle='--', lw=3, 
                    label='2050 Target')
        ax[1].set_title("Cumulative Pathway", fontsize=16)
        ax[1].set_ylabel("Cumulative Capacity (kt)", fontsize=14)
        
        # Unified formatting
        for a in ax:
            a.set_xlabel("Year", fontsize=14)
            a.set_xticks(range(2025, 2051, 5))
            a.tick_params(labelsize=12)
            a.grid(True, alpha=0.4)
        
        plt.tight_layout()
        filename = f"{save_dir}/{datetime.now().strftime('%Y%m%d')}_{region}_{institution}_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def plot_median_models_regional(strategic_plans, region, regional_targets):
    """Plot median regional forecasts with dynamic color handling."""
    years = list(range(2025, 2051))
    save_dir = f"strategic_plans/Regional_Analysis/{region}"
    os.makedirs(save_dir, exist_ok=True)
    
    plt.style.use("seaborn-whitegrid")
    color_palette = sns.color_palette("husl", 3)  # For 3 institutions
    
    for institution in regional_targets[region].keys():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
        fig.suptitle(f"{region} Median Forecast - {institution}", 
                    fontsize=20, y=1.05, weight='bold')
        
        # Calculate statistics with NaN protection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_yearly = [p[institution]['yearly'] 
                        for p in strategic_plans.values() 
                        if institution in p]
            median_yearly = np.nanmedian(all_yearly, axis=0)
            all_cumulative = [p[institution]['cumulative'] 
                            for p in strategic_plans.values() 
                            if institution in p]
            median_cumulative = np.nanmedian(all_cumulative, axis=0)
        
        # Yearly plot
        ax1.fill_between(years, 
                        np.nanpercentile(all_yearly, 25, axis=0),
                        np.nanpercentile(all_yearly, 75, axis=0),
                        color=color_palette[0], alpha=0.2)
        ax1.plot(years, median_yearly, color=color_palette[0], 
                label='Median Forecast')
        ax1.set_title("Yearly Capacity Distribution", fontsize=16)
        
        # Cumulative plot
        ax2.plot(years, median_cumulative, color=color_palette[1],
                label='Median Pathway')
        ax2.axhline(regional_targets[region][institution] * 1000,
                    color=color_palette[2], linestyle='--',
                    label='2050 Target')
        ax2.fill_between(years,
                        np.nanpercentile(all_cumulative, 10, axis=0),
                        np.nanpercentile(all_cumulative, 90, axis=0),
                        color=color_palette[1], alpha=0.2)
        ax2.set_title("Cumulative Pathway Distribution", fontsize=16)
        
        # Unified formatting
        for ax in (ax1, ax2):
            ax.set_xlabel("Year", fontsize=14)
            ax.set_ylabel("Capacity (kt)", fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.4)
        
        plt.tight_layout()
        filename = f"{save_dir}/{datetime.now().strftime('%Y%m%d')}_{region}_{institution}_median.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def plot_comparison_regional(strategic_plans, yearly_forecasts, region, 
                            regional_targets, historical_base):
    """Compare strategic plans with base forecasts for a region."""
    years = list(range(2025, 2051))
    save_dir = f"strategic_plans/Regional_Analysis/{region}"
    os.makedirs(save_dir, exist_ok=True)
    
    plt.style.use("seaborn-whitegrid")
    comparison_colors = ['#1f77b4', '#ff7f0e']  # Strategic vs Forecast
    
    for institution in regional_targets[region].keys():
        fig, axs = plt.subplots(2, 2, figsize=(24, 14))
        fig.suptitle(f"{region} Forecast Comparison - {institution}",
                    fontsize=22, y=1.02, weight='bold')
        
        # Calculate metrics with error handling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            strategic_data = [p[institution]['yearly'] 
                            for p in strategic_plans.values() 
                            if institution in p]
            forecast_data = [f['yearly'] for f in yearly_forecasts.values()]
            
            metrics = {
                'strat_med': np.nanmedian(strategic_data, axis=0),
                'strat_mean': np.nanmean(strategic_data, axis=0),
                'fcst_med': np.nanmedian(forecast_data, axis=0),
                'fcst_mean': np.nanmean(forecast_data, axis=0)
            }
        
        # Plot configurations
        plot_configs = [
            ('Median Comparison', metrics['strat_med'], metrics['fcst_med']),
            ('Mean Comparison', metrics['strat_mean'], metrics['fcst_mean'])
        ]
        
        for idx, (title, strat, fcst) in enumerate(plot_configs):
            # Yearly plots
            axs[0, idx].plot(years, strat, color=comparison_colors[0], 
                            label='Strategic')
            axs[0, idx].plot(years, fcst, color=comparison_colors[1], 
                            linestyle='--', label='Forecast')
            axs[0, idx].set_title(f"Yearly {title}", fontsize=16)
            
            # Cumulative plots
            strat_cum = historical_base + np.cumsum(strat)
            fcst_cum = historical_base + np.cumsum(fcst)
            axs[1, idx].plot(years, strat_cum, color=comparison_colors[0])
            axs[1, idx].plot(years, fcst_cum, color=comparison_colors[1], 
                            linestyle='--')
            axs[1, idx].axhline(regional_targets[region][institution] * 1000,
                                color='#d62728', linestyle=':')
            axs[1, idx].set_title(f"Cumulative {title}", fontsize=16)
        
        # Formatting
        for ax in axs.flat:
            ax.set_xlabel("Year", fontsize=14)
            ax.set_xticks(range(2025, 2051, 5))
            ax.tick_params(labelsize=12)
            ax.grid(True, alpha=0.4)
            ax.legend()
        
        plt.tight_layout()
        filename = f"{save_dir}/{datetime.now().strftime('%Y%m%d')}_{region}_{institution}_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # ------------------------------
    # Global Analysis
    # ------------------------------
    
    # 1. Load the augmented dataset
    augmented_data_path = "aggregated_data/augmented_datasets/augmented_data.csv"
    df_aug = pd.read_csv(augmented_data_path)
    logging.info(f"Loaded augmented dataset from: {augmented_data_path}, shape: {df_aug.shape}")
        
    # ------------------------------
    # Regional Analysis
    # ------------------------------
    # Pre-compute historical capacities and inferred targets per region
    regional_capacities = sum_regional_capacity()       # dict: {region: historical_capacity}
    global_sum_regions = sum(regional_capacities.values())
    regional_targets = compute_regional_contribution(regional_capacities, global_target_dict)
    print('regional_targets', regional_targets)

    criterion = torch.nn.SmoothL1Loss(beta=0.1)

    # Identify region columns (all columns except 'Year' and 'Total Yearly Capacity')
    exclude_cols = ['Year', 'Total Yearly Capacity']
    region_cols = [c for c in df_aug.columns if c not in exclude_cols]

    for region in region_cols:
        logging.info(f"\n[Regional Analysis] Region: {region}")
        
        # Prepare region-specific dataset and dataloaders
        region_dataset = prepare_regional_dataset(df_aug, region, SEQUENCE_LENGTH)
        region_train_loader, region_val_loader, region_test_loader = build_dataloaders(region_dataset, SEED, BATCH_SIZE)
        
        # Infer dimensions for the region
        sample_batch_r = next(iter(region_train_loader))
        inputs_sample_r, labels_sample_r = sample_batch_r
        region_input_dim = inputs_sample_r.shape[2]
        region_output_dim = labels_sample_r.shape[1] if labels_sample_r.dim() > 1 else 1
        logging.info(f"[Region={region}] input_dim={region_input_dim}, output_dim={region_output_dim}")
        
        # Initialize regional models
        region_models = {
            'Transformer': Transformer(
                input_dim=region_input_dim, output_dim=region_output_dim,
                d_model=512, n_head=512, num_layers=4,
                dim_feedforward=64, dropout=0.0001
            ).to(device),
            'Autoformer': Autoformer(
                input_dim=region_input_dim, output_dim=region_output_dim,
                d_model=64, n_heads=4, d_ff=64, e_layers=1,
                moving_avg_kernel=7, dropout=0.01, device=device
            ).to(device),
            'Informer': Informer(
                input_dim=region_input_dim, output_dim=region_output_dim,
                d_model=1024, n_heads=256, ff_dim=128, n_layers=1
            ).to(device),
            'DeepAR': DeepAR(
                input_dim=region_input_dim, hidden_dim=16, output_dim=region_output_dim, num_layers=100
            ).to(device),
            'Reformer': Reformer(
                input_dim=region_input_dim, output_dim=region_output_dim,
                d_model=256, n_heads=16, ff_dim=128, n_layers=1
            ).to(device),
            'TideNet': TideNet(
                input_dim=region_input_dim, hidden_dim=64, output_dim=region_output_dim
            ).to(device),
            'LogSparseTransformer': LogSparseTransformer(
                input_dim=region_input_dim, output_dim=region_output_dim,
                d_model=1024, n_heads=4, ff_dim=128, n_layers=1
            ).to(device),
            'WaveNet': WaveNet(
                input_dim=region_input_dim,
                residual_channels=64,
                output_dim=region_output_dim,
                dilation_depth=8,
                device=device
            ).to(device),
            'TCN': TemporalConvNetModel(
                input_dim=region_input_dim,
                output_dim=region_output_dim,
                num_channels=[64, 128, 256],
                kernel_size=2,
                dropout=0.2,
                device=device
            ).to(device)
        }
        
        # Initialize weights for regional models
        for name, model in region_models.items():
            model.apply(initialize_weights)
        
        # Containers for regional results and histories
        region_results = {}
        region_train_histories = {}
        region_val_histories = {}
        
        # Train & evaluate regional models
        for name, model in region_models.items():
            logging.info(f"[Region={region}] Training Model: {name}")
            optimizer, scheduler = configure_optimizer(model)
            trained_model, train_losses, val_losses = regional_train_model(
                model,
                region_train_loader,
                region_val_loader,
                criterion,
                optimizer,
                scheduler,
                num_epochs=2000,
                patience=60
            )
            region_train_histories[name] = train_losses
            region_val_histories[name] = val_losses
            mse, mae, r2, rmse = evaluate_model(
                trained_model,
                region_test_loader,
                region_output_dim,
                target_scaler=None
            )
            region_results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2, 'RMSE': rmse}
            logging.info(f"[Region={region}] {name} => MSE={mse:.2f}, MAE={mae:.2f}, R2={r2:.4f}, RMSE={rmse:.2f}")
        
        logging.info(f"\n[Region={region}] Model Performance Comparison:")
        performance_list = []
        for model_name, metrics in region_results.items():
            msg = (f"{model_name}: MSE={metrics['MSE']:.2f}, MAE={metrics['MAE']:.2f}, "
                f"R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}")
            logging.info(msg)
            performance_list.append({
                "Model": model_name,
                "MSE": metrics["MSE"],
                "MAE": metrics["MAE"],
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"]
            })

        # Create a DataFrame and save the metrics to a CSV file in a region-specific folder.
        df_metrics = pd.DataFrame(performance_list)
        save_dir = os.path.join("strategic_plans", "Regional Analysis", region)
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"{region}_model_performance_comparison.csv")
        df_metrics.to_csv(csv_path, index=False)
        logging.info(f"Model performance metrics saved to: {csv_path}")
        
        # Regional Forecasting & Strategic Planning
        yearly_forecasts = predict_future_regional_total_capacity(
            models=region_models,
            df=df_aug,
            sequence_length=SEQUENCE_LENGTH,
            region=region
        )
        logging.info("Regional forecasting completed. Check saved plot/CSV and returned forecasts.")
        
        # Compute and plot cumulative forecast for the region
        plot_regional_cumulative_forecast(
            yearly_forecasts,
            region,
            regional_targets[region],
            regional_capacities[region]
        )
        
        # Read capacity data and filter to current region
        capacity_csv_path = "final_dataset_Capacity_kt_H2_y.csv"
        df_capacity = pd.read_csv(capacity_csv_path, sep='$')
        df_capacity = df_capacity[['Year', region]]
        
        logging.info("\nInitializing Regional Strategic Capacity Planner...")
        # Now use the RegionalStrategicCapacityEnv instead of the global one.
        capacity_env = RegionalStrategicCapacityEnv(
            model_forecasts=yearly_forecasts,
            csv_data=df_capacity,
            targets=regional_targets[region],
            historical_base=regional_capacities[region],
            region=region
        )
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv([lambda: capacity_env])
        agent = DeepStrategicPlanner(
            vec_env,
            buffer_size=10000,
            batch_size=512,
            ent_coef='auto',
            target_entropy='auto'
        )
        logging.info("Training regional strategic planner...")
        agent.learn(
            total_timesteps=10000,
            callback=[StabilityCallback(), CheckNanCallback()],
            progress_bar=True
        )
        
        strategic_plans = {}
        for model_name in yearly_forecasts.keys():
            model_plans = {}
            for institution in regional_targets[region].keys():
                capacity_env.current_model = model_name
                capacity_env.current_institution = institution
                obs = capacity_env.reset()
                done = False
                while not done:
                    single_obs = {k: v[None, ...] for k, v in obs.items()}
                    action, _ = agent.predict(single_obs)
                    obs, _, done, _ = capacity_env.step(action[0])
                model_plans[institution] = {
                    'yearly': capacity_env.adjusted,
                    'cumulative': regional_capacities[region] + np.cumsum(capacity_env.adjusted)
                }
            strategic_plans[model_name] = model_plans
        
        # Call the regional strategic plotting function.
        plot_regional_strategic_results(strategic_plans, df_capacity, regional_targets[region], region)
        save_regional_strategic_plans(strategic_plans, region)

        # Visualization calls
        plot_all_models_regional(
            strategic_plans=strategic_plans,
            region=region,
            regional_targets=regional_targets
        )
        
        plot_median_models_regional(
            strategic_plans=strategic_plans,
            region=region,
            regional_targets=regional_targets
        )
        
        plot_comparison_regional(
            strategic_plans=strategic_plans,
            yearly_forecasts=yearly_forecasts,
            region=region,
            regional_targets=regional_targets,
            historical_base=regional_capacities[region]
        )

    logging.info("\nAll regional analyses completed.")

if __name__ == "__main__":
    main()
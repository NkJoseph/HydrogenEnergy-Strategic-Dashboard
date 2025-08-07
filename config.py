"""
Configuration file for hydrogen forecasting project
"""
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
SEED = 42
SEQUENCE_LENGTH = 12
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50  # Reduced for proof of concept
PATIENCE = 10    # Early stopping patience

# Data parameters
DATA_SPLIT = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

# Model parameters
MODEL_CONFIGS = {
    'transformer': {
        'd_model': 256,
        'n_head': 8,
        'num_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.3
    },
    'autoformer': {
        'd_model': 128,
        'n_heads': 4,
        'd_ff': 256,
        'e_layers': 2,
        'moving_avg_kernel': 5,
        'dropout': 0.1
    },
    'informer': {
        'd_model': 256,
        'n_heads': 8,
        'ff_dim': 1024,
        'n_layers': 4,
        'dropout': 0.1
    },
    'nbeats': {
        'hidden_size': 128,
        'n_blocks': 2,
        'n_layers': 2,
        'dropout': 0.3
    },
    'tft': {
        'hidden_size': 128,
        'n_heads': 4,
        'ff_dim': 256,
        'dropout': 0.1
    },
    'logsparsetransformer': {
        'd_model': 256,
        'n_heads': 8,
        'ff_dim': 1024,
        'n_layers': 4,
        'dropout': 0.1
    },
    'neuralode': {
        'hidden_dim': 128,
        'output_dim': 1
    }
}

# Regional settings
REGIONS = [
    "Africa", "Canada", "USA", "India", "China", "Europe",
    "East & Southeast Asia", "South Asia", "Central Asia",
    "Middle East", "Latin America", "Australia", "Oceania", "Others"
]

# Paths
MODEL_SAVE_PATH = "pretrained-models"
DATA_PATH = "dataset"
PROCESSED_DATA_PATH = "processed_data" 
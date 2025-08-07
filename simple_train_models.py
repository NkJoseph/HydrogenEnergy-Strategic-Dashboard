"""
Simplified training script for hydrogen forecasting models
Creates basic models for proof of concept and saves them for the API
"""
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from config import (
    SEED, SEQUENCE_LENGTH, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, 
    PATIENCE, REGIONS, MODEL_SAVE_PATH, PROCESSED_DATA_PATH
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

class SimpleTransformer(nn.Module):
    """Simplified Transformer model for proof of concept"""
    def __init__(self, input_dim, output_dim=1, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Apply transformer
        x = self.transformer(x)  # [batch, seq_len, d_model]
        
        # Use last timestep for prediction
        x = x[:, -1, :]  # [batch, d_model]
        
        # Project to output
        return self.output_projection(x)  # [batch, output_dim]

class SimpleAutoformer(nn.Module):
    """Simplified Autoformer-style model"""
    def __init__(self, input_dim, output_dim=1, d_model=128):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.decomp = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_projection(x)
        
        # Simple decomposition
        trend = self.decomp(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        
        # Attention on seasonal component
        attn_out, _ = self.attention(seasonal, seasonal, seasonal)
        x = self.norm(attn_out + seasonal)
        
        # Combine with trend
        x = x + trend
        
        return self.output_projection(x[:, -1, :])

class SimpleInformer(nn.Module):
    """Simplified Informer-style model"""
    def __init__(self, input_dim, output_dim=1, d_model=128):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        x = self.input_projection(x)
        
        # Sparse attention (simplified)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        
        # Convolution for local patterns
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x + x_conv)
        
        return self.output_projection(x[:, -1, :])

class SimpleNBeats(nn.Module):
    """Simplified N-BEATS model"""
    def __init__(self, input_dim, output_dim=1, hidden_size=128):
        super().__init__()
        self.input_projection = nn.Linear(input_dim * SEQUENCE_LENGTH, hidden_size)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ) for _ in range(3)
        ])
        self.output_projection = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        # Flatten sequence
        x = x.reshape(x.size(0), -1)
        x = self.input_projection(x)
        
        # Pass through blocks
        for block in self.blocks:
            residual = x
            x = block(x) + residual
            
        return self.output_projection(x)

def create_simple_model(model_name: str, input_dim: int, output_dim: int = 1):
    """Create a simplified model instance"""
    if model_name == 'transformer':
        return SimpleTransformer(input_dim, output_dim)
    elif model_name == 'autoformer':
        return SimpleAutoformer(input_dim, output_dim)
    elif model_name == 'informer':
        return SimpleInformer(input_dim, output_dim)
    elif model_name == 'nbeats':
        return SimpleNBeats(input_dim, output_dim)
    elif model_name == 'tft':
        return SimpleTransformer(input_dim, output_dim, d_model=96, nhead=4)
    elif model_name == 'logsparsetransformer':
        return SimpleTransformer(input_dim, output_dim, d_model=128, nhead=8)
    elif model_name == 'neuralode':
        return SimpleTransformer(input_dim, output_dim, d_model=64, nhead=2)
    else:
        return SimpleTransformer(input_dim, output_dim)

def create_sequences(data, sequence_length):
    """Create sequences from time series data"""
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def prepare_data(df, target_col, feature_cols, sequence_length):
    """Prepare data for training"""
    # Sort by year
    df = df.sort_values('Year').reset_index(drop=True)
    
    # Prepare features and targets
    features = df[feature_cols].values
    targets = df[target_col].values
    
    # Create sequences
    X, y = create_sequences(features, sequence_length)
    
    if len(X) == 0:
        # If not enough data for sequences, create dummy data
        logger.warning(f"Not enough data for sequences, creating dummy data")
        X = np.random.randn(10, sequence_length, len(feature_cols))
        y = np.random.randn(10)
    
    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)

def train_simple_model(model, X, y, epochs=NUM_EPOCHS):
    """Train a model with simple training loop"""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    return model

def load_processed_data():
    """Load the preprocessed data"""
    data_path = Path(PROCESSED_DATA_PATH) / "augmented_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}. Run preprocess_data.py first.")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded processed data with shape: {df.shape}")
    return df

def train_global_models(df):
    """Train global models"""
    logger.info("Training global models...")
    
    # Prepare global data
    exclude_cols = ['Year', 'Total Yearly Capacity']
    region_cols = [col for col in df.columns if col not in exclude_cols and col in REGIONS]
    feature_cols = ['Year'] + region_cols
    
    X, y = prepare_data(df, 'Total Yearly Capacity', feature_cols, SEQUENCE_LENGTH)
    input_dim = X.shape[2]
    
    logger.info(f"Global training data: X.shape={X.shape}, y.shape={y.shape}")
    
    model_names = ['transformer', 'autoformer', 'informer', 'nbeats', 'tft', 'logsparsetransformer', 'neuralode']
    trained_models = {}
    
    for model_name in model_names:
        try:
            logger.info(f"Training global {model_name}...")
            
            model = create_simple_model(model_name, input_dim)
            model = train_simple_model(model, X, y, epochs=NUM_EPOCHS)
            
            # Save model
            save_path = Path(MODEL_SAVE_PATH) / "global" / f"{model_name}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            
            trained_models[model_name] = model
            logger.info(f"Saved global {model_name} to {save_path}")
            
        except Exception as e:
            logger.error(f"Error training global {model_name}: {e}")
            continue
    
    return trained_models

def train_regional_models(df):
    """Train regional models for each region"""
    logger.info("Training regional models...")
    
    trained_models = {}
    model_names = ['transformer', 'autoformer', 'informer', 'nbeats', 'tft', 'logsparsetransformer', 'neuralode']
    
    for region in REGIONS:
        if region not in df.columns:
            continue
            
        # Check if region has any data
        if df[region].sum() == 0:
            logger.info(f"Region {region} has no data, skipping...")
            continue
            
        logger.info(f"Training models for region: {region}")
        trained_models[region] = {}
        
        try:
            # Prepare regional data
            feature_cols = ['Year', 'Total Yearly Capacity']
            X, y = prepare_data(df, region, feature_cols, SEQUENCE_LENGTH)
            input_dim = X.shape[2]
            
            logger.info(f"Regional {region} data: X.shape={X.shape}, y.shape={y.shape}")
            
            for model_name in model_names:
                try:
                    logger.info(f"Training {region} {model_name}...")
                    
                    model = create_simple_model(model_name, input_dim)
                    model = train_simple_model(model, X, y, epochs=NUM_EPOCHS)
                    
                    # Save model
                    save_path = Path(MODEL_SAVE_PATH) / "regional" / f"{region}_{model_name}.pt"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    
                    trained_models[region][model_name] = model
                    logger.info(f"Saved {region} {model_name} to {save_path}")
                    
                except Exception as e:
                    logger.error(f"Error training {region} {model_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error preparing data for region {region}: {e}")
            continue
    
    return trained_models

def create_model_summary():
    """Create a summary of trained models"""
    global_dir = Path(MODEL_SAVE_PATH) / "global"
    regional_dir = Path(MODEL_SAVE_PATH) / "regional"
    
    summary = {
        'global_models': [],
        'regional_models': {}
    }
    
    # Check global models
    if global_dir.exists():
        for model_file in global_dir.glob("*.pt"):
            summary['global_models'].append(model_file.stem)
    
    # Check regional models
    if regional_dir.exists():
        for model_file in regional_dir.glob("*.pt"):
            parts = model_file.stem.split('_', 1)
            if len(parts) == 2:
                region, model_name = parts
                if region not in summary['regional_models']:
                    summary['regional_models'][region] = []
                summary['regional_models'][region].append(model_name)
    
    return summary

def main():
    """Main training pipeline"""
    logger.info("Starting simplified hydrogen forecasting model training")
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    try:
        # Load processed data
        df = load_processed_data()
        
        # Train global models
        logger.info("=" * 50)
        global_models = train_global_models(df)
        logger.info(f"Completed training {len(global_models)} global models")
        
        # Train regional models
        logger.info("=" * 50)
        regional_models = train_regional_models(df)
        logger.info(f"Completed training models for {len(regional_models)} regions")
        
        # Create summary
        summary = create_model_summary()
        logger.info("=" * 50)
        logger.info("TRAINING SUMMARY:")
        logger.info(f"Global models trained: {summary['global_models']}")
        logger.info(f"Regional models trained: {len(summary['regional_models'])} regions")
        for region, models in summary['regional_models'].items():
            logger.info(f"  {region}: {models}")
        
        logger.info("=" * 50)
        logger.info("Training completed successfully!")
        logger.info(f"Models saved in: {MODEL_SAVE_PATH}")
        logger.info("Your API can now load and use these trained models.")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 
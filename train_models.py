"""
Main training script for hydrogen forecasting models
Trains both global and regional models and saves them for the API
"""
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Import from our analysis scripts (handling spaces in filenames)
import importlib.util
import sys

# Import global analysis
spec = importlib.util.spec_from_file_location("global_analysis", "global analysis.py")
global_analysis = importlib.util.module_from_spec(spec)
sys.modules["global_analysis"] = global_analysis
spec.loader.exec_module(global_analysis)

# Import specific classes and functions
from global_analysis import (
    Transformer, Autoformer, Informer, NBeats, TemporalFusionTransformer,
    LogSparseTransformer, NeuralODE, train_model, regional_train_model,
    prepare_global_dataset, prepare_regional_dataset, build_dataloaders,
    configure_optimizer
)
from config import (
    SEED, SEQUENCE_LENGTH, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, 
    PATIENCE, MODEL_CONFIGS, REGIONS, MODEL_SAVE_PATH, PROCESSED_DATA_PATH
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_processed_data():
    """Load the preprocessed data"""
    data_path = Path(PROCESSED_DATA_PATH) / "augmented_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}. Run preprocess_data.py first.")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded processed data with shape: {df.shape}")
    logger.info(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    return df

def create_model(model_name: str, input_dim: int, output_dim: int, device: str = 'cpu'):
    """Create a model instance based on the model name"""
    config = MODEL_CONFIGS.get(model_name, {})
    
    if model_name == 'transformer':
        model = Transformer(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=config.get('d_model', 256),
            n_head=config.get('n_head', 8),
            num_layers=config.get('num_layers', 6),
            dim_feedforward=config.get('dim_feedforward', 1024),
            dropout=config.get('dropout', 0.3)
        )
    elif model_name == 'autoformer':
        model = Autoformer(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 4),
            d_ff=config.get('d_ff', 256),
            e_layers=config.get('e_layers', 2),
            moving_avg_kernel=config.get('moving_avg_kernel', 5),
            dropout=config.get('dropout', 0.1),
            device=device
        )
    elif model_name == 'informer':
        model = Informer(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=config.get('d_model', 256),
            n_heads=config.get('n_heads', 8),
            ff_dim=config.get('ff_dim', 1024),
            n_layers=config.get('n_layers', 4),
            dropout=config.get('dropout', 0.1),
            device=device
        )
    elif model_name == 'nbeats':
        model = NBeats(
            input_size=SEQUENCE_LENGTH,
            hidden_size=config.get('hidden_size', 128),
            n_blocks=config.get('n_blocks', 2),
            n_layers=config.get('n_layers', 2),
            theta_size={'trend': 3, 'seasonality': 4, 'generic': 8},
            stack_types=['trend', 'seasonality', 'generic'],
            backcast_length=SEQUENCE_LENGTH,
            forecast_length=SEQUENCE_LENGTH,
            num_features=input_dim,
            device=device,
            dropout=config.get('dropout', 0.3)
        )
    elif model_name == 'tft':
        # For TFT, we need to adapt the interface
        model = TemporalFusionTransformer(
            input_sizes={'static': 0, 'known': input_dim, 'observed': 0},
            output_size=output_dim,
            hidden_size=config.get('hidden_size', 128),
            n_heads=config.get('n_heads', 4),
            ff_dim=config.get('ff_dim', 256),
            dropout=config.get('dropout', 0.1),
            seq_len=SEQUENCE_LENGTH,
            num_static_vars=0,
            num_known_vars=input_dim,
            num_observed_vars=0,
            device=device
        )
    elif model_name == 'logsparsetransformer':
        model = LogSparseTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=config.get('d_model', 256),
            n_heads=config.get('n_heads', 8),
            ff_dim=config.get('ff_dim', 1024),
            n_layers=config.get('n_layers', 4),
            dropout=config.get('dropout', 0.1),
            device=device
        )
    elif model_name == 'neuralode':
        model = NeuralODE(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 128),
            output_dim=output_dim,
            device=device
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model

def train_global_models(df: pd.DataFrame):
    """Train global models"""
    logger.info("Training global models...")
    
    # Prepare global dataset
    global_dataset = prepare_global_dataset(df, SEQUENCE_LENGTH)
    train_loader, val_loader, test_loader = build_dataloaders(global_dataset, SEED, BATCH_SIZE)
    
    # Get input/output dimensions
    sample_batch = next(iter(train_loader))
    inputs_sample, labels_sample = sample_batch
    input_dim = inputs_sample.shape[2]
    output_dim = labels_sample.shape[1] if labels_sample.dim() > 1 else 1
    
    logger.info(f"Global model input_dim: {input_dim}, output_dim: {output_dim}")
    
    # Train each model
    model_names = ['transformer', 'autoformer', 'informer', 'nbeats', 'tft', 'logsparsetransformer', 'neuralode']
    trained_models = {}
    
    for model_name in model_names:
        try:
            logger.info(f"Training global {model_name}...")
            
            # Create model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = create_model(model_name, input_dim, output_dim, device)
            
            # Setup training
            criterion = nn.SmoothL1Loss(beta=0.1)
            optimizer, scheduler = configure_optimizer(model, LEARNING_RATE)
            
            # Train model
            model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=NUM_EPOCHS, patience=PATIENCE
            )
            
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

def train_regional_models(df: pd.DataFrame):
    """Train regional models for each region"""
    logger.info("Training regional models...")
    
    trained_models = {}
    model_names = ['transformer', 'autoformer', 'informer', 'nbeats', 'tft', 'logsparsetransformer', 'neuralode']
    
    for region in REGIONS:
        if region not in df.columns:
            logger.warning(f"Region {region} not found in dataset, skipping...")
            continue
            
        # Check if region has any data
        if df[region].sum() == 0:
            logger.info(f"Region {region} has no data, skipping...")
            continue
            
        logger.info(f"Training models for region: {region}")
        trained_models[region] = {}
        
        try:
            # Prepare regional dataset
            regional_dataset = prepare_regional_dataset(df, region, SEQUENCE_LENGTH)
            train_loader, val_loader, test_loader = build_dataloaders(regional_dataset, SEED, BATCH_SIZE)
            
            # Get input/output dimensions
            sample_batch = next(iter(train_loader))
            inputs_sample, labels_sample = sample_batch
            input_dim = inputs_sample.shape[2]  # Should be 2: Year + Total Yearly Capacity
            output_dim = labels_sample.shape[1] if labels_sample.dim() > 1 else 1
            
            logger.info(f"Regional model for {region}: input_dim={input_dim}, output_dim={output_dim}")
            
            for model_name in model_names:
                try:
                    logger.info(f"Training {region} {model_name}...")
                    
                    # Create model
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model = create_model(model_name, input_dim, output_dim, device)
                    
                    # Setup training
                    criterion = nn.SmoothL1Loss(beta=0.1)
                    optimizer, scheduler = configure_optimizer(model, LEARNING_RATE)
                    
                    # Train model
                    model, train_losses, val_losses = regional_train_model(
                        model, train_loader, val_loader, criterion, optimizer, scheduler,
                        num_epochs=NUM_EPOCHS, patience=PATIENCE
                    )
                    
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
    logger.info("Starting hydrogen forecasting model training")
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
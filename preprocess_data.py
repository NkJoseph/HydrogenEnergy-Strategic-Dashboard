"""
Data preprocessing pipeline for hydrogen forecasting project
Uses the final aggregated dataset which is already in the correct format
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from config import REGIONS, DATA_PATH, PROCESSED_DATA_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_final_dataset(filepath: str) -> pd.DataFrame:
    """Load the final aggregated dataset with '$' delimiter"""
    try:
        df = pd.read_csv(filepath, delimiter='$')
        logger.info(f"Loaded final dataset with shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def sample_dataset(df: pd.DataFrame, sample_fraction: float = 0.2) -> pd.DataFrame:
    """Sample the dataset for proof of concept"""
    if sample_fraction >= 1.0:
        logger.info("Using full dataset")
        return df
    
    # Sample rows (years)
    sampled_df = df.sample(frac=sample_fraction, random_state=42).sort_values('Year')
    logger.info(f"Sampled {len(sampled_df)} rows from {len(df)} total rows")
    logger.info(f"Sampled year range: {sampled_df['Year'].min()} - {sampled_df['Year'].max()}")
    
    return sampled_df

def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the dataset"""
    logger.info("Validating dataset format")
    
    # Check required columns
    required_cols = ['Year', 'Total Yearly Capacity'] + REGIONS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Dataset missing columns: {missing_cols}")
    
    # Remove any rows with missing years
    df = df.dropna(subset=['Year'])
    
    # Fill missing values with 0 for region columns
    region_cols = [col for col in REGIONS if col in df.columns]
    df[region_cols] = df[region_cols].fillna(0)
    
    # Recalculate Total Yearly Capacity to ensure consistency
    df['Total Yearly Capacity'] = df[region_cols].sum(axis=1)
    
    # Sort by year
    df = df.sort_values('Year').reset_index(drop=True)
    
    logger.info(f"Dataset validation completed. Final shape: {df.shape}")
    return df

def extend_dataset_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Extend dataset with projected years if needed for training"""
    max_year = int(df['Year'].max())
    min_year = int(df['Year'].min())
    
    logger.info(f"Current year range: {min_year} - {max_year}")
    
    # For simplicity, just return the original dataset
    # The existing data should be sufficient for training
    logger.info("Using existing dataset without extension")
    
    return df

def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save processed data to CSV"""
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to: {output_path}")
    
    # Print summary statistics
    logger.info("\nDataset Summary:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    logger.info(f"Total capacity range: {df['Total Yearly Capacity'].min():.2f} - {df['Total Yearly Capacity'].max():.2f}")
    
    # Show non-zero regions
    region_cols = [col for col in REGIONS if col in df.columns]
    for region in region_cols:
        total_capacity = df[region].sum()
        if total_capacity > 0:
            logger.info(f"{region}: {total_capacity:.2f} kt total")

def main():
    """Main preprocessing pipeline"""
    logger.info("Starting data preprocessing pipeline for final dataset")
    
    # Load the final dataset
    dataset_path = Path(DATA_PATH) / "final_dataset_Capacity_kt_H2_y.csv"
    df = load_final_dataset(str(dataset_path))
    
    # Sample the dataset (20% for proof of concept)
    df_sampled = sample_dataset(df, sample_fraction=0.2)
    
    # Validate and clean the dataset
    df_processed = validate_dataset(df_sampled)
    
    # Extend dataset if needed for better training
    df_final = extend_dataset_for_training(df_processed)
    
    # Create output directory
    output_dir = Path(PROCESSED_DATA_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    output_path = output_dir / "augmented_data.csv"
    save_processed_data(df_final, str(output_path))
    
    logger.info("Data preprocessing completed successfully")
    logger.info(f"Dataset ready for training!")

if __name__ == "__main__":
    main() 
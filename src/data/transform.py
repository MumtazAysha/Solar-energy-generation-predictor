import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.common.config import load_config
from src.common.io_utils import read_parquet, write_parquet

logger = logging.getLogger(__name__)


def add_temporal_features(df):
    """Add basic temporal features"""
    logger.info("Adding temporal features...")
    
    # Basic temporal features
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['minute_of_day'] = df['hour'] * 60 + df['minute']
    df['season'] = (df['Month'] % 12 // 3 + 1)  # 1=Winter, 2=Spring, 3=Summer, 4=Fall

    logger.info(f"  ✅ Added: day_of_year, day_of_week, is_weekend, minute_of_day, season")
    
    return df

def add_cyclical_features(df):
    """Add cyclical encoding for temporal features (sin/cos)"""
    logger.info("Adding cyclical features...")
    
    # Hour encoding (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of year encoding (1-365/366)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    logger.info(f"  ✅ Added: hour_sin, hour_cos, doy_sin, doy_cos")
    
    return df

def add_solar_features(df):
    """Add approximate solar elevation angle"""
    logger.info("Adding solar features...")
    
    # Simplified solar elevation angle approximation
    # This is a rough approximation for Sri Lanka (latitude ~7°N)
    # More accurate calculation would require solar position library
    
    # Hour angle: how far the sun is from solar noon
    solar_noon = 12.0
    hour_angle = (df['hour'] + df['minute']/60 - solar_noon) * 15  # degrees
    
    # Day angle: variation through the year
    day_angle = 2 * np.pi * (df['day_of_year'] - 1) / 365
    
    # Solar declination (approximate)
    declination = 23.45 * np.sin(day_angle)
    
    # For Sri Lanka (latitude ~7°N)
    latitude = 7.0
    
    # Solar elevation angle (simplified)
    # elevation = arcsin(sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(hour_angle))
    elevation = np.arcsin(
        np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
        np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
    )
    
    df['solar_elev_approx'] = np.degrees(elevation)
    
    # Clip to reasonable range (sun below horizon = 0)
    df['solar_elev_approx'] = df['solar_elev_approx'].clip(lower=0)
    
    logger.info(f"  ✅ Added: solar_elev_approx")
    logger.info(f"  Solar elevation range: {df['solar_elev_approx'].min():.1f}° to {df['solar_elev_approx'].max():.1f}°")
    
    return df

def transform_data():
    """Main transformation function"""
    cfg = load_config()
    
    silver_path = Path(cfg.data_paths['silver'])
    gold_path = Path(cfg.data_paths['gold'])
    gold_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("STEP 3: TRANSFORMATION (Silver → Gold)")
    logger.info("="*60)

    # Read Silver data
    silver_file = silver_path / "silver_all_years.parquet"
    if not silver_file.exists():
        logger.error(f"❌ Silver file not found: {silver_file}")
        logger.info("Please run validation first: python -m src.data.validate")
        return
    
    df = read_parquet(silver_file)
    logger.info(f"Loaded {len(df):,} records from Silver layer\n")
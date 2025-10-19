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
import pandas as pd
import numpy as np
import logging 
import joblib
from pathlib import Path
from sklearn.ensemble import RandmForeestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from src.common.config import load_config
from src.common.io_utils import read_parquet

logger = logging.getLogger(__name__)

def load_and_prepare_data(cfg):
   """Load feature-engineered data and prepare for training"""
   logger.info("Loading feature data...")
    
   gold_path = Path(cfg.data_paths['gold'])
   features_file = gold_path / "gold_features_all_years.parquet"
    
   if not features_file.exists():
        logger.error(f"❌ Features file not found: {features_file}")
        logger.info("Please run feature engineering first: python -m src.features.builder")
        return None, None, None

   df = read_parquet(features_file)
   logger.info(f"Loaded {len(df):,} records")

   #Identify feature columns
   exclude_cols = ['datetime', 'Date', 'Year', 'Month', 'Day', 'target_kw', 'District']
   feature_cols = [c for c in df.columns if c not in exclude_cols]

   logger.info(f"Feature columns: {len(feature_cols)}")

   return df, feature_cols, exclude_cols





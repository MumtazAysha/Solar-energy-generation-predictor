"""
Data validatin Module
Validates Bronze data fr quality issues before transfrmatioon
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.common.config import load_config
from src.common.io_utils import read_parquet, write_parquet

logger = logging.getLogger(__name__)

def cheeck_missing_values(df):
    """Check for missing values in critical columns"""
    logger.info("Checking for missing values...")

    missing = df.isnull().sum()
    missing_pct = (missing/len(df)) * 100
    
    critical_cols = ['Datetime', 'Date', 'Year', 'Month', 'District', 'Generation_kw']

    issues = []
    for col in critical_cols:
        if col in df.columns:
            if missing[col] > 0:
                issues.append(f"  ⚠️ {col}: {missing[col]:,} missing ({missing_pct[col]:.2f}%)")
                logger.warning(f"  Missing {col}: {missing[col]:,} rows ({missing_pct[col]:.2f}%)")
    
    if not issues:
        logger.info("  ✅ No missing values in critical columns")
    
    return missing


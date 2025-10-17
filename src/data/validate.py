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

def check_date_ranges(df):
    """Validate date ranges are reasonable"""
    logger.info("Checking date ranges...")
    
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    logger.info(f"  Date range: {min_date.date()} to {max_date.date()}")
    
    # Check for future dates
    from datetime import datetime
    today = datetime.now().date()
    future_dates = df[df['Date'].dt.date > today]
    
    if len(future_dates) > 0:
        logger.warning(f"  ⚠️ Found {len(future_dates):,} records with future dates")
    else:
        logger.info(f"  ✅ No future dates found")
    
    # Check year distribution
    year_counts = df['Year'].value_counts().sort_index()
    logger.info(f"  Records per year:")
    for year, count in year_counts.items():
        logger.info(f"    {year}: {count:,} records")
    
    return min_date, max_date

"""I/O utilities for reading and writing data"""

import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def read_parquet(path):
    """Read parquet file and return DataFrame"""
    logger.info(f"Reading {path}" )
    df = pd.read_parquet(path)
    logger.info(f"LOaded {len(df)} rows")
    return df

def write_parquet(df, path):
    """Write DataFrame to parquet file"""
    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {path}")
    df.to_parquet(path, index=False, compression='snappy')
    logger.info(f" Saved {len(df)} rows to {path}")

def read_csv(path):
    """Read CSV file and return DataFrame"""
    logger.info(f"Reading {path}")
    df = pd.read_csv(path)
    logger.info(f"loaded {len(df)} rows to {path}")
    return df

def write_csv(df,path):
    """Write DataFrame to csv file"""
    path(path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {path}")
    df.to_csv(path, index=False)
    logger.info(f" saved {len(df)} rows to {path}")

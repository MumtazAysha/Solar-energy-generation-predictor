"""I/O utilities for reading and writing data"""

import pandas as pd
import logging
from pathlib import path

logger = logging.getLogger(__name__)

def read_parquet(path):
    """Read parquet file and return DataFrame"""
    logger.info(f"Reading {path}" )
    df = pd.read_parquet(path)
    logger.infoo(f"LOaded {Len(df)} rows")
    return df


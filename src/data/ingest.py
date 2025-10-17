"""
Data Ingestion Module
Reads raw Excel files and converts to Bronze layer (long-format parquet)
"""

import pandas as pd
import logging
from pathlib import Path
from src.common.config import load_config
from src.common.io_utils import write_paarquet

logger = logging.getLogger(__name__)

def read_wide_excel(file_path):
    """Read wide-format file with time inetervals as columns.
    
    Expected format:
    - Columns: Date, Year, Month, Day, District, Zone, 08.00, 08.05,..., 17.00
    
    Args:
      file_path(str): Path to Excel file
        
    Returns: 
      pd.DataFrame: Long-foormat dataframe with datetime column
    """
    logger.info(f"Reading {file_path.name}")

     # Read Excel
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip() for c in df.columns]
    
    # Identify metadata and time columns
    meta_cols = ['Date', 'Year', 'Month', 'Day', 'District', 'Zone']
    time_cols = [c for c in df.columns if c not in meta_cols]
    
    logger.info(f"  Found {len(time_cols)} time interval columns")
    
    # Melt to long format
    df_long = df.melt(
        id_vars=meta_cols,
        value_vars=time_cols,
        var_name='time_decimal',
        value_name='generation_kw'
    )
    
    # Convert Date to datetime
    df_long['Date'] = pd.to_datetime(df_long['Date'])
    
    # Parse time from time_decimal (e.g., "08.05" → hour=8, minute=5)
    df_long['time_decimal'] = df_long['time_decimal'].astype(str)
    df_long['hour'] = df_long['time_decimal'].str.split('.').str[0].astype(int)
    df_long['minute'] = df_long['time_decimal'].str.split('.').str[1].astype(int)
    
    # Create full datetime column
    df_long['datetime'] = pd.to_datetime(
        df_long['Date'].dt.strftime('%Y-%m-%d') + ' ' + 
        df_long['hour'].astype(str).str.zfill(2) + ':' + 
        df_long['minute'].astype(str).str.zfill(2)
    )
    
    # Select and reorder columns
    df_long = df_long[[
        'datetime', 'Date', 'Year', 'Month', 'Day', 
        'District', 'Zone', 'time_decimal', 'generation_kw'
    ]]
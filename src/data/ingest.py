"""
Data Ingestion Module
Reads raw Excel files and converts to Bronze layer (long-format parquet)
"""

import pandas as pd
import logging
from pathlib import Path
from src.common.config import load_config
from src.common.io_utils import write_paarquet, write_parquet

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

    # Sort by datetime
    df_long = df_long.sort_values(['District', 'datetime']).reset_index(drop=True)
    
    logger.info(f"  ✅ Converted to {len(df_long)} rows")
    
    return df_long

def ingest_data():
    """
    Main ingestion function: reads all raw Excel files and creates Bronze parquet.
    """
    cfg = load_config()
    
    raw_path = Path(cfg.data_paths['raw'])
    bronze_path = Path(cfg.data_paths['bronze'])
    
    # Create bronze directory if it doesn't exist
    bronze_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("STEP 1: INGESTION (Raw → Bronze)")
    logger.info("="*60)
    
    # Find all Excel files in raw directory
    excel_files = sorted(list(raw_path.glob('*.xlsx')) + list(raw_path.glob('*.xls')))
    
    if not excel_files:
        logger.error(f"❌ No Excel files found in {raw_path}")
        logger.info(f"Please add Excel files to {raw_path.absolute()}")
        return
    
    logger.info(f"Found {len(excel_files)} Excel files")

    # Process each file
    all_data = []
    success_count = 0
    for i, file_path in enumerate(excel_files, 1):
        try:
            logger.info(f"[{i}/{len(excel_files)}] Processing {file_path.name}...")
            df = read_wide_excel(file_path)
            all_data.append(df)
            success_count += 1
        except Exception as e:
            logger.error(f"❌ Failed to process {file_path.name}: {str(e)}")
            continue
    
    if not all_data:
        logger.error("❌ No data was successfully ingested")
        return
    
    # Combine all dataframes
    logger.info(f"\nCombining data from {success_count} files...")
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Save to Bronze as parquet
    output_file = bronze_path / "bronze_all_years.parquet"
    write_parquet(df_combined, output_file)
    
    logger.info("\n" + "="*60)
    logger.info("✅ INGESTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total records: {len(df_combined):,}")
    logger.info(f"Districts: {df_combined['District'].nunique()}")
    logger.info(f"Years: {sorted(df_combined['Year'].unique())}")
    logger.info(f"Date range: {df_combined['Date'].min()} to {df_combined['Date'].max()}")
    logger.info(f"Output: {output_file}")
    logger.info("="*60)


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    ingest_data()




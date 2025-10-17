"""
Data Ingestion Module
Reads raw Excel files and converts to Bronze layer (long-format parquet)
"""

import pandas as pd
import logging
from pathlib import Path
from src.common.config import load_config
from src.common.io_utils import write_parquet

logger = logging.getLogger(__name__)


def read_wide_excel(file_path):
    """
    Read wide-format Excel file with Date + numeric time columns.
    
    Structure:
    - Column 1: Date
    - Columns 2-N: Time in decimal format (8, 8.05, 8.1, ..., 17)
    
    Args:
        file_path (Path): Path to Excel file
        
    Returns:
        pd.DataFrame: Long-format dataframe with datetime column
    """
    logger.info(f"Reading {file_path.name}")
    
    # Read Excel
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip() for c in df.columns]
    
    # First column is Date, rest are time intervals
    date_col = df.columns[0]  # Should be "Date"
    time_cols = df.columns[1:]  # All numeric time columns
    
    logger.info(f"  Found {len(time_cols)} time interval columns")
    
    # Melt to long format
    df_long = df.melt(
        id_vars=[date_col],
        value_vars=time_cols,
        var_name='time_decimal',
        value_name='generation_kw'
    )
    
    # Rename to standard 'Date' column
    df_long = df_long.rename(columns={date_col: 'Date'})
    
    # Convert Date to datetime
    df_long['Date'] = pd.to_datetime(df_long['Date'])
    
    # Extract Year, Month, Day
    df_long['Year'] = df_long['Date'].dt.year
    df_long['Month'] = df_long['Date'].dt.month
    df_long['Day'] = df_long['Date'].dt.day
    
    # Extract district from filename
    # Pattern: "Annual Generation data in Ampara 2018.xlsx"
    filename_parts = file_path.stem.split()
    if 'in' in filename_parts:
        in_index = filename_parts.index('in')
        if in_index + 1 < len(filename_parts):
            district = filename_parts[in_index + 1]
        else:
            district = "Unknown"
    else:
        district = "Unknown"
    
    df_long['District'] = district
    df_long['Zone'] = ""  # Empty zone
    
    # Convert time_decimal to hour and minute
    df_long['time_decimal_float'] = pd.to_numeric(df_long['time_decimal'], errors='coerce')
    df_long['hour'] = df_long['time_decimal_float'].astype(int)
    df_long['minute'] = ((df_long['time_decimal_float'] - df_long['hour']) * 100).round().astype(int)
    
    # Create full datetime column
    df_long['datetime'] = df_long['Date'] + pd.to_timedelta(
        df_long['hour'].astype(str) + ' hours'
    ) + pd.to_timedelta(
        df_long['minute'].astype(str) + ' minutes'
    )
    
    # Select and reorder columns
    df_long = df_long[[
        'datetime', 'Date', 'Year', 'Month', 'Day', 
        'District', 'Zone', 'time_decimal', 'generation_kw'
    ]]
    
    # Sort by datetime
    df_long = df_long.sort_values(['District', 'datetime']).reset_index(drop=True)
    
    logger.info(f"  ✅ Converted to {len(df_long):,} rows")
    
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
    
    logger.info(f"Found {len(excel_files)} Excel files\n")
    
    # Process each file
    all_data = []
    success_count = 0
    failed_files = []
    
    for i, file_path in enumerate(excel_files, 1):
        try:
            logger.info(f"[{i}/{len(excel_files)}] Processing...")
            df = read_wide_excel(file_path)
            all_data.append(df)
            success_count += 1
        except Exception as e:
            logger.error(f"❌ Failed: {file_path.name}")
            logger.error(f"   Error: {str(e)}")
            failed_files.append(file_path.name)
            continue
    
    if not all_data:
        logger.error("\n❌ No data was successfully ingested")
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
    logger.info(f"Successful: {success_count}/{len(excel_files)} files")
    logger.info(f"Total records: {len(df_combined):,}")
    logger.info(f"Districts: {sorted(df_combined['District'].unique())}")
    logger.info(f"Years: {sorted(df_combined['Year'].unique())}")
    logger.info(f"Date range: {df_combined['Date'].min().date()} to {df_combined['Date'].max().date()}")
    logger.info(f"Output: {output_file}")
    
    if failed_files:
        logger.warning(f"\n⚠️ Failed files: {len(failed_files)}")
    logger.info("="*60)


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    ingest_data()





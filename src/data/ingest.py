"""
Data Ingestion Module
Reads raw Excel files and converts to Bronze layer (long-format parquet)
Handles both single-sheet (2022) and multi-sheet (other years) formats
Automatically skips README or info sheets
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
    Handles both single-sheet and multi-sheet Excel files.
    Skips README/info sheets automatically.
    
    Args:
        file_path (Path): Path to Excel file
        
    Returns:
        pd.DataFrame: Long-format dataframe with datetime column
    """
    logger.info(f"Reading {file_path.name}")
    
    # Extract district from filename
    filename_parts = file_path.stem.split()
    if 'in' in filename_parts:
        in_index = filename_parts.index('in')
        if in_index + 1 < len(filename_parts):
            district = filename_parts[in_index + 1]
        else:
            district = "Unknown"
    else:
        district = "Unknown"
    
    # Read all sheets from the Excel file
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    
    # Filter out README or info sheets (case-insensitive)
    data_sheets = [
        sheet for sheet in sheet_names 
        if 'readme' not in sheet.lower() and 'info' not in sheet.lower()
    ]
    
    logger.info(f"  Found {len(sheet_names)} total sheets, processing {len(data_sheets)} data sheets")
    
    all_sheets_data = []
    
    for sheet_name in data_sheets:
        try:
            # Read each sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df.columns = [str(c).strip() for c in df.columns]
            
            # Skip if sheet is empty or doesn't have expected structure
            if len(df.columns) < 2:
                logger.warning(f"  ⚠️ Skipping sheet '{sheet_name}' - insufficient columns")
                continue
            
            # First column is Date, rest are time intervals
            date_col = df.columns[0]
            time_cols = df.columns[1:]
            
            # Melt to long format
            df_long = df.melt(
                id_vars=[date_col],
                value_vars=time_cols,
                var_name='time_decimal',
                value_name='generation_kw'
            )
            
            # Rename to standard 'Date' column
            df_long = df_long.rename(columns={date_col: 'Date'})
            
            all_sheets_data.append(df_long)
            
        except Exception as e:
            logger.warning(f"  ⚠️ Skipping sheet '{sheet_name}': {str(e)}")
            continue
    
    if not all_sheets_data:
        logger.error(f"  ❌ No valid data sheets found in {file_path.name}")
        return None
    
    # Combine all sheets
    df_combined = pd.concat(all_sheets_data, ignore_index=True)
    
    # Convert Date to datetime
    df_combined['Date'] = pd.to_datetime(df_combined['Date'], errors='coerce')
    
    # Drop rows with invalid dates
    df_combined = df_combined.dropna(subset=['Date'])
    
    # Extract Year, Month, Day
    df_combined['Year'] = df_combined['Date'].dt.year
    df_combined['Month'] = df_combined['Date'].dt.month
    df_combined['Day'] = df_combined['Date'].dt.day
    
    # Add district and zone
    df_combined['District'] = district
    df_combined['Zone'] = ""
    
    # Convert time_decimal to hour and minute
    df_combined['time_decimal_float'] = pd.to_numeric(df_combined['time_decimal'], errors='coerce')
    
    # Drop rows with invalid time values
    df_combined = df_combined.dropna(subset=['time_decimal_float'])
    
    df_combined['hour'] = df_combined['time_decimal_float'].astype(int)
    df_combined['minute'] = ((df_combined['time_decimal_float'] - df_combined['hour']) * 100).round().astype(int)
    
    # Create full datetime column
    df_combined['datetime'] = df_combined['Date'] + pd.to_timedelta(
        df_combined['hour'].astype(str) + ' hours'
    ) + pd.to_timedelta(
        df_combined['minute'].astype(str) + ' minutes'
    )
    
    # Select and reorder columns
    df_combined = df_combined[[
        'datetime', 'Date', 'Year', 'Month', 'Day', 
        'District', 'Zone', 'time_decimal', 'generation_kw'
    ]]
    
    # Sort by datetime
    df_combined = df_combined.sort_values(['District', 'datetime']).reset_index(drop=True)
    
    logger.info(f"  ✅ Processed {len(data_sheets)} sheet(s) → {len(df_combined):,} rows")
    
    return df_combined


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
            if df is not None and len(df) > 0:
                all_data.append(df)
                success_count += 1
            else:
                failed_files.append(file_path.name)
        except Exception as e:
            logger.error(f"❌ Failed: {file_path.name}")
            logger.error(f"   Error: {str(e)}")
            failed_files.append(file_path.name)
            continue
    
    if not all_data:
        logger.error("\n❌ No data was successfully ingested")
        return
    
    # Combine all dataframes
    logger.info(f"\n{'='*60}")
    logger.info(f"Combining data from {success_count} files...")
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates (in case of overlap)
    logger.info(f"Checking for duplicates...")
    original_len = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['District', 'datetime'])
    duplicates_removed = original_len - len(df_combined)
    if duplicates_removed > 0:
        logger.info(f"  Removed {duplicates_removed:,} duplicate records")
    
    # Save to Bronze as parquet
    output_file = bronze_path / "bronze_all_years.parquet"
    write_parquet(df_combined, output_file)
    
    logger.info("\n" + "="*60)
    logger.info("✅ INGESTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Successful: {success_count}/{len(excel_files)} files")
    logger.info(f"Total records: {len(df_combined):,}")
    logger.info(f"Unique districts: {df_combined['District'].nunique()}")
    logger.info(f"Districts: {sorted(df_combined['District'].unique())}")
    logger.info(f"Years: {sorted(df_combined['Year'].unique())}")
    logger.info(f"Date range: {df_combined['Date'].min().date()} to {df_combined['Date'].max().date()}")
    logger.info(f"Output: {output_file}")
    
    if failed_files:
        logger.warning(f"\n⚠️ Failed/Empty files: {len(failed_files)}")
        for f in failed_files[:10]:
            logger.warning(f"  - {f}")
    logger.info("="*60)


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    ingest_data()





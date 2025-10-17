"""
Data Ingestion Module
Reads raw Excel files and converts to Bronze layer (long-format parquet)
Handles both 2022 and 2018-2021, 2023-2024 formats
"""

import pandas as pd
import logging
from pathlib import Path
import re
from src.common.config import load_config
from src.common.io_utils import write_parquet

logger = logging.getLogger(__name__)


def extract_district_from_filename(filename):
    """
    Extract district name from various filename patterns.
    
    Handles patterns like:
    - "Annual Generation data in Ampara 2022.xlsx"
    - "Colombo-2022.xlsx"
    - "Generation Colombo 2022.xlsx"
    
    Args:
        filename (str): Excel filename
        
    Returns:
        str: District name
    """
    # Remove file extension
    name = filename.replace('.xlsx', '').replace('.xls', '')
    
    # Split by common delimiters
    parts = name.replace('-', ' ').replace('_', ' ').split()
    
    # Known district names in Sri Lanka
    districts = [
        'Ampara', 'Anuradhapura', 'Badulla', 'Batticaloa', 'Colombo',
        'Galle', 'Gampaha', 'Hambantota', 'Jaffna', 'Kalutara',
        'Kandy', 'Kegalle', 'Kilinochchi', 'Kurunegala', 'Mannar',
        'Matale', 'Matara', 'Monaragala', 'Mullaitivu', 'NuwaraEliya',
        'Nuwara Eliya', 'Polonnaruwa', 'Puttalam', 'Ratnapura',
        'Trincomalee', 'Vavuniya'
    ]
    
    # Find first matching district name in the filename
    for part in parts:
        # Check exact match (case-insensitive)
        for district in districts:
            if part.lower() == district.lower():
                return district
            # Handle "NuwaraEliya" vs "Nuwara Eliya"
            if district.replace(' ', '').lower() == part.lower():
                return district.replace(' ', '')
    
    # If "in" keyword exists, take the word after it
    if 'in' in parts:
        in_index = parts.index('in')
        if in_index + 1 < len(parts):
            return parts[in_index + 1]
    
    return "Unknown"


def parse_time_column(col_name):
    """
    Parse time column name to extract hour and minute.
    Handles formats: 8, 8.05, 08:00, 08:05, etc.
    
    Args:
        col_name: Column name
        
    Returns:
        tuple: (hour, minute) or None if not a time column
    """
    col_str = str(col_name).strip()
    
    # Format 1: HH:MM (e.g., "08:00", "08:05")
    if ':' in col_str:
        try:
            parts = col_str.split(':')
            hour = int(parts[0])
            minute = int(parts[1])
            return (hour, minute)
        except:
            return None
    
    # Format 2: Decimal (e.g., 8, 8.05, 8.1)
    try:
        time_float = float(col_str)
        hour = int(time_float)
        minute = int((time_float - hour) * 100)
        return (hour, minute)
    except:
        return None


def read_wide_excel(file_path):
    """
    Read wide-format Excel file with Date + time interval columns.
    Handles both 2022 and 2018-2021, 2023-2024 formats.
    
    Args:
        file_path (Path): Path to Excel file
        
    Returns:
        pd.DataFrame: Long-format dataframe with datetime column
    """
    logger.info(f"Reading {file_path.name}")
    
    # Extract district
    district = extract_district_from_filename(file_path.name)
    logger.info(f"  Detected district: {district}")
    
    # Read all sheets
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    
    # Filter out README sheets
    data_sheets = [
        s for s in sheet_names 
        if 'readme' not in s.lower() and 'info' not in s.lower()
    ]
    
    logger.info(f"  Found {len(sheet_names)} total sheets, processing {len(data_sheets)} data sheets")
    
    all_sheets_data = []
    
    for sheet_name in data_sheets:
        try:
            # Read sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df.columns = [str(c).strip() for c in df.columns]
            
            if len(df.columns) < 2 or len(df) == 0:
                logger.warning(f"  ⚠️ Skipping sheet '{sheet_name}' - insufficient data")
                continue
            
            # Identify metadata columns and time columns
            meta_cols_possible = ['Date', 'Month', 'Day', 'Year', 'District', 'Zone']
            meta_cols = [c for c in df.columns if c in meta_cols_possible]
            
            # Time columns are those that can be parsed as time
            time_cols = []
            time_mapping = {}  # Maps column name to (hour, minute)
            
            for col in df.columns:
                if col not in meta_cols:
                    time_tuple = parse_time_column(col)
                    if time_tuple:
                        time_cols.append(col)
                        time_mapping[col] = time_tuple
            
            if not time_cols:
                logger.warning(f"  ⚠️ Skipping sheet '{sheet_name}' - no time columns found")
                continue
            
            # Melt to long format
            df_long = df.melt(
                id_vars=meta_cols,
                value_vars=time_cols,
                var_name='time_decimal',
                value_name='generation_kw'
            )
            
            # Extract hour and minute from time_mapping
            df_long['hour'] = df_long['time_decimal'].map(lambda x: time_mapping.get(x, (0, 0))[0])
            df_long['minute'] = df_long['time_decimal'].map(lambda x: time_mapping.get(x, (0, 0))[1])
            
            # Handle Date column
            if 'Date' in df_long.columns:
                # Try to parse as datetime
                df_long['Date'] = pd.to_datetime(df_long['Date'], errors='coerce')
                
                # If dates are invalid (like day numbers), reconstruct from sheet name and year
                if df_long['Date'].isna().all() or df_long['Date'].dt.year.min() < 1900:
                    # Extract year from filename
                    year_match = re.search(r'20\d{2}', file_path.name)
                    if year_match:
                        year = int(year_match.group())
                        
                        # Extract month from sheet name or use Month column
                        if 'Month' in df_long.columns:
                            month = df_long['Month'].iloc[0]
                        else:
                            month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                                         'July', 'August', 'September', 'October', 'November', 'December']
                            if sheet_name in month_names:
                                month = month_names.index(sheet_name) + 1
                            else:
                                month = 1
                        
                        # Use Day column if exists, otherwise use Date as day number
                        if 'Day' in df_long.columns:
                            day = df_long['Day']
                        else:
                            # Date column contains day numbers (1, 2, 3, ...)
                            day = pd.to_numeric(df_long['Date'], errors='coerce')
                        
                        # Reconstruct date
                        df_long['Date'] = pd.to_datetime(
                            {'year': year, 'month': month, 'day': day},
                            errors='coerce'
                        )
            
            # Drop rows with invalid dates
            df_long = df_long.dropna(subset=['Date'])
            
            if len(df_long) == 0:
                logger.warning(f"  ⚠️ Skipping sheet '{sheet_name}' - all dates invalid")
                continue
            
            # Extract Year, Month, Day if not present
            if 'Year' not in df_long.columns:
                df_long['Year'] = df_long['Date'].dt.year
            if 'Month' not in df_long.columns:
                df_long['Month'] = df_long['Date'].dt.month
            if 'Day' not in df_long.columns:
                df_long['Day'] = df_long['Date'].dt.day
            
            # Add district and zone
            df_long['District'] = district
            df_long['Zone'] = ""
            
            # Create datetime column
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
            df_long = df_long.sort_values('datetime').reset_index(drop=True)
            
            all_sheets_data.append(df_long)
            
        except Exception as e:
            logger.warning(f"  ⚠️ Skipping sheet '{sheet_name}': {str(e)}")
            continue
    
    if not all_sheets_data:
        logger.error(f"  ❌ No valid data sheets found")
        return None
    
    # Combine all sheets
    df_combined = pd.concat(all_sheets_data, ignore_index=True)
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
    logger.info(f"Combined: {len(df_combined):,} rows")
    
    # FILTER OUT INVALID YEARS (1970 and other errors)
    logger.info(f"Filtering out invalid dates...")
    before_filter = len(df_combined)
    df_combined = df_combined[
        (df_combined['Year'] >= 2018) & 
        (df_combined['Year'] <= 2025)
    ]
    after_filter = len(df_combined)
    removed = before_filter - after_filter
    if removed > 0:
        logger.warning(f"  Removed {removed:,} rows with invalid years")
    logger.info(f"After filter: {len(df_combined):,} rows")
    
    # Check for duplicates
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





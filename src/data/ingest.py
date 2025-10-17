"""
Data Ingestion Module
Reads raw Excel files and converts to Bronze layer (long-format parquet)
Handles both 2022 format and 2018-2021, 2023-2024 formats
"""

import pandas as pd
import logging
from pathlib import Path
import re
from src.common.config import load_config
from src.common.io_utils import write_parquet

logger = logging.getLogger(__name__)


def extract_district_from_filename(filename):
    """Extract district name from various filename patterns"""
    name = filename.replace('.xlsx', '').replace('.xls', '')
    parts = name.replace('-', ' ').replace('_', ' ').split()
    
    districts = [
        'Ampara', 'Anuradhapura', 'Badulla', 'Batticaloa', 'Colombo',
        'Galle', 'Gampaha', 'Hambantota', 'Jaffna', 'Kalutara',
        'Kandy', 'Kegalle', 'Kilinochchi', 'Kurunegala', 'Mannar',
        'Matale', 'Matara', 'Monaragala', 'Mullaitivu', 'NuwaraEliya',
        'Nuwara Eliya', 'Polonnaruwa', 'Puttalam', 'Ratnapura',
        'Trincomalee', 'Vavuniya'
    ]
    
    for part in parts:
        for district in districts:
            if part.lower() == district.lower():
                return district
            if district.replace(' ', '').lower() == part.lower():
                return district.replace(' ', '')
    
    if 'in' in parts:
        in_index = parts.index('in')
        if in_index + 1 < len(parts):
            return parts[in_index + 1]
    
    return "Unknown"


def parse_time_column(col_name):
    """
    Parse time column name to extract hour and minute.
    Handles formats: 8, 8.05, 08:00, 08:05, etc.
    
    Returns: (hour, minute) tuple or None if not a time column
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
    """
    logger.info(f"Reading {file_path.name}")
    
    district = extract_district_from_filename(file_path.name)
    logger.info(f"  Detected district: {district}")
    
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    
    # Filter out README sheets
    data_sheets = [s for s in sheet_names if 'readme' not in s.lower() and 'info' not in s.lower()]
    logger.info(f"  Found {len(sheet_names)} total sheets, processing {len(data_sheets)} data sheets")
    
    all_sheets_data = []
    
    for sheet_name in data_sheets:
        try:
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
            
            logger.debug(f"  Sheet '{sheet_name}': {len(meta_cols)} metadata cols, {len(time_cols)} time cols")
            
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
                            day = df_long['Date']
                        
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
            logger.debug(f"  ✓ Sheet '{sheet_name}': {len(df_long)} rows")
            
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
    """Main ingestion function"""
    cfg = load_config()
    
    raw_path = Path(cfg.data_paths['raw'])
    bronze_path = Path(cfg.data_paths['bronze'])
    bronze_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("STEP 1: INGESTION (Raw → Bronze)")
    logger.info("="*60)
    
    excel_files = sorted(list(raw_path.glob('*.xlsx')) + list(raw_path.glob('*.xls')))
    
    if not excel_files:
        logger.error(f"❌ No Excel files found in {raw_path}")
        return
    
    logger.info(f"Found {len(excel_files)} Excel files\n")
    
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
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Combining data from {success_count} files...")
    df_combined = pd.concat(all_data, ignore_index=True)
    
    logger.info(f"Checking for duplicates...")
    original_len = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['District', 'datetime'])
    duplicates_removed = original_len - len(df_combined)
    if duplicates_removed > 0:
        logger.info(f"  Removed {duplicates_removed:,} duplicate records")
    
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    ingest_data()





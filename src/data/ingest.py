"""
Data Ingestion Module
Reads raw Excel files and converts to Bronze layer (long-format parquet)
Handles both 2022 and 2018-2021, 2023-2024 formats
"""

import pandas as pd
import logging
from pathlib import Path
import re
from datetime import datetime 
from src.common.config import load_config
from src.common.io_utils import write_parquet

logger = logging.getLogger(__name__)


def extract_district_from_filename(filename):
    """Extract district name from various filename patterns."""
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
    """Parse time column name to extract hour and minute."""
    col_str = str(col_name).strip()
    
    if ':' in col_str:
        try:
            parts = col_str.split(':')
            hour = int(parts[0])
            minute = int(parts[1])
            return (hour, minute)
        except:
            return None
    
    try:
        time_float = float(col_str)
        hour = int(time_float)
        minute = int((time_float - hour) * 100)
        return (hour, minute)
    except:
        return None


def read_wide_excel(file_path):
    """Read wide-format Excel file with Date + time interval columns."""
    logger.info(f"Reading {file_path.name}")
    
    district = extract_district_from_filename(file_path.name)
    logger.info(f"  Detected district: {district}")
    
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    
    data_sheets = [
        s for s in sheet_names 
        if 'readme' not in s.lower() and 'info' not in s.lower()
    ]
    
    logger.info(f"  Found {len(sheet_names)} total sheets, processing {len(data_sheets)} data sheets")
    
    all_sheets_data = []
    
    for sheet_name in data_sheets:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            df.columns = [str(c).strip() for c in df.columns]
            
            if len(df.columns) < 2 or len(df) == 0:
                logger.warning(f"  ⚠️ Skipping sheet '{sheet_name}' - insufficient data")
                continue
            
            meta_cols_possible = ['Date', 'Month', 'Day', 'Year', 'District', 'Zone']
            meta_cols = [c for c in df.columns if c in meta_cols_possible]
            
            time_cols = []
            time_mapping = {}
            
            for col in df.columns:
                if col not in meta_cols:
                    time_tuple = parse_time_column(col)
                    if time_tuple:
                        time_cols.append(col)
                        time_mapping[col] = time_tuple
            
            if not time_cols:
                logger.warning(f"  ⚠️ Skipping sheet '{sheet_name}' - no time columns found")
                continue
            
            df_long = df.melt(
                id_vars=meta_cols,
                value_vars=time_cols,
                var_name='time_decimal',
                value_name='generation_kw'
            )
            
            df_long['hour'] = df_long['time_decimal'].map(lambda x: time_mapping.get(x, (0, 0))[0])
            df_long['minute'] = df_long['time_decimal'].map(lambda x: time_mapping.get(x, (0, 0))[1])
            
            # Handle Date column
            if 'Date' in df_long.columns:
                df_long['Date'] = pd.to_datetime(df_long['Date'], errors='coerce')
                
                # If dates are completely invalid, try reconstruction
                if df_long['Date'].isna().all() or (df_long['Date'].notna().any() and df_long['Date'].dt.year.min() < 1900):
                    year_match = re.search(r'20\d{2}', file_path.name)
                    if year_match:
                        year = int(year_match.group())
                        
                        if 'Month' in df_long.columns:
                            month = df_long['Month'].iloc[0]
                        else:
                            month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                                         'July', 'August', 'September', 'October', 'November', 'December']
                            if sheet_name in month_names:
                                month = month_names.index(sheet_name) + 1
                            else:
                                month = 1
                        
                        if 'Day' in df_long.columns:
                            day = df_long['Day']
                        else:
                            day = pd.to_numeric(df_long['Date'], errors='coerce')
                        
                        df_long['Date'] = pd.to_datetime(
                            {'year': year, 'month': month, 'day': day},
                            errors='coerce'
                        )
            
            # Drop invalid dates
            df_long = df_long.dropna(subset=['Date'])
            
            if len(df_long) == 0:
                logger.warning(f"  ⚠️ Skipping sheet '{sheet_name}' - all dates invalid")
                continue
            
            # ✅ ALWAYS extract Year, Month, Day from the Date column
            df_long['Year'] = df_long['Date'].dt.year
            df_long['Month'] = df_long['Date'].dt.month
            df_long['Day'] = df_long['Date'].dt.day
            
            # ✅ FIX: If Year is 1970 (Unix epoch bug), replace with year from filename
            if (df_long['Year'] == 1970).any():
                year_match = re.search(r'20\d{2}', file_path.name)
                if year_match:
                    correct_year = int(year_match.group())
                    wrong_year_mask = df_long['Year'] == 1970
                    
                    # Fix Year column
                    df_long.loc[wrong_year_mask, 'Year'] = correct_year
                    
                    # Fix Date column by replacing the year
                    df_long.loc[wrong_year_mask, 'Date'] = df_long.loc[wrong_year_mask].apply(
                        lambda row: row['Date'].replace(year=correct_year), axis=1
                    )
            
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
            
            df_long = df_long.sort_values('datetime').reset_index(drop=True)
            all_sheets_data.append(df_long)
            
        except Exception as e:
            logger.warning(f"  ⚠️ Skipping sheet '{sheet_name}': {str(e)}")
            continue
    
    if not all_sheets_data:
        logger.error(f"  ❌ No valid data sheets found")
        return None
    
    df_combined = pd.concat(all_sheets_data, ignore_index=True)
    logger.info(f"  ✅ Processed {len(data_sheets)} sheet(s) → {len(df_combined):,} rows")
    
    return df_combined


def ingest_data():
    """Main ingestion function: reads all raw Excel files and creates Bronze parquet."""
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
    logger.info(f"Combined: {len(df_combined):,} rows")
    
    # DEBUG: Show year distribution BEFORE filtering
    print("\n" + "="*60)
    print("DEBUG: Year distribution BEFORE filtering:")
    print(df_combined['Year'].value_counts().sort_index())
    print(f"\nYear column dtype: {df_combined['Year'].dtype}")
    print(f"Total unique years: {df_combined['Year'].nunique()}")
    print(f"Min year: {df_combined['Year'].min()}")
    print(f"Max year: {df_combined['Year'].max()}")
    
    nan_years = df_combined['Year'].isna().sum()
    if nan_years > 0:
        print(f"⚠️ WARNING: {nan_years:,} rows have NaN/NULL year values!")
    
    suspicious = df_combined[(df_combined['Year'] < 2018) | (df_combined['Year'] > 2030)]
    if len(suspicious) > 0:
        print(f"⚠️ WARNING: {len(suspicious):,} rows have suspicious year values:")
        print(suspicious['Year'].value_counts().sort_index())
    print("="*60 + "\n")
    
    # Filter invalid years
    current_year = datetime.now().year
    logger.info(f"Filtering out invalid dates...")
    before_filter = len(df_combined)
    df_combined = df_combined[
        (df_combined['Year'] >= 2018) & 
        (df_combined['Year'] <= current_year + 1)
    ]
    after_filter = len(df_combined)
    removed = before_filter - after_filter
    if removed > 0:
        logger.warning(f"  Removed {removed:,} rows with invalid years")
    logger.info(f"  Keeping years: 2018-{current_year + 1}")
    logger.info(f"After filter: {len(df_combined):,} rows")
    
    # DEBUG: Show year distribution AFTER filtering
    print("\n" + "="*60)
    print("DEBUG: Year distribution AFTER filtering:")
    print(df_combined['Year'].value_counts().sort_index())
    print("="*60 + "\n")
    
    # Check for duplicates
    logger.info(f"Checking for duplicates...")
    original_len = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['District', 'datetime'])
    duplicates_removed = original_len - len(df_combined)
    if duplicates_removed > 0:
        logger.info(f"  Removed {duplicates_removed:,} duplicate records")
    
    # Save to Bronze
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




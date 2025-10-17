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

def check_generation_values(df):
    """Validate generation values are within expected range"""
    logger.info("Checking generation values...")
    
    # Basic stats
    gen_stats = df['generation_kw'].describe()
    logger.info(f"  Generation statistics:")
    logger.info(f"    Min: {gen_stats['min']:.2f} kW")
    logger.info(f"    Max: {gen_stats['max']:.2f} kW")
    logger.info(f"    Mean: {gen_stats['mean']:.2f} kW")
    logger.info(f"    Median: {gen_stats['50%']:.2f} kW")
    
    # Check for negative values
    negative = df[df['generation_kw'] < 0]
    if len(negative) > 0:
        logger.warning(f"  ⚠️ Found {len(negative):,} negative generation values")
    else:
        logger.info(f"  ✅ No negative values")
    
    # Check for zeros
    zeros = df[df['generation_kw'] == 0]
    zero_pct = (len(zeros) / len(df)) * 100
    logger.info(f"  Zero values: {len(zeros):,} ({zero_pct:.2f}%)")
    
    # Check for extreme outliers (> 99.9th percentile)
    p999 = df['generation_kw'].quantile(0.999)
    outliers = df[df['generation_kw'] > p999]
    if len(outliers) > 0:
        logger.info(f"  Extreme outliers (>{p999:.0f} kW): {len(outliers):,} records")
    
    return gen_stats

def check_time_coverage(df):
    """Check for gaps in time series"""
    logger.info("Checking time coverage...")
    
    # Expected intervals per day (8:00 to 17:00, 5-minute intervals)
    expected_intervals_per_day = 109  # (17:00 - 8:00) * 12 + 1
    
    # Group by date and district, count intervals
    coverage = df.groupby(['District', 'Date']).size().reset_index(name='interval_count')
    
    # Find days with incomplete data
    incomplete = coverage[coverage['interval_count'] < expected_intervals_per_day]
    
    if len(incomplete) > 0:
        incomplete_pct = (len(incomplete) / len(coverage)) * 100
        logger.warning(f"  ⚠️ {len(incomplete):,} district-days with incomplete data ({incomplete_pct:.2f}%)")
        logger.info(f"    Example: {incomplete.iloc[0]['District']} on {incomplete.iloc[0]['Date'].date()} has {incomplete.iloc[0]['interval_count']} intervals")
    else:
        logger.info(f"  ✅ All district-days have complete data")
    
    return coverage


def check_district_coverage(df):
    """Check data availability per district"""
    logger.info("Checking district coverage...")
    
    district_stats = df.groupby('District').agg({
        'Date': ['min', 'max', 'nunique'],
        'generation_kw': ['count', 'mean', 'std']
    }).round(2)
    
    logger.info(f"  Districts found: {df['District'].nunique()}")
    logger.info(f"  District coverage summary:")
    
    for district in sorted(df['District'].unique()):
        district_data = df[df['District'] == district]
        min_date = district_data['Date'].min().date()
        max_date = district_data['Date'].max().date()
        n_days = district_data['Date'].nunique()
        n_records = len(district_data)
        
        logger.info(f"    {district}: {n_days} days, {n_records:,} records ({min_date} to {max_date})")
    
    return district_stats

def validate_data():
    """Main validation function"""
    cfg = load_config()
    
    bronze_path = Path(cfg.data_paths['bronze'])
    silver_path = Path(cfg.data_paths['silver'])
    silver_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("STEP 2: VALIDATION (Bronze → Silver)")
    logger.info("="*60)
    
    # Read Bronze data
    bronze_file = bronze_path / "bronze_all_years.parquet"
    if not bronze_file.exists():
        logger.error(f"❌ Bronze file not found: {bronze_file}")
        logger.info("Please run ingestion first: python -m src.data.ingest")
        return
    
    df = read_parquet(bronze_file)
    logger.info(f"Loaded {len(df):,} records from Bronze layer\n")
    
    # Run validation checks
    validation_results = {}
    
    # 1. Missing values
    validation_results['missing'] = check_missing_values(df)
    print()
    
    # 2. Date ranges
    validation_results['date_range'] = check_date_ranges(df)
    print()
    
    # 3. Generation values
    validation_results['generation_stats'] = check_generation_values(df)
    print()
    
    # 4. Time coverage
    validation_results['coverage'] = check_time_coverage(df)
    print()
    
    # 5. District coverage
    validation_results['district_stats'] = check_district_coverage(df)
    print()
    
    # Basic cleaning: remove invalid rows
    logger.info("Applying basic cleaning...")
    original_len = len(df)
    
    # Remove rows with missing critical values
    df = df.dropna(subset=['datetime', 'District', 'generation_kw'])
    
    # Remove negative generation values
    df = df[df['generation_kw'] >= 0]
    
    # Remove duplicate datetime-district combinations
    df = df.drop_duplicates(subset=['District', 'datetime'], keep='first')
    
    cleaned_len = len(df)
    removed = original_len - cleaned_len
    
    if removed > 0:
        logger.info(f"  Removed {removed:,} invalid rows ({(removed/original_len)*100:.2f}%)")
    logger.info(f"  Clean data: {cleaned_len:,} rows")
    
    # Save to Silver
    output_file = silver_path / "silver_all_years.parquet"
    write_parquet(df, output_file)
    
    logger.info("\n" + "="*60)
    logger.info("✅ VALIDATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Input: {original_len:,} records")
    logger.info(f"Output: {cleaned_len:,} records")
    logger.info(f"Removed: {removed:,} records")
    logger.info(f"Silver data saved to: {output_file}")
    logger.info("="*60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    validate_data()
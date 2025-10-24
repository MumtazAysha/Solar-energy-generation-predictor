"""
Feature Engineering Module
Builds model-ready features from Gold data
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple

from src.common.config import load_config
from src.common.io_utils import read_parquet, write_parquet

logger = logging.getLogger(__name__)


def add_lag_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """
    Add lag features with year carryover - fully vectorized (no row loops).
    """
    logger.info(f"Adding lag features: {lags}")
    df = df.sort_values(['District', 'datetime']).copy()
    
    # Standard lag creation
    for lag in lags:
        col = f'generation_kw_lag{lag}'
        df[col] = df.groupby('District', group_keys=False)['generation_kw'].shift(lag)
    
    # Previous-day same-time lag
    prev_day_lag = 109
    if prev_day_lag not in lags:
        col = f'generation_kw_lag{prev_day_lag}'
        logger.info(f"Adding previous-day same-time lag: {prev_day_lag}")
        df[col] = df.groupby('District', group_keys=False)['generation_kw'].shift(prev_day_lag)
    
    # ✅ SUPER FAST: Handle Jan 1 with merge instead of loops
    logger.info("Applying year carryover for January 1st...")
    
    # Identify Jan 1 rows
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['year'] = df['datetime'].dt.year
    df['time'] = df['datetime'].dt.time
    
    jan1_mask = (df['month'] == 1) & (df['day'] == 1)
    
    if jan1_mask.sum() > 0:
        # Create a reference dataset for previous year Jan 1
        jan1_ref = df[jan1_mask].copy()
        jan1_ref['target_year'] = jan1_ref['year'] + 1  # This year's Jan 1 → next year's Jan 1
        jan1_ref = jan1_ref.rename(columns={'generation_kw': 'prev_year_gen'})
        jan1_ref = jan1_ref[['District', 'target_year', 'time', 'prev_year_gen']]
        
        # Merge back to fill lag109 for Jan 1 rows
        df = df.merge(
            jan1_ref,
            left_on=['District', 'year', 'time'],
            right_on=['District', 'target_year', 'time'],
            how='left'
        )
        
        # Fill lag109 for Jan 1 where we found a match
        jan1_with_prev = jan1_mask & df['prev_year_gen'].notna()
        df.loc[jan1_with_prev, 'generation_kw_lag109'] = df.loc[jan1_with_prev, 'prev_year_gen']
        
        # For short-term lags, estimate from previous year's value
        if jan1_with_prev.sum() > 0:
            df.loc[jan1_with_prev, 'generation_kw_lag1'] = df.loc[jan1_with_prev, 'prev_year_gen'] * 0.98
            df.loc[jan1_with_prev, 'generation_kw_lag12'] = df.loc[jan1_with_prev, 'prev_year_gen'] * 0.95
            df.loc[jan1_with_prev, 'generation_kw_lag24'] = df.loc[jan1_with_prev, 'prev_year_gen'] * 0.90
            if 'generation_kw_lag36' in df.columns:
                df.loc[jan1_with_prev, 'generation_kw_lag36'] = df.loc[jan1_with_prev, 'prev_year_gen'] * 0.85
        
        # For 2018 Jan 1 (no previous year), use Jan 2-7 average
        jan1_2018 = jan1_mask & (df['year'] == 2018)
        if jan1_2018.sum() > 0:
            # Calculate average from Jan 2-7 by district and time
            jan2_7 = df[(df['month'] == 1) & (df['day'].between(2, 7))]
            jan2_7_avg = jan2_7.groupby(['District', 'time'])['generation_kw'].mean().reset_index()
            jan2_7_avg = jan2_7_avg.rename(columns={'generation_kw': 'jan2_7_avg'})
            
            df = df.merge(jan2_7_avg, on=['District', 'time'], how='left')
            df.loc[jan1_2018, 'generation_kw_lag109'] = df.loc[jan1_2018, 'jan2_7_avg']
            df.loc[jan1_2018, 'generation_kw_lag1'] = df.loc[jan1_2018, 'jan2_7_avg']
            df.loc[jan1_2018, 'generation_kw_lag12'] = df.loc[jan1_2018, 'jan2_7_avg']
            df.loc[jan1_2018, 'generation_kw_lag24'] = df.loc[jan1_2018, 'jan2_7_avg']
            if 'generation_kw_lag36' in df.columns:
                df.loc[jan1_2018, 'generation_kw_lag36'] = df.loc[jan1_2018, 'jan2_7_avg']
            
            df = df.drop(columns=['jan2_7_avg'], errors='ignore')
        
        # Clean up merge columns
        df = df.drop(columns=['target_year', 'prev_year_gen'], errors='ignore')
        
        jan1_fixed = jan1_with_prev.sum() + jan1_2018.sum()
        logger.info(f"Applied year carryover to {jan1_fixed} January 1st records")
    
    # Clean up helper columns
    df = df.drop(columns=['month', 'day', 'year', 'time'])
    
    # Fill remaining NaN with 0
    lag_cols = [c for c in df.columns if c.startswith('generation_kw_lag')]
    for col in lag_cols:
        df[col] = df[col].fillna(0)
    
    return df


def add_rolling_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Add rolling statistics on generation_kw per District.
    windows are in rows (5-minute steps). Example: 12 for 1 hour.
    """
    logger.info(f"Adding rolling window features: {windows}")
    df = df.sort_values(['District', 'datetime']).copy()

    for w in windows:
        # Reasonable min_periods to get early estimates
        minp = max(1, w // 2)
        s = df.groupby('District', group_keys=False)['generation_kw']

        df[f'gen_roll_mean_{w}'] = s.transform(lambda x: x.rolling(window=w, min_periods=minp).mean())
        df[f'gen_roll_std_{w}']  = s.transform(lambda x: x.rolling(window=w, min_periods=minp).std())
        df[f'gen_roll_min_{w}']  = s.transform(lambda x: x.rolling(window=w, min_periods=minp).min())
        df[f'gen_roll_max_{w}']  = s.transform(lambda x: x.rolling(window=w, min_periods=minp).max())

    # Simple EMA as additional signal (fixed half-life ~ 1 hour)
    logger.info("Adding EMA features (fixed half-life ~1 hour)")
    ema_span = 12  # 1 hour at 5-min intervals
    df['gen_ema_12'] = df.groupby('District', group_keys=False)['generation_kw'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False, min_periods=ema_span//2).mean()
    )

    return df


def select_feature_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    base_features = [
        'hour', 'minute', 'day_of_year', 'day_of_week', 'is_weekend',
        'minute_of_day', 'season', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
        'solar_elev_approx'
    ]

    lag_cols = [c for c in df.columns if c.startswith('generation_kw_lag')]
    roll_cols = [c for c in df.columns if c.startswith('gen_roll_')]
    ema_cols = [c for c in df.columns if c.startswith('gen_ema_')]

    # Keep District only as a metadata column; do NOT include it again in feature_cols
    cat_cols: List[str] = []  # or leave it out of features entirely here

    feature_cols = [c for c in base_features + lag_cols + roll_cols + ema_cols + cat_cols if c in df.columns]

    keep_cols = ['datetime', 'Date', 'Year', 'Month', 'Day', 'District', 'generation_kw'] + feature_cols

    # Deduplicate while preserving order
    keep_cols = list(dict.fromkeys(keep_cols))

    df_selected = df[keep_cols].copy()

    # Guarantee unique columns (belt-and-suspenders)
    df_selected = df_selected.loc[:, ~df_selected.columns.duplicated()]

    logger.info(f"Feature columns selected: {len(feature_cols)}")
    return df_selected, feature_cols



def drop_na_after_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NA in key engineered features, BUT keep January 1st rows.
    """
    before = len(df)
    
    # Identify January 1st rows to preserve
    jan1_mask = (df['datetime'].dt.month == 1) & (df['datetime'].dt.day == 1)
    
    # Get engineered feature columns
    engineered_cols = [c for c in df.columns if c.startswith('generation_kw_lag') or 
                       c.startswith('gen_roll_') or c.startswith('gen_ema_')]
    
    # Drop NA only for non-January-1st rows
    df_jan1 = df[jan1_mask].copy()
    df_other = df[~jan1_mask].dropna(subset=engineered_cols)
    
    # Recombine
    df = pd.concat([df_jan1, df_other], ignore_index=True).sort_values(['District', 'datetime'])
    
    after = len(df)
    logger.info(f"Dropped {before - after:,} rows with NA (preserved January 1st); remaining: {after:,}")
    return df

def build_features():
    """
    Main function: reads Gold data, adds features, saves train-ready dataset.
    """
    cfg = load_config()
    gold_path = Path(cfg.data_paths['gold'])
    out_path  = Path(cfg.data_paths['gold'])  # Save features alongside Gold
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("STEP 4: FEATURE ENGINEERING (Gold → Gold-Features)")
    logger.info("="*60)

    gold_file = gold_path / "gold_all_years.parquet"
    if not gold_file.exists():
        logger.error(f"❌ Gold file not found: {gold_file}")
        logger.info("Please run transformation first: python -m src.data.transform")
        return

    df = read_parquet(gold_file)
    logger.info(f"Loaded {len(df):,} rows from Gold layer\n")

    # Get lags and rolling windows from config
    lags = cfg.features.get('lags', [1, 12, 24, 36])
    windows = cfg.features.get('rolling', {}).get('windows', [6, 12, 24])

    # Add lag and rolling features
    df = add_lag_features(df, lags=lags)
    logger.info("")
    df = add_rolling_features(df, windows=windows)
    logger.info("")

    # Select features and clean
    df, feature_cols = select_feature_columns(df)

    # Target
    df = df.rename(columns={'generation_kw': 'target_kw'})

    # Ensure no NA in engineered columns for training
    df = drop_na_after_features(df)

    # Final sort and save
    df = df.sort_values(['District', 'datetime']).reset_index(drop=True)

    # Persist features
    features_file = out_path / "gold_features_all_years.parquet"
    write_parquet(df, features_file)

    # Also export a compact CSV sample (optional)
    sample_file = out_path / "gold_features_sample.csv"
    try:
        df_sample = df.sample(n=min(5000, len(df)), random_state=42)
        df_sample.to_csv(sample_file, index=False)
        logger.info(f"Sample saved to {sample_file}")
    except Exception as e:
        logger.warning(f"Could not save sample CSV: {e}")

    # Log final info
    logger.info("\n" + "="*60)
    logger.info("✅ FEATURE ENGINEERING COMPLETE")
    logger.info("="*60)
    logger.info(f"Records (train-ready): {len(df):,}")
    logger.info(f"Total columns: {len(df.columns)}")
    logger.info(f"Feature columns: {len(feature_cols)}")
    logger.info(f"Output: {features_file}")
    logger.info("="*60)

    # Show a small preview
    preview_cols = ['datetime', 'District', 'target_kw'] + feature_cols[:10]
    logger.info(f"\nPreview:\n{df[preview_cols].head(5).to_string(index=False)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    build_features()

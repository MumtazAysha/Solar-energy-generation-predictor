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
    Add lag features for generation_kw per District.
    lags are in units of rows (5-minute steps).
    """
    logger.info(f"Adding lag features: {lags}")
    df = df.sort_values(['District', 'datetime']).copy()

    for lag in lags:
        col = f'generation_kw_lag{lag}'
        df[col] = df.groupby('District', group_keys=False)['generation_kw'].shift(lag)

    # Previous-day same-time lag (109 intervals: 8:00–17:00 inclusive at 5-min steps)
    prev_day_lag = 109
    if prev_day_lag not in lags:
        col = f'generation_kw_lag{prev_day_lag}'
        logger.info(f"Adding previous-day same-time lag: {prev_day_lag}")
        df[col] = df.groupby('District', group_keys=False)['generation_kw'].shift(prev_day_lag)

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
    Drop rows with NA in key engineered features to ensure train readiness.
    """
    before = len(df)
    # Drop rows where any lag/rolling feature is NA
    engineered_cols = [c for c in df.columns if c.startswith('generation_kw_lag') or c.startswith('gen_roll_') or c.startswith('gen_ema_')]
    drop_cols = engineered_cols  # lags/rollings define usable training window
    df = df.dropna(subset=drop_cols)
    after = len(df)
    logger.info(f"Dropped {before - after:,} rows with NA in engineered features; remaining: {after:,}")
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

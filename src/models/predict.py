"""
Interactive CLI for on-demand solar prediction
Run: python -m src.models.predict
"""
import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from src.common.config import load_config
from src.common.io_utils import read_parquet
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")



def load_artifacts(cfg):
    models_path = Path(cfg.output_paths['models'])
    model = joblib.load(models_path / "random_forest_model.pkl")
    le = joblib.load(models_path / "district_encoder.pkl")
    feature_cols = (models_path / "feature_names.txt").read_text().splitlines()
    print("✅ Model and encoder loaded")
    return model, le, feature_cols

def build_feature_row(target_dt, district, le, gold_features_path):
    """
    Load actual features for the exact datetime from gold dataset.
    """
    gold_df = read_parquet(gold_features_path)
    
    # Filter by district
    district_df = gold_df[gold_df['District'] == district].copy()
    if district_df.empty:
        raise ValueError(f"District '{district}' not found in dataset")
    
    # Try exact datetime match first
    exact_match = district_df[district_df['datetime'] == target_dt]
    
    if not exact_match.empty:
        # Perfect match - use actual features from data!
        template = exact_match.iloc[0].copy()
    else:
        # Fallback: Find same time-of-day from nearest date
        target_time = target_dt.time()
        target_date = pd.Timestamp(target_dt).date()
        
        # Filter to same time of day
        district_df['time_only'] = district_df['datetime'].dt.time
        same_time = district_df[district_df['time_only'] == target_time].copy()
        
        if same_time.empty:
            # No exact time match - find nearest minute
            target_min = target_dt.hour * 60 + target_dt.minute
            district_df['minutes_day'] = (
                district_df['datetime'].dt.hour * 60 + 
                district_df['datetime'].dt.minute
            )
            idx = (district_df['minutes_day'] - target_min).abs().idxmin()
            template = district_df.loc[idx].copy()
        else:
            # Find same time from nearest date
            same_time['date_only'] = same_time['datetime'].dt.date
            same_time['date_diff'] = abs((same_time['date_only'] - target_date).apply(lambda x: x.days))
            idx = same_time['date_diff'].idxmin()
            template = same_time.loc[idx].copy()
        
        # Update only the date/time metadata (keep all lag features!)
        template['datetime'] = pd.Timestamp(target_dt)
        template['Year'] = target_dt.year
        template['Month'] = target_dt.month
        template['Day'] = target_dt.day
        template['day_of_year'] = target_dt.timetuple().tm_yday
        template['day_of_week'] = target_dt.weekday()
        template['is_weekend'] = int(target_dt.weekday() >= 5)
        
        # Update cyclic features
        template['hour_sin'] = np.sin(2 * np.pi * template['hour'] / 24)
        template['hour_cos'] = np.cos(2 * np.pi * template['hour'] / 24)
        template['doy_sin'] = np.sin(2 * np.pi * template['day_of_year'] / 365)
        template['doy_cos'] = np.cos(2 * np.pi * template['day_of_year'] / 365)
    
    # Add District_encoded (required by model)
    template['District_encoded'] = le.transform([district])[0]
    
    return template



def make_interpolated_prediction(model, le, feature_cols, district, dt_str, gold_features_path):
    target_dt = pd.to_datetime(dt_str)
    prev_dt = target_dt.floor('5min')
    next_dt = target_dt.ceil('5min')
    if prev_dt == next_dt:
        row = build_feature_row(prev_dt, district, le, gold_features_path)
        X = row[feature_cols].values.reshape(1, -1)
        y_hat = model.predict(X)[0]
        return float(y_hat), False
    row_prev = build_feature_row(prev_dt, district, le, gold_features_path)
    row_next = build_feature_row(next_dt, district, le, gold_features_path)
    X_prev = row_prev[feature_cols].values.reshape(1, -1)
    X_next = row_next[feature_cols].values.reshape(1, -1)
    y_prev = model.predict(X_prev)[0]
    y_next = model.predict(X_next)[0]
    t0 = prev_dt.hour * 60 + prev_dt.minute
    t1 = next_dt.hour * 60 + next_dt.minute
    t = target_dt.hour * 60 + target_dt.minute
    frac = (t - t0) / (t1 - t0)
    interp = y_prev + frac * (y_next - y_prev)
    return float(interp), True

def predict_day(model, le, feature_cols, district, date_str, gold_features_path):
    """Vectorized full-day prediction for one district (100–300x faster)."""
    date = pd.to_datetime(date_str).date()
    times = pd.date_range(f"{date} 00:00", f"{date} 23:55", freq="5min")
    rows = []

    # Build all rows quickly
    for t in times:
        row = build_feature_row(t, district, le, gold_features_path)
        rows.append(row[feature_cols].values)

    X = np.vstack(rows)
    preds = model.predict(X)

    df = pd.DataFrame({
        "datetime": times,
        "district": district,
        "prediction": preds,
        "interpolated": [False] * len(times)
    })
    return df


# =====================================================
# Entry point – choose Interactive or Full‑Day Export
# =====================================================
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    print("\nWelcome to Solar Predictor!")
    print("[1] Individual prediction mode")
    print("[2] Full‑day export for all districts")
    print("[Q] Quit")

    mode = input("\nSelect mode: ").strip().lower()
    if mode in {"q", "quit", "exit"}:
        print("Exiting.")
        raise SystemExit

    cfg = load_config()
    model, le, feature_cols = load_artifacts(cfg)
    gold_features_path = Path(cfg.data_paths['gold']) / "gold_features_all_years.parquet"

    # -----  MODE 1: Interactive prediction  -----
    if mode == "1":
        while True:
            district = input("\nEnter district (or Q to quit): ").strip()
            if district.lower() in {"q", "quit", "exit"}:
                break
            if district not in le.classes_:
                print("District not recognized.")
                continue
            dt_str = input("Enter datetime (YYYY-MM-DD HH:MM): ").strip()
            if dt_str.lower() in {"q", "quit", "exit"}:
                break
            try:
                y_pred, interpolated = make_interpolated_prediction(
                    model, le, feature_cols, district, dt_str, gold_features_path
                )
                tag = "[INTERPOLATED]" if interpolated else ""
                print(f"{tag} {district} @ {dt_str} → {y_pred:.2f} kW")
            except Exception as e:
                print(f"Error: {e}")

    # -----  MODE 2: Full‑day export  -----
    elif mode == "2":
        date_str = input("Enter date (YYYY-MM-DD): ").strip()
        all_results = []
        print(f"\n🕒 Generating 5‑minute predictions for {date_str}...\n")
        for district in le.classes_[:3]:  # test first 3 (Ampara, Anuradhapura, Badulla)
            print(f"  → {district:<15}", end="", flush=True)
            df_pred = predict_day(model, le, feature_cols, district, date_str, gold_features_path)
            df_pred["District"] = district
            all_results.append(df_pred)
            print("✓")

        all_df = pd.concat(all_results, ignore_index=True)
        merged_file = Path(f"outputs/models/pred_all_{date_str}.csv")
        all_df.to_csv(merged_file, index=False)
        print(f"\n✅ Combined daily CSV saved → {merged_file}")
        print(f"   Total rows: {len(all_df):,}")
        raise SystemExit  # <<— this line forces the program to END once CSV is saved

    else:
        print("Invalid choice. Run again.")

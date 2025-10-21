"""
Interactive CLI for on-demand solar prediction
Run: python -m src.models.predict
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
from src.common.config import load_config
from src.common.io_utils import read_parquet

def load_artifacts(cfg):
    models_path = Path(cfg.output_paths['models'])
    model = joblib.load(models_path / "random_forest_model.pkl")
    le = joblib.load(models_path / "district_encoder.pkl")
    feature_cols = (models_path / "feature_names.txt").read_text().splitlines()
    print("✅ Model and encoder loaded")
    return model, le, feature_cols

def build_feature_row(target_dt, district, le, gold_features_path):
    """
    Robust feature generator for any future or unseen datetime.
    Always finds the nearest time-of-day sample for the district
    without causing indexer or matching errors.
    """
    gold_df = read_parquet(gold_features_path)

    # Filter by district
    district_df = gold_df[gold_df['District'] == district].copy()
    if district_df.empty:
        print(f"⚠ District '{district}' not found in gold dataset. Using first available district.")
        district_df = gold_df.copy()

    # Compute minutes-of-day difference, fall back safely
    target_min = target_dt.hour * 60 + target_dt.minute
    district_df['minutes_day'] = (
        district_df['datetime'].dt.hour * 60 + district_df['datetime'].dt.minute
    )
    idx = (district_df['minutes_day'] - target_min).abs().idxmin()
    if np.isnan(idx):
        idx = 0
    template = district_df.loc[idx].copy()

    # Update time-based features for requested timestamp
    template['datetime'] = pd.Timestamp(target_dt)
    template['Year'] = target_dt.year
    template['Month'] = target_dt.month
    template['Day'] = target_dt.day
    template['hour'] = target_dt.hour
    template['minute'] = target_dt.minute
    template['day_of_year'] = target_dt.timetuple().tm_yday
    template['day_of_week'] = target_dt.weekday()
    template['is_weekend'] = int(target_dt.weekday() >= 5)
    template['minute_of_day'] = target_min

    # Cyclic encodings
    template['hour_sin'] = np.sin(2 * np.pi * template['hour'] / 24)
    template['hour_cos'] = np.cos(2 * np.pi * template['hour'] / 24)
    template['doy_sin'] = np.sin(2 * np.pi * template['day_of_year'] / 365)
    template['doy_cos'] = np.cos(2 * np.pi * template['day_of_year'] / 365)

    # Solar elevation approximation
    latitude = 7.0
    solar_noon = 12.0
    hour_angle = (template['hour'] + template['minute'] / 60 - solar_noon) * 15
    day_angle = 2 * np.pi * (template['day_of_year'] - 1) / 365
    declination = 23.45 * np.sin(day_angle)
    elev = np.arcsin(
        np.sin(np.radians(latitude)) * np.sin(np.radians(declination))
        + np.cos(np.radians(latitude)) * np.cos(np.radians(declination))
        * np.cos(np.radians(hour_angle))
    )
    template['solar_elev_approx'] = np.degrees(elev).clip(0)

    # Encode district
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
    date = pd.to_datetime(date_str).date()
    times = pd.date_range(f"{date} 00:00", f"{date} 23:59", freq="5min")
    predictions = []
    for t in times:
        try:
            y_pred, interpolated = make_interpolated_prediction(
                model, le, feature_cols, district, str(t), gold_features_path
            )
            predictions.append({
                'datetime': t,
                'district': district,
                'prediction': y_pred,
                'interpolated': interpolated
            })
        except Exception as e:
            print(f"Error predicting for {t}: {e}")
    return pd.DataFrame(predictions)

def main():
    cfg = load_config()
    model, le, feature_cols = load_artifacts(cfg)
    gold_features_path = Path(cfg.data_paths['gold']) / "gold_features_all_years.parquet"
    print("\nWelcome to Solar Predictor!")
    print("To exit: type q at any prompt.")
    print("Available districts:", sorted(le.classes_))
    while True:
        district = input("\nEnter district: ")
        if district.lower() in {'q', 'quit', 'exit'}:
            break
        if district not in le.classes_:
            print("District not recognized. Available:", sorted(le.classes_))
            continue
        dt_str = input("Enter datetime (YYYY-MM-DD HH:MM): ")
        if dt_str.lower() in {'q', 'quit', 'exit'}:
            break
        try:
            y_pred, interpolated = make_interpolated_prediction(
                model, le, feature_cols, district, dt_str, gold_features_path
            )
            style = "[INTERPOLATED]" if interpolated else ""
            print(f"{style} Predicted generation for {district} at {dt_str}: {y_pred:.2f} kW")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
    cfg = load_config()
    model, le, feature_cols = load_artifacts(cfg)
    gold_features_path = Path(cfg.data_paths['gold']) / "gold_features_all_years.parquet"
    
    date_str = input("Enter date (YYYY-MM-DD): ")
    districts = le.classes_
    all_results = []
    for district in districts:
        df_pred = predict_day(model, le, feature_cols, district, date_str, gold_features_path)
        df_pred.to_csv(f"outputs/models/pred_day_{district}_{date_str}.csv", index=False)
        all_results.append(df_pred)
    all_df = pd.concat(all_results)
    all_df.to_csv(f"outputs/models/pred_all_{date_str}.csv", index=False)

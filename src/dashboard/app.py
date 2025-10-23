import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
import joblib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.config import load_config
from src.common.io_utils import read_parquet

st.set_page_config(page_title="Solar Energy Predictor", layout="wide")

# Load model + gold data ONCE (cached)
@st.cache_resource
def load_model_and_data():
    cfg = load_config()
    models_path = Path(cfg.output_paths['models'])
    model = joblib.load(models_path / "random_forest_model.pkl")
    le = joblib.load(models_path / "district_encoder.pkl")
    feature_cols = (models_path / "feature_names.txt").read_text().splitlines()
    gold_path = Path(cfg.data_paths['gold']) / "gold_features_all_years.parquet"
    gold_df = read_parquet(gold_path)
    return model, le, feature_cols, gold_df

model, le, feature_cols, gold_df = load_model_and_data()

def build_features_fast(times, district, le, gold_df, feature_cols):
    """Fast vectorized feature building."""
    district_data = gold_df[gold_df['District'] == district].copy()
    if district_data.empty:
        district_data = gold_df.copy()
    
    rows = []
    for t in times:
        target_min = t.hour * 60 + t.minute
        district_data['minutes_day'] = (
            district_data['datetime'].dt.hour * 60 + district_data['datetime'].dt.minute
        )
        idx = (district_data['minutes_day'] - target_min).abs().idxmin()
        template = district_data.loc[idx].copy()
        
        # Update time features
        template['datetime'] = pd.Timestamp(t)
        template['Year'] = t.year
        template['Month'] = t.month
        template['Day'] = t.day
        template['hour'] = t.hour
        template['minute'] = t.minute
        template['day_of_year'] = t.timetuple().tm_yday
        template['day_of_week'] = t.weekday()
        template['is_weekend'] = int(t.weekday() >= 5)
        template['minute_of_day'] = target_min
        
        # Cyclical
        template['hour_sin'] = np.sin(2 * np.pi * template['hour'] / 24)
        template['hour_cos'] = np.cos(2 * np.pi * template['hour'] / 24)
        template['doy_sin'] = np.sin(2 * np.pi * template['day_of_year'] / 365)
        template['doy_cos'] = np.cos(2 * np.pi * template['day_of_year'] / 365)
        
        # Solar elevation
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
        template['District_encoded'] = le.transform([district])[0]
        
        rows.append(template[feature_cols].values)
    
    return np.vstack(rows)

st.title("☀️ Solar Energy Generation Dashboard")
st.write("Generate predictions for any date with time resolution — **Fast & Interactive**")

col1, col2 = st.columns(2)
with col1:
    date_input = st.date_input("Select date", value=pd.to_datetime("2026-01-23"))
with col2:
    time_interval = st.selectbox(
        "Time resolution", 
        ["1 minute", "5 minutes", "15 minutes", "1 hour"]
    )

districts = sorted(le.classes_)
selected_districts = st.multiselect("Select districts", districts, default=districts[:3])

if st.button("⚡ Generate Predictions"):
    with st.spinner("Generating predictions..."):
        
        freq_map = {
            "1 minute": "1min",
            "5 minutes": "5min",
            "15 minutes": "15min",
            "1 hour": "1h"
        }
        freq = freq_map[time_interval]
        date_str = date_input.strftime("%Y-%m-%d")
        times = pd.date_range(f"{date_str} 00:00", f"{date_str} 23:59", freq=freq)
        
        all_predictions = []
        progress = st.progress(0)
        
        for idx, district in enumerate(selected_districts):
            X = build_features_fast(times, district, le, gold_df, feature_cols)
            preds = model.predict(X)
            
            df_district = pd.DataFrame({
                "datetime": times,
                "district": district,
                "prediction": preds
            })
            all_predictions.append(df_district)
            progress.progress((idx + 1) / len(selected_districts))
        
        df = pd.concat(all_predictions, ignore_index=True)
        
        fig = px.line(df, x="datetime", y="prediction", color="district",
                      title=f"Predicted Solar Generation – {date_str} ({time_interval} resolution)",
                      labels={"prediction":"Predicted kW"})
        st.plotly_chart(fig, use_container_width=True)
        
        summary = df.groupby("district")["prediction"].agg(["mean","max","min"]).reset_index()
        summary.columns = ["District", "Average (kW)", "Peak (kW)", "Minimum (kW)"]
        st.subheader("📊 District Summary")
        st.dataframe(summary.style.format(precision=2))
        
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Predictions as CSV", 
            csv, 
            f"predictions_{date_str}_{time_interval.replace(' ', '_')}.csv", 
            "text/csv"
        )
        
        st.success(f"✅ Generated {len(df):,} predictions for {len(selected_districts)} districts!")
        st.info(f"📊 Time resolution: **{time_interval}** ({len(times)} intervals per district)")


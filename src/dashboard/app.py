import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
import joblib
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.predict import build_feature_row
from src.common.config import load_config
from src.common.io_utils import read_parquet

st.set_page_config(page_title="Solar Energy Predictor", layout="wide")

# Load model artifacts (cached)
@st.cache_resource
def load_model():
    cfg = load_config()
    models_path = Path(cfg.output_paths['models'])
    model = joblib.load(models_path / "random_forest_model.pkl")
    le = joblib.load(models_path / "district_encoder.pkl")
    feature_cols = (models_path / "feature_names.txt").read_text().splitlines()
    gold_path = Path(cfg.data_paths['gold']) / "gold_features_all_years.parquet"
    return model, le, feature_cols, gold_path

model, le, feature_cols, gold_path = load_model()

st.title("☀️ Solar Energy Generation Dashboard")
st.write("Generate predictions for any date and time interval — **Fast & Interactive**")

# User inputs
col1, col2 = st.columns(2)
with col1:
    date_input = st.date_input("Select date", value=pd.to_datetime("2026-01-23"))
with col2:
    time_interval = st.selectbox("Time resolution", ["5 minutes", "15 minutes", "1 hour"])

districts = sorted(le.classes_)
selected_districts = st.multiselect("Select districts", districts, default=districts[:3])

if st.button("⚡ Generate Predictions"):
    with st.spinner("Generating predictions..."):
        
        # Map user choice to pandas frequency
        freq_map = {"5 minutes": "5min", "15 minutes": "15min", "1 hour": "1h"}
        freq = freq_map[time_interval]
        
        date_str = date_input.strftime("%Y-%m-%d")
        times = pd.date_range(f"{date_str} 00:00", f"{date_str} 23:59", freq=freq)
        
        all_predictions = []
        
        # Vectorized prediction per district (FAST!)
        for district in selected_districts:
            # Build feature matrix for all timestamps at once
            rows = []
            for t in times:
                row = build_feature_row(t, district, le, gold_path)
                rows.append(row[feature_cols].values)
            
            X = np.vstack(rows)  # Stack into one matrix
            preds = model.predict(X)  # Single batch prediction call
            
            df_district = pd.DataFrame({
                "datetime": times,
                "district": district,
                "prediction": preds
            })
            all_predictions.append(df_district)
        
        df = pd.concat(all_predictions, ignore_index=True)
        
        # Visualization
        fig = px.line(df, x="datetime", y="prediction", color="district",
                      title=f"Predicted Solar Generation – {date_str} ({time_interval} resolution)",
                      labels={"prediction":"Predicted kW"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary stats
        summary = df.groupby("district")["prediction"].agg(["mean","max","min"]).reset_index()
        summary.columns = ["District", "Average (kW)", "Peak (kW)", "Minimum (kW)"]
        st.subheader("📊 District Summary")
        st.dataframe(summary.style.format(precision=2))
        
        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name=f"predictions_{date_str}_{time_interval.replace(' ', '_')}.csv",
            mime="text/csv"
        )
        
        st.success(f"✅ Generated {len(df):,} predictions for {len(selected_districts)} districts in seconds!")

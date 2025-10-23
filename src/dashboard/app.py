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

st.set_page_config(page_title="Solar Energy Predictor", layout="wide")

# Load model artifacts
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
st.write("Generate predictions for any date and time interval.")

# User inputs
col1, col2 = st.columns(2)
with col1:
    date_input = st.date_input("Select date", value=pd.to_datetime("2026-01-23"))
with col2:
    time_interval = st.selectbox("Time resolution", ["1 minute", "5 minutes", "15 minutes", "1 hour"])

districts = sorted(le.classes_)
selected_districts = st.multiselect("Select districts", districts, default=districts[:3])

if st.button("Generate Predictions"):
    with st.spinner("Generating predictions..."):
        
        # Generate time range based on user selection
        freq_map = {"1 minute": "1min", "5 minutes": "5min", "15 minutes": "15min", "1 hour": "1h"}
        freq = freq_map[time_interval]
        
        date_str = date_input.strftime("%Y-%m-%d")
        times = pd.date_range(f"{date_str} 00:00", f"{date_str} 23:59", freq=freq)
        
        all_predictions = []
        
        for district in selected_districts:
            district_preds = []
            
            for t in times:
                # Build features and predict
                row = build_feature_row(t, district, le, gold_path)
                X = row[feature_cols].values.reshape(1, -1)
                pred = model.predict(X)[0]
                district_preds.append(pred)
            
            df_district = pd.DataFrame({
                "datetime": times,
                "district": district,
                "prediction": district_preds
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
        st.subheader("District Summary")
        st.dataframe(summary.style.format("{:.2f}"))
        
        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name=f"predictions_{date_str}_{time_interval.replace(' ', '_')}.csv",
            mime="text/csv"
        )
        
        st.success(f"✅ Generated {len(df):,} predictions for {len(selected_districts)} districts")

# Show existing pre-saved files
st.sidebar.header("Pre-saved Predictions")
output_dir = Path("outputs/models")
if output_dir.exists():
    files = sorted(output_dir.glob("pred_all_*.csv"))
    if files:
        for f in files:
            st.sidebar.write(f"📄 {f.name}")
    else:
        st.sidebar.write("No saved files yet")

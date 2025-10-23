import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Solar Energy Predictor", layout="wide")

st.title("☀️ Solar Energy Generation Dashboard")
st.write("Visualize predicted solar generation and download CSV outputs.")

# Date selector (look for available files)
output_dir = Path("outputs/models")
files = sorted(output_dir.glob("pred_all_*.csv"))
dates = [f.stem.replace("pred_all_", "") for f in files]

date_choice = st.selectbox("Select date", dates)

if date_choice:
    file_path = output_dir / f"pred_all_{date_choice}.csv"
    df = pd.read_csv(file_path)
    st.success(f"Loaded predictions for {date_choice}")

    districts = sorted(df["district"].unique())
    district_choice = st.multiselect("Select districts", districts, default=districts[:3])

    df_filtered = df[df["district"].isin(district_choice)]

    fig = px.line(df_filtered, x="datetime", y="prediction", color="district",
                  title=f"Predicted Solar Generation – {date_choice}",
                  labels={"prediction":"Predicted kW"})
    st.plotly_chart(fig, use_container_width=True)

    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download visible data as CSV", csv, f"predicted_{date_choice}.csv", "text/csv")

    avg_table = df_filtered.groupby("district")["prediction"].agg(["mean","max","min"]).reset_index()
    st.subheader("District Summary")
    st.dataframe(avg_table.style.format("{:.2f}"))

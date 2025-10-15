import pandas as pd
import numpy as np
import os

# === USER INPUT ===
INPUT_FILE = "Colombo_2018_Weather_PV.xlsx"   # weather Excel you attached
OUTPUT_FILE = "Colombo_2018_Weather_5Min_Wide.xlsx"

# which column names to use from your file
# adjust these if your file has different ones
datetime_col = "datetime"     # timestamp column name
temp_col = "temperature"      # temperature column
rh_col = "humidity"           # relative humidity
rain_col = "rainfall"         # rainfall
wind_col = "wind_speed"       # optional

# === LOAD AND CLEAN ===
df = pd.read_excel(INPUT_FILE)
df.columns = [c.strip().lower() for c in df.columns]  # normalize column names

# ensure datetime column exists
if datetime_col not in df.columns:
    datetime_col = df.columns[0]  # assume the first column is datetime

df[datetime_col] = pd.to_datetime(df[datetime_col])
df = df.sort_values(datetime_col).drop_duplicates(subset=[datetime_col])

# set datetime as index for resampling
df = df.set_index(datetime_col)

# resample to exact 5‑minute intervals
df_5min = df.resample("5T").ffill().reset_index()

# create TIME (00.00, 00.05 … etc.) and DATE columns
df_5min["Date"] = df_5min[datetime_col].dt.date
df_5min["Time"] = df_5min[datetime_col].dt.strftime("%H.%M")

# choose which weather variable to pivot — repeat for each separately if needed
value_columns = [col for col in df_5min.columns if col not in [datetime_col, "Date", "Time"]]
print("Detected weather variables:", value_columns)

# pivot each variable to wide (times as columns)
for value_col in value_columns:
    pivot = df_5min.pivot(index="Date", columns="Time", values=value_col)
    pivot = pivot.reindex(columns=sorted(pivot.columns, key=lambda t: float(t.replace('.', ''))))
    pivot.index.name = "Date"
    sheet_name = value_col.capitalize()
    with pd.ExcelWriter(OUTPUT_FILE, mode="a" if os.path.exists(OUTPUT_FILE) else "w") as writer:
        pivot.to_excel(writer, sheet_name=sheet_name)
    print(f"Saved sheet: {sheet_name}")

print(f"\n✅ Done! File saved as {OUTPUT_FILE}")

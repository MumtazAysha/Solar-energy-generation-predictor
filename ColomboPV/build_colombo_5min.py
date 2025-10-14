import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time, timezone
import pytz

# Settings
INPUT_XLSX = "Colombo_2018_Weather_PV.xlsx"
INPUT_SHEET = "Data_2018"
OUTPUT_XLSX = "Colombo_2018_5min_0800_1700.xlsx"
SITE_ID = "Colombo"
TZ_NAME = "Asia/Colombo"  # UTC+5:30
START_LOCAL = time(8, 0)
END_LOCAL = time(17, 0)
STEP_MIN = 5

PR = 0.80
POA_MULT = 1.05           # simple POA proxy: POA ≈ POA_MULT * GHI
TEMP_DERATE_PER_C = 0.004 # −0.4% per °C above 25°C
DERATE_REF_C = 25.0

def time_grid_for_day_local(day: pd.Timestamp, tz: pytz.timezone, start_t: time, end_t: time, step_min: int):
    start_dt = tz.localize(datetime.combine(day.date(), start_t))
    end_dt = tz.localize(datetime.combine(day.date(), end_t))
    # inclusive grid
    times = pd.date_range(start=start_dt, end=end_dt, freq=f"{step_min}min")
    return times

def smooth_bell(n):
    # symmetric bell-shaped weights (0..1..0), length n
    x = np.linspace(-1, 1, n)
    w = np.exp(-3 * x**2)
    return w / w.sum()

def shape_irradiance_from_daily(total_kwh_m2_day: float, n_steps: int):
    """Allocate daily kWh/m2 to 08:00–17:00 using a bell curve, output instantaneous W/m2 per 5-min step."""
    if total_kwh_m2_day is None or np.isnan(total_kwh_m2_day) or total_kwh_m2_day <= 0:
        return np.zeros(n_steps)
    # Assume a daytime share fraction for 08:00–17:00 relative to the whole day.
    # For tropics near Colombo, 08–17 covers most usable irradiance; use 0.82 as typical share.
    share = 0.82
    day_kwh_in_window = total_kwh_m2_day * share
    # 5-min interval hours
    dt_h = STEP_MIN / 60.0
    # Create shape weights
    w = smooth_bell(n_steps)
    # Convert kWh/m2 over window into average W/m2 profile per step
    # Energy per step (kWh/m2) = day_kwh_in_window * w_i
    step_kwh = day_kwh_in_window * w
    # Instantaneous W/m2 approx: (step_kWh / dt_h) * 1000 (kW -> W)
    w_per_m2 = (step_kwh / dt_h) * 1000.0
    return w_per_m2

def shape_temp_from_mean(mean_c: float, n_steps: int):
    if mean_c is None or np.isnan(mean_c):
        mean_c = 28.0
    # Diurnal amplitude ~3.0 C during 08–17
    amp = 3.0
    x = np.linspace(0, 2*np.pi, n_steps, endpoint=False)
    # peak around 14:00 ~ index 72 (assuming 108 steps)
    phase_shift = -np.pi/2
    vals = mean_c + amp * np.sin(x + phase_shift) * 0.7
    return vals

def shape_wind_from_mean(mean_ms: float, n_steps: int):
    if mean_ms is None or np.isnan(mean_ms):
        mean_ms = 3.0
    x = np.linspace(0, 2*np.pi, n_steps, endpoint=False)
    vals = mean_ms * (0.8 + 0.4 * np.sin(x - np.pi/3))
    vals = np.clip(vals, 0, None)
    return vals

def shape_cloud_proxy(daily_cloud_pct: float, n_steps: int):
    if daily_cloud_pct is None or np.isnan(daily_cloud_pct):
        daily_cloud_pct = 60.0
    x = np.linspace(0, 2*np.pi, n_steps, endpoint=False)
    vals = daily_cloud_pct + 10.0 * np.sin(x)
    return np.clip(vals, 0, 100)

def shape_rain_from_daily(rain_mm: float, n_steps: int):
    # allocate some daytime fraction; here ~40% inside 08–17 for rainy days
    if rain_mm is None or np.isnan(rain_mm) or rain_mm <= 0:
        return np.zeros(n_steps)
    day_fraction = 0.4
    mm_window = rain_mm * day_fraction
    vals = np.zeros(n_steps)
    # Make 1–2 short events
    rng = np.random.default_rng(42)  # deterministic
    events = 2 if mm_window > 10 else 1
    for _ in range(events):
        width = rng.integers(2, 7)  # 10–30 minutes
        start = rng.integers(0, n_steps - width)
        amount = mm_window / events
        vals[start:start+width] += amount / width
    return vals

def compute_pv_kw_1mwp(ghi_wm2: np.ndarray, temp_c: np.ndarray):
    poa = POA_MULT * ghi_wm2
    pdc_kw = (poa / 1000.0) * 1000.0  # 1 MWp DC
    derate = 1.0 - np.clip((temp_c - DERATE_REF_C), 0, None) * TEMP_DERATE_PER_C
    pac_kw = PR * pdc_kw * derate
    pac_kw = np.clip(pac_kw, 0, None)
    return pac_kw

def main():
    tz = pytz.timezone(TZ_NAME)
    df = pd.read_excel(INPUT_XLSX, sheet_name=INPUT_SHEET)
    # Expected columns: Date, District/Zone, GHI_kWh_m2_day, DNI_kWh_m2_day, DHI_kWh_m2_day, Mean_Temperature_C,
    # Wind_Speed_m_s, Cloud_Cover_percent, Rainfall_mm
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    rows = []
    for _, r in df.iterrows():
        day = pd.Timestamp(r["Date"])
        zone = r.get("Zone", "")
        ghi_day = r.get("GHI_kWh_m2_day", np.nan)
        dni_day = r.get("DNI_kWh_m2_day", np.nan)
        dhi_day = r.get("DHI_kWh_m2_day", np.nan)
        t_mean = r.get("Mean_Temperature_C", np.nan)
        w_mean = r.get("Wind_Speed_m_s", np.nan)
        cloud = r.get("Cloud_Cover_percent", np.nan)
        rain = r.get("Rainfall_mm", np.nan)

        times = time_grid_for_day_local(day, tz, START_LOCAL, END_LOCAL, STEP_MIN)
        n = len(times)

        ghi = shape_irradiance_from_daily(ghi_day, n)
        # Preserve beam/diffuse ratio using daily split; if absent, derive simple split
        if (dni_day is not None and not np.isnan(dni_day)) and (dhi_day is not None and not np.isnan(dhi_day)) and dni_day > 0 and dhi_day > 0:
            total = dni_day + dhi_day
            frac_dni = dni_day / total
        else:
            frac_dni = 0.55
        dni = ghi * frac_dni
        dhi = ghi * (1 - frac_dni)

        temp = shape_temp_from_mean(t_mean, n)
        wind = shape_wind_from_mean(w_mean, n)
        cloud_proxy = shape_cloud_proxy(cloud, n)
        rain_5m = shape_rain_from_daily(rain, n)

        pv_kw = compute_pv_kw_1mwp(ghi, temp)

        for i in range(n):
            rows.append({
                "timestamp_local": times[i].strftime("%Y-%m-%d %H:%M:%S"),
                "date": day.strftime("%Y-%m-%d"),
                "site_id": SITE_ID,
                "zone": zone,
                "GHI_Wm2": round(float(ghi[i]), 2),
                "DNI_Wm2": round(float(dni[i]), 2),
                "DHI_Wm2": round(float(dhi[i]), 2),
                "temp_C": round(float(temp[i]), 2),
                "wind_ms": round(float(wind[i]), 2),
                "cloud_pct_proxy": round(float(cloud_proxy[i]), 2),
                "rain_mm_per_5min": round(float(rain_5m[i]), 3),
                "PV_kW_1MWp": round(float(pv_kw[i]), 2),
            })

    out = pd.DataFrame(rows, columns=[
        "timestamp_local","date","site_id","zone","GHI_Wm2","DNI_Wm2","DHI_Wm2",
        "temp_C","wind_ms","cloud_pct_proxy","rain_mm_per_5min","PV_kW_1MWp"
    ])
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="FiveMin_2018", index=False)
    print(f"Wrote {len(out):,} rows to {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()

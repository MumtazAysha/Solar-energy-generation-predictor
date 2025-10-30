import pandas as pd
from src.common.io_utils import read_parquet

print("Loading gold_features_all_years.parquet...")
df = read_parquet("data/gold/gold_features_all_years.parquet")

print("\n✅ File loaded successfully!")
print(f"Total rows: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")

print("\n" + "="*70)
print("Checking Ampara data for 2022-01-01...")
print("="*70)

ampara_jan1 = df[
    (df['District'] == 'Ampara') & 
    (df['datetime'].dt.year == 2022) &
    (df['datetime'].dt.month == 1) &
    (df['datetime'].dt.day == 1)
]

print(f"\nFound {len(ampara_jan1)} records for Ampara on 2022-01-01")

if len(ampara_jan1) > 0:
    print("\n✅ SUCCESS! 2022-01-01 data exists!")
    print("\nFirst 15 records:")
    print(ampara_jan1[['datetime', 'target_kw']].head(15).to_string(index=False))
    
    # Check for 8 AM specifically
    am8 = ampara_jan1[ampara_jan1['datetime'].dt.hour == 8]
    if len(am8) > 0:
        print(f"\n🎯 Records at 8 AM: {len(am8)}")
        print("\n8 AM data:")
        print(am8[['datetime', 'target_kw']].head(5).to_string(index=False))
        
        # Check for 8:00 exactly
        am8_00 = ampara_jan1[ampara_jan1['datetime'] == pd.to_datetime('2022-01-01 08:00:00')]
        if len(am8_00) > 0:
            actual_value = am8_00['target_kw'].values[0]
            print(f"\n✅ 2022-01-01 08:00:00 - Actual value: {actual_value:.2f} kW")
            print("   (Expected: ~838.56 kW from your Excel file)")
else:
    print("\n❌ No data found!")

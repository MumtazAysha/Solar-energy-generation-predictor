import pandas as pd
from src.common.io_utils import read_parquet

# Load the gold features file
print("Loading gold_features_all_years.parquet...")
df = read_parquet("data/gold/gold_features_all_years.parquet")

print(f"\n✅ File loaded successfully!")
print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# Filter for Ampara, 2022-01-01
print("\n" + "="*70)
print("Checking Ampara data for 2022-01-01...")
print("="*70)

ampara_jan1 = df[
    (df['District'] == 'Ampara') & 
    (df['datetime'].dt.date == pd.to_datetime('2022-01-01').date())
].copy()

print(f"\nFound {len(ampara_jan1)} records for Ampara on 2022-01-01")

if len(ampara_jan1) > 0:
    print("\nFirst 15 records (datetime and Generation_kW):")
    print(ampara_jan1[['datetime', 'Generation_kW']].head(15).to_string(index=False))
    
    print("\n" + "="*70)
    print("Looking for your specific value: 838.5641 kW")
    print("="*70)
    
    # Search for exact match (with rounding tolerance)
    exact_match = ampara_jan1[
        (ampara_jan1['Generation_kW'] >= 838.56) & 
        (ampara_jan1['Generation_kW'] <= 838.57)
    ]
    
    if len(exact_match) > 0:
        print(f"\n✅ FOUND! The value 838.5641 exists at:")
        print(exact_match[['datetime', 'Generation_kW']].to_string(index=False))
    else:
        print(f"\n❌ Value 838.5641 NOT found in gold features for 2022-01-01")
        print(f"\nClosest values:")
        closest = ampara_jan1.iloc[(ampara_jan1['Generation_kW'] - 838.5641).abs().argsort()[:5]]
        print(closest[['datetime', 'Generation_kW']].to_string(index=False))
    
    print("\n" + "="*70)
    print("Summary statistics for this day:")
    print("="*70)
    print(f"Min generation: {ampara_jan1['Generation_kW'].min():.2f} kW")
    print(f"Max generation: {ampara_jan1['Generation_kW'].max():.2f} kW")
    print(f"Mean generation: {ampara_jan1['Generation_kW'].mean():.2f} kW")
else:
    print("❌ No data found for Ampara on 2022-01-01!")
    print("\nChecking what dates ARE available for Ampara...")
    ampara_all = df[df['District'] == 'Ampara']
    print(f"Date range: {ampara_all['datetime'].min()} to {ampara_all['datetime'].max()}")


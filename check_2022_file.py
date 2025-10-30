import pandas as pd

# Correct filename with spaces
file_path = "data/raw/Annual Generation data in Ampara 2022.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

print("Columns:", df.columns.tolist())
print("\nFirst 10 rows:")
print(df.head(10))

if 'Year' in df.columns:
    print("\nYear column unique values:")
    print(df['Year'].value_counts())
elif 'Date' in df.columns:
    print("\nDate column sample:")
    print(df['Date'].head(10))
    print("\nYear extracted from Date:")
    print(pd.to_datetime(df['Date'], errors='coerce').dt.year.value_counts())

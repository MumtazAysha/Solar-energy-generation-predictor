import pandas as pd
import os
import glob
import warnings
import numpy as np
import calendar

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def convert_to_2022_format(input_file, output_file, year=None):
    """
    Convert to 2022 format with continuous day numbering and correct summary calculations
    
    Parameters:
    input_file: Path to input file
    output_file: Path to output file
    year: Year of the data (to determine leap year for correct day count)
    """
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Extract year from filename if not provided
    if year is None:
        import re
        year_match = re.search(r'20\d{2}', os.path.basename(input_file))
        if year_match:
            year = int(year_match.group())
        else:
            year = 2018  # Default fallback
    
    xl = pd.ExcelFile(input_file)
    month_sheets = [sheet for sheet in xl.sheet_names if sheet.lower() != 'readme']
    
    print(f"Processing {len(month_sheets)} sheets from {os.path.basename(input_file)}...")
    print(f"  Year: {year} (Leap year: {calendar.isleap(year)})")
    
    all_data = []
    day_counter = 1
    
    for sheet in month_sheets:
        print(f"  Processing {sheet}...")
        df_month = pd.read_excel(input_file, sheet_name=sheet)
        
        # Process time columns efficiently
        metadata_cols = ['Date', 'Month', 'Day']
        time_data = {}
        
        for col in df_month.columns:
            if col in metadata_cols:
                continue
                
            if isinstance(col, str) and ':' in col:
                try:
                    hour, minute = col.split(':')
                    decimal_time = int(hour) + int(minute) / 60
                    decimal_time = round(decimal_time, 2)
                    time_data[decimal_time] = df_month[col].values
                except:
                    pass
            else:
                if col not in metadata_cols:
                    time_data[col] = df_month[col].values
        
        result_df = pd.DataFrame(time_data)
        
        # Create continuous day numbering
        num_days = len(df_month)
        continuous_days = list(range(day_counter, day_counter + num_days))
        result_df.insert(0, 'Date', continuous_days)
        
        day_counter += num_days
        all_data.append(result_df)
    
    # Concatenate all months
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Sort columns: Date first, then time columns in order
    time_columns = sorted([col for col in final_df.columns if col != 'Date'])
    final_df = final_df[['Date'] + time_columns]
    
    # === ADD SUMMARY ROWS WITH CORRECT FORMULAS ===
    print("  Calculating summary rows...")
    
    # Constants
    IRRADIANCE_CONSTANT = 5.25  # Sri Lanka's irradiance constant
    num_days = len(final_df)
    
    # Get only numeric columns (time columns)
    numeric_cols = [col for col in final_df.columns if col != 'Date']
    
    # Calculate summary rows
    summary_rows = []
    
    # Row 1: Total kW = SUM(all values) / 1000
    total_kw_row = {'Date': 'Total kW'}
    for col in numeric_cols:
        total_kw_row[col] = final_df[col].sum() / 1000
    summary_rows.append(total_kw_row)
    
    # Row 2: Daily Average per day per kW = Total kW / (5.25 × num_days)
    daily_avg_row = {'Date': 'Daily Avarage per day per kW'}  # Keep typo as in 2022
    for col in numeric_cols:
        daily_avg_row[col] = total_kw_row[col] / (IRRADIANCE_CONSTANT * num_days)
    summary_rows.append(daily_avg_row)
    
    # Row 3: Energy kWh = Daily Average × (5/60)
    energy_kwh_row = {'Date': 'Energy kWh'}
    for col in numeric_cols:
        energy_kwh_row[col] = daily_avg_row[col] * (5 / 60)
    summary_rows.append(energy_kwh_row)
    
    # Append summary rows to dataframe
    summary_df = pd.DataFrame(summary_rows)
    final_df = pd.concat([final_df, summary_df], ignore_index=True)
    
    print(f"✓ Conversion complete!")
    print(f"  Final shape: {final_df.shape[0]} rows × {final_df.shape[1]} columns")
    print(f"  Day range: Day 1 to Day {num_days}")
    print(f"  Summary calculations:")
    print(f"    - Total kW = SUM / 1000")
    print(f"    - Daily Average = Total kW / ({IRRADIANCE_CONSTANT} × {num_days})")
    print(f"    - Energy kWh = Daily Average × (5/60)")
    
    # Ensure output file has .xlsx extension
    if not output_file.endswith('.xlsx'):
        output_file = output_file.rsplit('.', 1)[0] + '.xlsx'
    
    # Save
    try:
        final_df.to_excel(output_file, index=False, sheet_name='Sheet1', engine='openpyxl')
        print(f"  Saved to: {output_file}\n")
        return True
    except Exception as e:
        print(f"  ERROR saving file: {str(e)}\n")
        return False


def batch_convert_all_files(raw_folder, output_folder, target_year=2022):
    """Batch convert with correct solar energy formulas"""
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all Excel files
    excel_files = glob.glob(os.path.join(raw_folder, '*.xlsx')) + glob.glob(os.path.join(raw_folder, '*.xls'))
    
    print(f"Found {len(excel_files)} Excel files in {raw_folder}\n")
    print("="*70)
    
    converted_count = 0
    skipped_count = 0
    error_count = 0
    
    for file_path in excel_files:
        filename = os.path.basename(file_path)
        
        # Skip 2022 files
        if str(target_year) in filename:
            print(f"⊘ Skipping {filename} (already in {target_year} format)")
            skipped_count += 1
            continue
        
        # Skip already converted files
        if 'CONVERTED' in filename:
            print(f"⊘ Skipping {filename} (already converted)")
            skipped_count += 1
            continue
        
        # Create output filename
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}-CONVERTED.xlsx"
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"\n{'='*70}")
        print(f"Converting: {filename}")
        print(f"Output: {output_filename}")
        print('='*70)
        
        try:
            success = convert_to_2022_format(file_path, output_path)
            if success:
                converted_count += 1
                print(f"✓ Successfully saved: {output_filename}")
            else:
                error_count += 1
        except Exception as e:
            print(f"✗ Error: {str(e)}\n")
            import traceback
            traceback.print_exc()
            error_count += 1
    
    print("\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"✓ Successfully converted: {converted_count} files")
    print(f"⊘ Skipped: {skipped_count} files")
    print(f"✗ Errors: {error_count} files")
    print(f"Total processed: {len(excel_files)} files")
    print(f"\nConverted files saved to: {output_folder}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_folder = os.path.join(project_root, 'data', 'raw')
    output_folder = os.path.join(project_root, 'data', 'processed')
    
    print("="*70)
    print("SOLAR ENERGY DATA CONVERTER - Sri Lanka")
    print("="*70)
    print("Features:")
    print("  • Continuous day numbering (1-365 or 1-366)")
    print("  • Correct solar energy calculations:")
    print("    - Total kW = SUM / 1000")
    print("    - Daily Average = Total kW / (5.25 × days)")
    print("    - Energy kWh = Daily Average × (5/60)")
    print("  • Irradiance constant: 5.25 (Sri Lanka)")
    print("="*70)
    print(f"Input folder:  {raw_folder}")
    print(f"Output folder: {output_folder}\n")
    
    batch_convert_all_files(raw_folder, output_folder)

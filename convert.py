import pandas as pd
import os
import glob
import warnings
from datetime import datetime

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def convert_to_2022_format(input_file, output_file):
    """Convert to 2022 format with continuous day numbering (1-365)"""
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    xl = pd.ExcelFile(input_file)
    month_sheets = [sheet for sheet in xl.sheet_names if sheet.lower() != 'readme']
    
    print(f"Processing {len(month_sheets)} sheets from {os.path.basename(input_file)}...")
    
    all_data = []
    day_counter = 1  # CRITICAL: Start continuous day counter
    
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
        
        # CRITICAL FIX: Create continuous day numbering
        num_days = len(df_month)
        continuous_days = list(range(day_counter, day_counter + num_days))
        result_df.insert(0, 'Date', continuous_days)
        
        # Update counter for next month
        day_counter += num_days
        
        all_data.append(result_df)
    
    # Concatenate all months
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Sort columns: Date first, then time columns in order
    time_columns = sorted([col for col in final_df.columns if col != 'Date'])
    final_df = final_df[['Date'] + time_columns]
    
    print(f"✓ Conversion complete!")
    print(f"  Final shape: {final_df.shape[0]} rows × {final_df.shape[1]} columns")
    print(f"  Day range: Day {final_df['Date'].min()} to Day {final_df['Date'].max()}")
    
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
    """Batch convert with continuous day numbering"""
    
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
    print("SOLAR ENERGY DATA CONVERTER")
    print("Converting to 2022 format with continuous day numbering (1-365/366)")
    print("="*70)
    print(f"Input folder:  {raw_folder}")
    print(f"Output folder: {output_folder}\n")
    
    batch_convert_all_files(raw_folder, output_folder)

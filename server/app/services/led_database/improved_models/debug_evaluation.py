"""
Debug script for model_evaluation module
"""
import os
import sys
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

try:
    from model_evaluation import load_data, preprocess_data
    print("Successfully imported model_evaluation functions")
except Exception as e:
    print(f"Error importing model_evaluation: {e}")
    sys.exit(1)

def debug_data_loading():
    """Test data loading function"""
    print("\n=== Testing data loading ===")
    try:
        hyomen_file = os.path.join(parent_dir, "predict_database.tsv")
        neon_file = os.path.join(parent_dir, "predict_database_neon.tsv")
        
        print(f"Loading surface emission data from: {hyomen_file}")
        df_hyomen = load_data(hyomen_file)
        print(f"Surface emission data loaded: {df_hyomen.shape} rows")
        print(f"Columns: {df_hyomen.columns.tolist()}")
        
        print("\nDistance column sample (first 3 rows):")
        for i, val in enumerate(df_hyomen['distance'].head(3)):
            print(f"Row {i}, type: {type(val)}, value: {val[:100]}..." if isinstance(val, str) else f"Row {i}, type: {type(val)}, value: {val}")
        
        print("\nLoading neon emission data from: {neon_file}")
        df_neon = load_data(neon_file, is_neon=True)
        print(f"Neon emission data loaded: {df_neon.shape} rows")
        
        return df_hyomen, df_neon
    except Exception as e:
        print(f"Error in data loading: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def debug_preprocessing(df_hyomen, df_neon):
    """Test preprocessing function"""
    print("\n=== Testing preprocessing ===")
    try:
        print("Preprocessing surface emission data...")
        df_hyomen_processed = preprocess_data(df_hyomen)
        print(f"Processed surface emission data: {df_hyomen_processed.shape} rows")
        print(f"Columns after preprocessing: {df_hyomen_processed.columns.tolist()}")
        
        distance_cols = [col for col in df_hyomen_processed.columns if 'distance' in col]
        print(f"\nDistance-derived columns: {distance_cols}")
        
        print("\nNaN values in key columns:")
        for col in ['Area', 'Peri', 'skeleton_length', 'led'] + distance_cols[:5]:
            if col in df_hyomen_processed.columns:
                nan_count = df_hyomen_processed[col].isna().sum()
                print(f"  {col}: {nan_count} NaNs ({nan_count/len(df_hyomen_processed)*100:.2f}%)")
        
        print("\nPreprocessing neon emission data...")
        df_neon_processed = preprocess_data(df_neon, is_neon=True)
        print(f"Processed neon emission data: {df_neon_processed.shape} rows")
        
        return df_hyomen_processed, df_neon_processed
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main function"""
    print("Starting debug script for model_evaluation module")
    
    df_hyomen, df_neon = debug_data_loading()
    if df_hyomen is None:
        print("Data loading failed, exiting")
        return
    
    df_hyomen_processed, df_neon_processed = debug_preprocessing(df_hyomen, df_neon)
    if df_hyomen_processed is None:
        print("Preprocessing failed, exiting")
        return
    
    print("\n=== Debug completed successfully ===")
    print("The model_evaluation module is working correctly for data loading and preprocessing")

if __name__ == "__main__":
    main()

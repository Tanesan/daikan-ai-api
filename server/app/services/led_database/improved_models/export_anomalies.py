"""
Export anomaly records to CSV file
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from model_evaluation import load_data, preprocess_data
from data_quality import detect_and_handle_outliers
from enhanced_features import enhance_skeleton_topology

def export_anomaly_records():
    """
    Export anomaly records to CSV file
    """
    print("Loading data...")
    
    hyomen_file = os.path.join(parent_dir, "predict_database.tsv")
    df_hyomen = load_data(hyomen_file)
    print(f"Surface emission data loaded: {df_hyomen.shape[0]} records")
    
    neon_file = os.path.join(parent_dir, "predict_database_neon.tsv")
    if os.path.exists(neon_file):
        df_neon = load_data(neon_file)
        print(f"Neon data loaded: {df_neon.shape[0]} records")
        df_combined = pd.concat([df_hyomen, df_neon], ignore_index=True)
        df_combined['emission_type'] = np.where(
            df_combined.index < len(df_hyomen), 'surface', 'neon'
        )
    else:
        df_combined = df_hyomen.copy()
        df_combined['emission_type'] = 'surface'
    
    print(f"Combined data: {df_combined.shape[0]} records")
    
    df_processed = preprocess_data(df_combined)
    print(f"Preprocessed data: {df_processed.shape[0]} records")
    
    df_enhanced = enhance_skeleton_topology(df_processed)
    
    contamination = 0.05  # 5% of data as anomalies
    df_with_anomalies = detect_and_handle_outliers(df_enhanced, contamination=contamination)
    
    anomaly_records = df_with_anomalies[df_with_anomalies['anomaly'] == 1].copy()
    print(f"Detected {anomaly_records.shape[0]} anomaly records ({anomaly_records.shape[0]/df_with_anomalies.shape[0]*100:.2f}%)")
    
    anomaly_records = add_anomaly_reasons(anomaly_records, df_with_anomalies)
    
    output_dir = os.path.join(current_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"anomaly_records_{timestamp}.csv")
    
    anomaly_records.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Anomaly records exported to: {output_file}")
    
    return output_file

def add_anomaly_reasons(anomaly_df, full_df):
    """
    Add columns indicating likely reasons for anomaly detection
    """
    stats = {}
    numeric_cols = full_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col not in ['anomaly', 'pj', 'index']:
            stats[col] = {
                'mean': full_df[col].mean(),
                'std': full_df[col].std(),
                'q1': full_df[col].quantile(0.25),
                'q3': full_df[col].quantile(0.75),
                'iqr': full_df[col].quantile(0.75) - full_df[col].quantile(0.25)
            }
    
    for col in ['Area', 'Peri', 'skeleton_length', 'led']:
        if col in anomaly_df.columns:
            lower_bound = stats[col]['q1'] - 1.5 * stats[col]['iqr']
            upper_bound = stats[col]['q3'] + 1.5 * stats[col]['iqr']
            
            anomaly_df[f'{col}_is_outlier'] = (
                (anomaly_df[col] < lower_bound) | 
                (anomaly_df[col] > upper_bound)
            )
            
            anomaly_df[f'{col}_zscore'] = (anomaly_df[col] - stats[col]['mean']) / stats[col]['std']
    
    if 'Area' in anomaly_df.columns and 'skeleton_length' in anomaly_df.columns:
        full_ratio = full_df['Area'] / (full_df['skeleton_length'] + 1)
        ratio_mean = full_ratio.mean()
        ratio_std = full_ratio.std()
        
        anomaly_df['area_skeleton_ratio'] = anomaly_df['Area'] / (anomaly_df['skeleton_length'] + 1)
        anomaly_df['area_skeleton_ratio_zscore'] = (
            (anomaly_df['area_skeleton_ratio'] - ratio_mean) / ratio_std
        )
        anomaly_df['unusual_area_skeleton_ratio'] = abs(anomaly_df['area_skeleton_ratio_zscore']) > 2
    
    if 'led' in anomaly_df.columns and 'Area' in anomaly_df.columns:
        full_density = full_df['led'] / (full_df['Area'] + 1)
        density_mean = full_density.mean()
        density_std = full_density.std()
        
        anomaly_df['led_density'] = anomaly_df['led'] / (anomaly_df['Area'] + 1)
        anomaly_df['led_density_zscore'] = (
            (anomaly_df['led_density'] - density_mean) / density_std
        )
        anomaly_df['unusual_led_density'] = abs(anomaly_df['led_density_zscore']) > 2
    
    if 'branch_points' in anomaly_df.columns and 'endpoint' in anomaly_df.columns:
        anomaly_df['branch_endpoint_ratio'] = anomaly_df['branch_points'] / (anomaly_df['endpoint'] + 1)
        
        full_ratio = full_df['branch_points'] / (full_df['endpoint'] + 1)
        ratio_mean = full_ratio.mean()
        ratio_std = full_ratio.std()
        
        anomaly_df['branch_endpoint_ratio_zscore'] = (
            (anomaly_df['branch_endpoint_ratio'] - ratio_mean) / ratio_std
        )
        anomaly_df['unusual_branch_structure'] = abs(anomaly_df['branch_endpoint_ratio_zscore']) > 2
    
    anomaly_df['primary_anomaly_reason'] = 'Multiple factors'
    
    for record_idx in anomaly_df.index:
        max_zscore = 0
        max_reason = 'Multiple factors'
        
        for col in anomaly_df.columns:
            if col.endswith('_zscore'):
                abs_zscore = abs(anomaly_df.loc[record_idx, col])
                if abs_zscore > max_zscore:
                    max_zscore = abs_zscore
                    base_col = col.replace('_zscore', '')
                    max_reason = f'Unusual {base_col} (z-score: {anomaly_df.loc[record_idx, col]:.2f})'
        
        anomaly_df.loc[record_idx, 'primary_anomaly_reason'] = max_reason
    
    return anomaly_df

if __name__ == "__main__":
    export_anomaly_records()

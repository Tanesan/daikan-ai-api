import pandas as pd
import numpy as np
import os
import sys
from ast import literal_eval

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from data_segmentation import segment_by_size_and_complexity
from enhanced_features import enhance_features_for_large_signs
from model_builder import train_all_models
from column_definitions import standardize_dataframe

def preprocess_data(df):
    """
    データの前処理
    """
    if 'distance' in df.columns:
        df['distance'] = df['distance'].apply(
            lambda x: literal_eval(x) if isinstance(x, str) else x
        )
        
        df['distance_average'] = df['distance'].apply(
            lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan
        )
        df['distance_min'] = df['distance'].apply(
            lambda x: min(x) if isinstance(x, list) and len(x) > 0 else np.nan
        )
        df['distance_max'] = df['distance'].apply(
            lambda x: max(x) if isinstance(x, list) and len(x) > 0 else np.nan
        )
    
    if 'zunguri' not in df.columns:
        df['zunguri'] = df['Area'] / (df['Peri'] + 1e-5)
    
    return df

def main():
    hyomen_file = "../predict_database.tsv"
    neon_file = "../predict_database_neon.tsv"
    
    print(f"表面発光データの読み込み: {hyomen_file}")
    df_hyomen = pd.read_csv(hyomen_file, sep='\t')
    
    print(f"ネオン発光データの読み込み: {neon_file}")
    df_neon = pd.read_csv(neon_file, sep='\t')
    
    df_all = pd.concat([df_hyomen, df_neon])
    print(f"全データ数: {len(df_all)}")
    
    df_all = preprocess_data(df_all)
    
    df_all = segment_by_size_and_complexity(df_all)
    
    df_all = enhance_features_for_large_signs(df_all)
    
    df_all = standardize_dataframe(df_all, for_training=True)
    
    segment_counts = df_all['segment'].value_counts()
    print("\n各セグメントのデータ数:")
    print(segment_counts)
    
    model_results = train_all_models(df_all)
    
    print("\nモデルトレーニング完了")
    print("\n各モデルのMAPE:")
    print(model_results[['segment', 'mape']].sort_values('mape'))

if __name__ == "__main__":
    main()

"""
改良されたモデルを保存するスクリプト
"""
import os
import sys
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from enhanced_features import enhance_distance_features, enhance_geometric_features, custom_led_density_features, detect_potential_outliers, enhance_skeleton_topology
from data_segmentation import advanced_segmentation, get_segment_models_mapping, get_specific_models_for_large_signs
from model_architecture import build_ensemble_model
from large_signs_handler import enhance_large_sign_features, build_specialized_large_sign_model
from model_evaluation import load_data, preprocess_data, train_and_evaluate_models, evaluate_models

def train_and_save_models():
    """
    改良されたモデルを訓練して保存する関数
    """
    print("データをロード中...")
    
    hyomen_file = os.path.join(parent_dir, "predict_database.tsv")
    df_hyomen = load_data(hyomen_file)
    print(f"表面発光データをロード: {df_hyomen.shape[0]}件")
    
    neon_file = os.path.join(parent_dir, "predict_database_neon.tsv")
    if os.path.exists(neon_file):
        df_neon = load_data(neon_file)
        print(f"ネオンデータをロード: {df_neon.shape[0]}件")
        df_combined = pd.concat([df_hyomen, df_neon], ignore_index=True)
        df_combined['emission_type'] = np.where(
            df_combined.index < len(df_hyomen), 'surface', 'neon'
        )
    else:
        df_combined = df_hyomen.copy()
        df_combined['emission_type'] = 'surface'
    
    print(f"データの合計: {df_combined.shape[0]}件")
    
    df_processed = preprocess_data(df_combined)
    
    df_enhanced = df_processed.copy()
    df_enhanced = enhance_distance_features(df_enhanced)
    df_enhanced = enhance_geometric_features(df_enhanced)
    df_enhanced = custom_led_density_features(df_enhanced)
    df_enhanced = detect_potential_outliers(df_enhanced)
    
    df_enhanced = enhance_skeleton_topology(df_enhanced)
    
    df_enhanced, _, _ = advanced_segmentation(df_enhanced)
    df_enhanced = get_segment_models_mapping(df_enhanced)
    df_enhanced = get_specific_models_for_large_signs(df_enhanced)
    
    segments = {}
    if 'large_segment' in df_enhanced.columns:
        segments['large_signs'] = df_enhanced[df_enhanced['large_segment'].notna()].copy()
    
    segments['normal_signs'] = df_enhanced[~df_enhanced.index.isin(segments.get('large_signs', pd.DataFrame()).index)].copy()
    
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    if 'large_signs' in segments and not segments['large_signs'].empty:
        print("\n大型看板モデルを訓練中...")
        large_df = enhance_large_sign_features(segments['large_signs'])
        large_model, feature_importance = build_specialized_large_sign_model(large_df)
        
        large_model_path = os.path.join(models_dir, "large_sign_model.json")
        large_model.save_model(large_model_path)
        print(f"大型看板モデルを保存: {large_model_path}")
        
        feature_importance.to_csv(os.path.join(models_dir, "large_sign_feature_importance.csv"), index=False)
    
    if 'normal_signs' in segments and not segments['normal_signs'].empty:
        print("\n通常サイズのアンサンブルモデルを訓練中...")
        normal_df = segments['normal_signs']
        
        X = normal_df[[col for col in normal_df.columns if col != 'led' and col != 'index']]
        y = normal_df['led']
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        ensemble_model, _ = build_ensemble_model(X_train, y_train, X_val, y_val)
        
        ensemble_model_path = os.path.join(models_dir, "ensemble_model.pkl")
        joblib.dump(ensemble_model, ensemble_model_path)
        print(f"アンサンブルモデルを保存: {ensemble_model_path}")
    
    print("\nすべてのモデルの保存が完了しました")
    return models_dir

if __name__ == "__main__":
    train_and_save_models()

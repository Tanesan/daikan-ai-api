"""
大型看板の判定基準パラメータを様々に変更して精度を評価するスクリプト
"""
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from enhanced_features import enhance_distance_features, enhance_geometric_features, custom_led_density_features, detect_potential_outliers, enhance_skeleton_topology
from data_segmentation import advanced_segmentation, get_segment_models_mapping, get_specific_models_for_large_signs
from model_architecture import build_ensemble_model
from large_signs_handler import enhance_large_sign_features

def load_data(file_path):
    """
    データをロードする関数
    """
    print(f"データをロード中: {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    
    if 'distance' in df.columns:
        try:
            df['distance'] = df['distance'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        except:
            print("distanceカラムの変換に失敗しました。スキップします。")
    
    return df

def calculate_mape(y_true, y_pred):
    """
    平均絶対パーセント誤差（MAPE）を計算する関数
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_with_parameters(df, area_threshold, zunguri_threshold, area_only_threshold):
    """
    指定されたパラメータで評価を行う関数
    """
    is_huge = ((df['Area'] > area_threshold) & (df['zunguri'] >= zunguri_threshold)) | (df['Area'] > area_only_threshold)
    
    df_huge = df[is_huge].copy()
    df_normal = df[~is_huge].copy()
    
    y_pred = np.zeros(len(df))
    
    large_model_path = os.path.join(current_dir, 'models', 'large_sign_model.json')
    
    ensemble_model_path = os.path.join(current_dir, 'models', 'ensemble_model.pkl')
    
    if not df_huge.empty and os.path.exists(large_model_path):
        large_df = enhance_large_sign_features(df_huge)
        model = xgb.Booster()
        model.load_model(large_model_path)
        
        X_large = large_df[[col for col in large_df.columns if col != 'led' and col != 'index']]
        dmatrix = xgb.DMatrix(X_large)
        large_preds = model.predict(dmatrix)
        
        for i, idx in enumerate(df_huge.index):
            y_pred[idx] = round(large_preds[i])
    
    if not df_normal.empty and os.path.exists(ensemble_model_path):
        normal_df = df_normal.copy()
        model = joblib.load(ensemble_model_path)
        
        X_normal = normal_df[[col for col in normal_df.columns if col != 'led' and col != 'index']]
        normal_preds = model.predict(X_normal)
        
        for i, idx in enumerate(df_normal.index):
            y_pred[idx] = round(normal_preds[i])
    
    y_true = df['led']
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    huge_mape = calculate_mape(df_huge['led'], y_pred[df_huge.index]) if not df_huge.empty else 0
    normal_mape = calculate_mape(df_normal['led'], y_pred[df_normal.index]) if not df_normal.empty else 0
    
    return {
        'area_threshold': area_threshold,
        'zunguri_threshold': zunguri_threshold,
        'area_only_threshold': area_only_threshold,
        'huge_count': len(df_huge),
        'normal_count': len(df_normal),
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'huge_mape': huge_mape,
        'normal_mape': normal_mape
    }

def test_multiple_parameters():
    """
    複数のパラメータセットでテストを実行する関数
    """
    hyomen_file = os.path.join(parent_dir, "predict_database.tsv")
    df_hyomen = load_data(hyomen_file)
    print(f"表面発光データをロード: {df_hyomen.shape[0]}件")
    
    df_enhanced = df_hyomen.copy()
    df_enhanced = enhance_distance_features(df_enhanced)
    df_enhanced = enhance_geometric_features(df_enhanced)
    df_enhanced = custom_led_density_features(df_enhanced)
    df_enhanced = detect_potential_outliers(df_enhanced)
    df_enhanced = enhance_skeleton_topology(df_enhanced)
    
    parameter_sets = [
        {'area': 25000, 'zunguri': 20, 'area_only': 50000},
        {'area': 20000, 'zunguri': 15, 'area_only': 40000},
        {'area': 18000, 'zunguri': 15, 'area_only': 35000},
        {'area': 15000, 'zunguri': 12, 'area_only': 30000},
        {'area': 22000, 'zunguri': 18, 'area_only': 45000},
        {'area': 20000, 'zunguri': 12, 'area_only': 40000},
        {'area': 15000, 'zunguri': 10, 'area_only': 35000},
    ]
    
    results = []
    
    for params in parameter_sets:
        print(f"\nパラメータセットをテスト中: Area > {params['area']} & zunguri >= {params['zunguri']} | Area > {params['area_only']}")
        result = evaluate_with_parameters(
            df_enhanced, 
            params['area'], 
            params['zunguri'], 
            params['area_only']
        )
        results.append(result)
        
        print(f"大型看板: {result['huge_count']}件, 通常サイズ: {result['normal_count']}件")
        print(f"全体MAPE: {result['mape']:.2f}%, 大型看板MAPE: {result['huge_mape']:.2f}%, 通常サイズMAPE: {result['normal_mape']:.2f}%")
    
    results_df = pd.DataFrame(results)
    
    results_df = results_df.sort_values('mape')
    
    print("\n=== パラメータテスト結果 (MAPE順) ===")
    print(results_df[['area_threshold', 'zunguri_threshold', 'area_only_threshold', 
                      'huge_count', 'normal_count', 'mape', 'huge_mape', 'normal_mape']])
    
    best_params = results_df.iloc[0]
    print(f"\n最適なパラメータセット:")
    print(f"Area > {best_params['area_threshold']} & zunguri >= {best_params['zunguri_threshold']} | Area > {best_params['area_only_threshold']}")
    print(f"全体MAPE: {best_params['mape']:.2f}%")
    
    return results_df

if __name__ == "__main__":
    print("大型看板の判定基準パラメータテストを開始します...")
    results = test_multiple_parameters()
    
    output_path = os.path.join(current_dir, 'parameter_test_results.csv')
    results.to_csv(output_path, index=False)
    print(f"\n結果を保存しました: {output_path}")

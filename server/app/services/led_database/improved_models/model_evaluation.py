import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import shap
import os
import sys

import enhanced_features
import data_segmentation
import model_architecture
import data_quality
import large_signs_handler

from enhanced_features import enhance_distance_features, enhance_geometric_features, custom_led_density_features
from data_segmentation import advanced_segmentation, get_segment_models_mapping, get_specific_models_for_large_signs
from model_architecture import build_segment_model, build_ensemble_model, weighted_rmse_obj, weighted_rmse_eval
from data_quality import detect_and_handle_outliers, create_robust_features, handle_missing_values
from large_signs_handler import enhance_large_sign_features, split_large_signs_by_complexity, build_specialized_large_sign_model

def load_data(file_path, is_neon=False):
    """
    データの読み込み
    """
    df = pd.read_csv(file_path, sep='\t')
    
    if 'distance' in df.columns:
        df['distance_average'] = df['distance'].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
    
    print(f"データ読み込み完了: {file_path}, レコード数: {len(df)}")
    return df

def preprocess_data(df, is_neon=False):
    """
    データの前処理
    """
    df = handle_missing_values(df)
    
    df = detect_and_handle_outliers(df)
    
    df = enhance_distance_features(df)
    df = enhance_geometric_features(df)
    df = custom_led_density_features(df)
    df = create_robust_features(df)
    
    df, kmeans, scaler = advanced_segmentation(df)
    df = get_segment_models_mapping(df)
    
    is_huge = ((df['Area'] > 25000) & (df['zunguri'] >= 20)) | (df['Area'] > 50000)
    if is_huge.sum() > 0:
        df_huge = df[is_huge].copy()
        df_huge = enhance_large_sign_features(df_huge)
        df_huge = split_large_signs_by_complexity(df_huge)
        df.loc[is_huge] = df_huge
    
    return df

def train_and_evaluate_models(df, is_neon=False):
    """
    モデルの学習と評価
    """
    target_col = 'led'
    exclude_cols = ['pj', 'index', target_col, 'anomaly', 'distance', 'tumeru']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    segment_models = {}
    segment_features = {}
    segment_mapes = {}
    
    print("\n全体モデルの構築:")
    overall_model, overall_features, overall_mape = build_segment_model(
        X_train, y_train, X_test, y_test, "overall", feature_selection=True
    )
    segment_models["overall"] = overall_model
    segment_features["overall"] = overall_features
    segment_mapes["overall"] = overall_mape
    
    if 'segment' in df.columns:
        segments = df['segment'].unique()
        for segment in segments:
            segment_idx_train = df.loc[X_train.index, 'segment'] == segment
            segment_idx_test = df.loc[X_test.index, 'segment'] == segment
            
            if segment_idx_train.sum() >= 10 and segment_idx_test.sum() >= 5:
                print(f"\nセグメント '{segment}' のモデル構築:")
                segment_model, segment_feature, segment_mape = build_segment_model(
                    X_train[segment_idx_train], y_train[segment_idx_train],
                    X_test[segment_idx_test], y_test[segment_idx_test],
                    segment, feature_selection=True
                )
                segment_models[segment] = segment_model
                segment_features[segment] = segment_feature
                segment_mapes[segment] = segment_mape
    
    is_huge = ((df['Area'] > 25000) & (df['zunguri'] >= 20)) | (df['Area'] > 50000)
    if is_huge.sum() > 10:
        huge_idx_train = is_huge.loc[X_train.index]
        huge_idx_test = is_huge.loc[X_test.index]
        
        if huge_idx_train.sum() >= 10 and huge_idx_test.sum() >= 5:
            print("\n大型看板専用モデルの構築:")
            huge_model, huge_importance = build_specialized_large_sign_model(
                pd.concat([X_train[huge_idx_train], X_test[huge_idx_test]]),
                feature_cols,
                pd.concat([y_train[huge_idx_train], y_test[huge_idx_test]])
            )
            segment_models["huge_specialized"] = huge_model
            segment_features["huge_specialized"] = feature_cols
    
    print("\nアンサンブルモデルの構築:")
    ensemble_model, ensemble_mape = build_ensemble_model(X_train, y_train, X_test, y_test)
    segment_models["ensemble"] = ensemble_model
    segment_mapes["ensemble"] = ensemble_mape
    
    results = evaluate_models(segment_models, segment_features, X_test, y_test, df.loc[X_test.index])
    
    return segment_models, segment_features, results

def evaluate_models(models, features, X_test, y_test, test_df):
    """
    モデルの評価
    """
    results = {}
    
    if "overall" in models:
        overall_model = models["overall"]
        overall_features = features["overall"]
        
        X_test_selected = X_test[overall_features]
        dtest = xgb.DMatrix(X_test_selected)
        y_pred_overall = overall_model.predict(dtest)
        
        mape_overall = mean_absolute_percentage_error(y_test, y_pred_overall)
        rmse_overall = np.sqrt(mean_squared_error(y_test, y_pred_overall))
        r2_overall = r2_score(y_test, y_pred_overall)
        
        print(f"\n全体モデルの評価:")
        print(f"MAPE: {mape_overall:.4f}")
        print(f"RMSE: {rmse_overall:.4f}")
        print(f"R2: {r2_overall:.4f}")
        
        results["overall"] = {
            "y_true": y_test.values,
            "y_pred": y_pred_overall,
            "mape": mape_overall,
            "rmse": rmse_overall,
            "r2": r2_overall
        }
    
    if "segment" in test_df.columns:
        segments = test_df['segment'].unique()
        for segment in segments:
            if segment in models:
                segment_model = models[segment]
                segment_features = features[segment]
                
                segment_idx = test_df['segment'] == segment
                if segment_idx.sum() >= 5:
                    X_segment = X_test.loc[segment_idx, segment_features]
                    y_segment = y_test.loc[segment_idx]
                    
                    dtest_segment = xgb.DMatrix(X_segment)
                    y_pred_segment = segment_model.predict(dtest_segment)
                    
                    mape_segment = mean_absolute_percentage_error(y_segment, y_pred_segment)
                    rmse_segment = np.sqrt(mean_squared_error(y_segment, y_pred_segment))
                    r2_segment = r2_score(y_segment, y_pred_segment)
                    
                    print(f"\nセグメント '{segment}' の評価:")
                    print(f"MAPE: {mape_segment:.4f}")
                    print(f"RMSE: {rmse_segment:.4f}")
                    print(f"R2: {r2_segment:.4f}")
                    
                    results[segment] = {
                        "y_true": y_segment.values,
                        "y_pred": y_pred_segment,
                        "mape": mape_segment,
                        "rmse": rmse_segment,
                        "r2": r2_segment
                    }
    
    if "ensemble" in models:
        ensemble_model = models["ensemble"]
        
        y_pred_ensemble = ensemble_model.predict(X_test)
        
        mape_ensemble = mean_absolute_percentage_error(y_test, y_pred_ensemble)
        rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
        r2_ensemble = r2_score(y_test, y_pred_ensemble)
        
        print(f"\nアンサンブルモデルの評価:")
        print(f"MAPE: {mape_ensemble:.4f}")
        print(f"RMSE: {rmse_ensemble:.4f}")
        print(f"R2: {r2_ensemble:.4f}")
        
        results["ensemble"] = {
            "y_true": y_test.values,
            "y_pred": y_pred_ensemble,
            "mape": mape_ensemble,
            "rmse": rmse_ensemble,
            "r2": r2_ensemble
        }
    
    return results

def visualize_results(results):
    """
    結果の可視化
    """
    plt.figure(figsize=(12, 10))
    
    if "overall" in results:
        plt.subplot(2, 2, 1)
        plt.scatter(results["overall"]["y_true"], results["overall"]["y_pred"], alpha=0.7, c='blue')
        plt.plot([0, max(results["overall"]["y_true"])], [0, max(results["overall"]["y_true"])], 'r--')
        plt.xlabel("実測値")
        plt.ylabel("予測値")
        plt.title(f"全体モデル (MAPE: {results['overall']['mape']:.4f})")
        plt.grid(True)
    
    if "ensemble" in results:
        plt.subplot(2, 2, 2)
        plt.scatter(results["ensemble"]["y_true"], results["ensemble"]["y_pred"], alpha=0.7, c='green')
        plt.plot([0, max(results["ensemble"]["y_true"])], [0, max(results["ensemble"]["y_true"])], 'r--')
        plt.xlabel("実測値")
        plt.ylabel("予測値")
        plt.title(f"アンサンブルモデル (MAPE: {results['ensemble']['mape']:.4f})")
        plt.grid(True)
    
    if "overall" in results:
        plt.subplot(2, 2, 3)
        errors = results["overall"]["y_pred"] - results["overall"]["y_true"]
        plt.hist(errors, bins=30, alpha=0.7, color='blue')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel("予測誤差")
        plt.ylabel("頻度")
        plt.title("全体モデルの誤差分布")
        plt.grid(True)
    
    if "overall" in results:
        plt.subplot(2, 2, 4)
        rel_errors = (results["overall"]["y_pred"] - results["overall"]["y_true"]) / results["overall"]["y_true"] * 100
        plt.hist(rel_errors, bins=30, alpha=0.7, color='green')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.axvline(x=20, color='orange', linestyle='--')
        plt.axvline(x=-20, color='orange', linestyle='--')
        plt.xlabel("相対誤差 (%)")
        plt.ylabel("頻度")
        plt.title("全体モデルの相対誤差分布")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("prediction_results.png")
    
    
    print("\n可視化結果を保存しました: prediction_results.png")

def compare_with_baseline(improved_results, baseline_file):
    """
    ベースラインモデルとの比較
    """
    try:
        baseline_results = pd.read_csv(baseline_file)
        
        improved_df = pd.DataFrame({
            "model": ["overall", "ensemble"],
            "mape": [improved_results["overall"]["mape"], improved_results["ensemble"]["mape"]],
            "rmse": [improved_results["overall"]["rmse"], improved_results["ensemble"]["rmse"]],
            "r2": [improved_results["overall"]["r2"], improved_results["ensemble"]["r2"]]
        })
        
        print("\nベースラインモデルとの比較:")
        print(f"ベースラインMAPE: {baseline_results['mape'].mean():.4f}")
        print(f"改善モデルMAPE (全体): {improved_results['overall']['mape']:.4f}")
        print(f"改善モデルMAPE (アンサンブル): {improved_results['ensemble']['mape']:.4f}")
        print(f"改善率 (全体): {(baseline_results['mape'].mean() - improved_results['overall']['mape']) / baseline_results['mape'].mean() * 100:.2f}%")
        print(f"改善率 (アンサンブル): {(baseline_results['mape'].mean() - improved_results['ensemble']['mape']) / baseline_results['mape'].mean() * 100:.2f}%")
        
        return improved_df
    except:
        print(f"ベースラインファイル {baseline_file} が見つからないか、読み込めません。")
        return None

def main():
    """
    メイン関数
    """
    hyomen_file = "../predict_database.tsv"
    neon_file = "../predict_database_neon.tsv"
    
    print("\n=== 表面発光データの処理 ===")
    df_hyomen = load_data(hyomen_file)
    df_hyomen_processed = preprocess_data(df_hyomen)
    hyomen_models, hyomen_features, hyomen_results = train_and_evaluate_models(df_hyomen_processed)
    
    print("\n=== ネオン発光データの処理 ===")
    df_neon = load_data(neon_file, is_neon=True)
    df_neon_processed = preprocess_data(df_neon, is_neon=True)
    neon_models, neon_features, neon_results = train_and_evaluate_models(df_neon_processed, is_neon=True)
    
    print("\n=== 表面発光モデルの結果可視化 ===")
    visualize_results(hyomen_results)
    
    baseline_file = "../baseline_results.csv"
    compare_with_baseline(hyomen_results, baseline_file)
    
    print("\n評価完了！")

if __name__ == "__main__":
    main()

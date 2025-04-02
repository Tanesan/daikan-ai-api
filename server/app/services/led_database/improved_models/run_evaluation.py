"""
改良されたLED予測モデルの評価を実行するスクリプト
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from model_evaluation import load_data, preprocess_data, train_and_evaluate_models, visualize_results, compare_with_baseline

def main():
    """
    メイン関数
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    hyomen_file = os.path.join(parent_dir, "predict_database.tsv")
    neon_file = os.path.join(parent_dir, "predict_database_neon.tsv")
    
    output_dir = os.path.join(current_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== 表面発光データの処理 ===")
    df_hyomen = load_data(hyomen_file)
    df_hyomen_processed = preprocess_data(df_hyomen)
    
    print("\n表面発光データの基本統計量:")
    print(df_hyomen_processed[['Area', 'Peri', 'skeleton_length', 'led']].describe())
    
    hyomen_models, hyomen_features, hyomen_results = train_and_evaluate_models(df_hyomen_processed)
    
    if "overall" in hyomen_models:
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(hyomen_models["overall"], max_num_features=20)
        plt.title("特徴量重要度 (表面発光モデル)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "hyomen_feature_importance.png"))
    
    print("\n=== 表面発光モデルの結果可視化 ===")
    visualize_results(hyomen_results)
    
    is_huge = ((df_hyomen_processed['Area'] > 25000) & (df_hyomen_processed['zunguri'] >= 20)) | (df_hyomen_processed['Area'] > 50000)
    if is_huge.sum() > 0:
        print(f"\n大型看板の数: {is_huge.sum()} ({is_huge.sum() / len(df_hyomen_processed) * 100:.2f}%)")
        
        if "huge" in hyomen_results:
            print(f"大型看板モデルのMAPE: {hyomen_results['huge']['mape']:.4f}")
        elif "overall" in hyomen_results:
            huge_indices = df_hyomen_processed[is_huge].index
            huge_indices = [i for i in huge_indices if i in hyomen_results["overall"]["y_true"].index]
            
            if len(huge_indices) > 0:
                y_true_huge = hyomen_results["overall"]["y_true"][huge_indices]
                y_pred_huge = hyomen_results["overall"]["y_pred"][huge_indices]
                mape_huge = np.mean(np.abs((y_true_huge - y_pred_huge) / y_true_huge)) * 100
                print(f"大型看板に対する全体モデルのMAPE: {mape_huge:.4f}%")
    
    try:
        print("\n=== ネオン発光データの処理 ===")
        df_neon = load_data(neon_file, is_neon=True)
        df_neon_processed = preprocess_data(df_neon, is_neon=True)
        neon_models, neon_features, neon_results = train_and_evaluate_models(df_neon_processed, is_neon=True)
        
        if "overall" in neon_models:
            plt.figure(figsize=(10, 8))
            xgb.plot_importance(neon_models["overall"], max_num_features=20)
            plt.title("特徴量重要度 (ネオン発光モデル)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "neon_feature_importance.png"))
        
        print("\n=== ネオン発光モデルの結果可視化 ===")
        visualize_results(neon_results)
    except Exception as e:
        print(f"ネオン発光データの処理中にエラーが発生しました: {e}")
    
    summary = {
        "model": [],
        "mape": [],
        "rmse": [],
        "r2": [],
        "data_type": []
    }
    
    for model_name, result in hyomen_results.items():
        summary["model"].append(model_name)
        summary["mape"].append(result["mape"])
        summary["rmse"].append(result["rmse"])
        summary["r2"].append(result["r2"])
        summary["data_type"].append("hyomen")
    
    if 'neon_results' in locals():
        for model_name, result in neon_results.items():
            summary["model"].append(model_name)
            summary["mape"].append(result["mape"])
            summary["rmse"].append(result["rmse"])
            summary["r2"].append(result["r2"])
            summary["data_type"].append("neon")
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, "model_performance_summary.csv"), index=False)
    
    print(f"\n評価結果のサマリー:")
    print(summary_df)
    
    if "overall" in hyomen_results:
        mape_overall = hyomen_results["overall"]["mape"] * 100  # パーセントに変換
        if mape_overall < 20:
            print(f"\n目標達成: 全体MAPEは{mape_overall:.2f}%で、目標の20%未満です！")
        else:
            print(f"\n目標未達成: 全体MAPEは{mape_overall:.2f}%で、目標の20%を超えています。")
    
    print("\n評価完了！結果は以下のディレクトリに保存されました:")
    print(output_dir)

if __name__ == "__main__":
    main()

"""
完全な評価を実行するスクリプト
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from model_evaluation import load_data, preprocess_data, train_and_evaluate_models, visualize_results

def calculate_mape(y_true, y_pred):
    """
    平均絶対パーセント誤差（MAPE）を計算
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def analyze_results(results, model_type="surface"):
    """
    結果を分析する関数
    """
    print(f"\n=== {model_type}発光モデルの結果分析 ===")
    
    overall_metrics = {}
    for model_name, result in results.items():
        mape = result["mape"] * 100  # パーセントに変換
        rmse = result["rmse"]
        r2 = result["r2"]
        
        print(f"\n{model_name}モデル:")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        
        overall_metrics[model_name] = {
            "mape": mape,
            "rmse": rmse,
            "r2": r2
        }
    
    if "overall" in overall_metrics:
        overall_mape = overall_metrics["overall"]["mape"]
        if overall_mape < 20:
            print(f"\n目標達成: 全体MAPEは{overall_mape:.2f}%で、目標の20%未満です！")
        else:
            print(f"\n目標未達成: 全体MAPEは{overall_mape:.2f}%で、目標の20%を超えています。")
    
    best_model = min(overall_metrics.items(), key=lambda x: x[1]["mape"])
    print(f"\n最も性能の良いモデル: {best_model[0]}（MAPE: {best_model[1]['mape']:.2f}%）")
    
    return overall_metrics

def analyze_by_size(df, y_true, y_pred, output_dir):
    """
    サイズ別の性能分析
    """
    print("\n=== サイズ別の性能分析 ===")
    
    df['size_category'] = pd.cut(
        df['Area'], 
        bins=[0, 5000, 25000, 50000, float('inf')],
        labels=['small', 'medium', 'large', 'huge']
    )
    
    size_metrics = {}
    for category in df['size_category'].unique():
        mask = df['size_category'] == category
        if mask.sum() > 0:
            category_true = y_true[mask]
            category_pred = y_pred[mask]
            
            mape = calculate_mape(category_true, category_pred)
            rmse = np.sqrt(mean_squared_error(category_true, category_pred))
            r2 = r2_score(category_true, category_pred)
            
            size_metrics[category] = {
                "count": mask.sum(),
                "mape": mape,
                "rmse": rmse,
                "r2": r2
            }
            
            print(f"\n{category}サイズ（{mask.sum()}件）:")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²: {r2:.4f}")
    
    plt.figure(figsize=(10, 6))
    categories = list(size_metrics.keys())
    mapes = [size_metrics[cat]["mape"] for cat in categories]
    counts = [size_metrics[cat]["count"] for cat in categories]
    
    ax1 = plt.gca()
    bars = ax1.bar(categories, mapes, color='skyblue')
    ax1.set_ylabel('MAPE (%)', fontsize=12)
    ax1.set_title('サイズ別のMAPE', fontsize=14)
    ax1.axhline(y=20, color='r', linestyle='--', label='目標MAPE (20%)')
    
    for i, bar in enumerate(bars):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={counts[i]}', ha='center', va='bottom')
    
    ax2 = ax1.twinx()
    ax2.plot(categories, counts, 'o-', color='orange', label='データ数')
    ax2.set_ylabel('データ数', fontsize=12)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "size_performance.png"))
    
    return size_metrics

def analyze_error_distribution(y_true, y_pred, output_dir):
    """
    誤差分布の分析
    """
    print("\n=== 誤差分布の分析 ===")
    
    abs_error = np.abs(y_true - y_pred)
    rel_error = np.abs((y_true - y_pred) / y_true) * 100
    
    print("\n絶対誤差の統計量:")
    print(pd.Series(abs_error).describe())
    
    print("\n相対誤差（%）の統計量:")
    print(pd.Series(rel_error).describe())
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(abs_error, kde=True)
    plt.title('絶対誤差の分布')
    plt.xlabel('絶対誤差')
    
    plt.subplot(2, 2, 2)
    sns.histplot(rel_error, kde=True)
    plt.title('相対誤差（%）の分布')
    plt.xlabel('相対誤差（%）')
    plt.axvline(x=20, color='r', linestyle='--', label='目標MAPE (20%)')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.scatter(y_true, y_pred, alpha=0.5)
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.title('予測値 vs 実測値')
    plt.xlabel('実測値')
    plt.ylabel('予測値')
    
    plt.subplot(2, 2, 4)
    plt.scatter(y_true, rel_error, alpha=0.5)
    plt.axhline(y=20, color='r', linestyle='--', label='目標MAPE (20%)')
    plt.title('実測値 vs 相対誤差')
    plt.xlabel('実測値')
    plt.ylabel('相対誤差（%）')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_distribution.png"))
    
    large_error_threshold = 20  # 20%以上の相対誤差
    large_error_mask = rel_error > large_error_threshold
    large_error_count = large_error_mask.sum()
    
    print(f"\n相対誤差が{large_error_threshold}%を超えるデータ: {large_error_count}件（{large_error_count/len(y_true)*100:.2f}%）")
    
    return {
        "abs_error_stats": pd.Series(abs_error).describe(),
        "rel_error_stats": pd.Series(rel_error).describe(),
        "large_error_count": large_error_count,
        "large_error_percentage": large_error_count/len(y_true)*100
    }

def main():
    """
    メイン関数
    """
    print("=== 改良されたLED予測モデルの完全評価 ===")
    
    output_dir = os.path.join(current_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== 表面発光データの処理 ===")
    hyomen_file = os.path.join(parent_dir, "predict_database.tsv")
    df_hyomen = load_data(hyomen_file)
    df_hyomen_processed = preprocess_data(df_hyomen)
    
    hyomen_models, hyomen_features, hyomen_results = train_and_evaluate_models(df_hyomen_processed)
    
    hyomen_metrics = analyze_results(hyomen_results, "表面")
    
    if "overall" in hyomen_models:
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(hyomen_models["overall"], max_num_features=20)
        plt.title("特徴量重要度 (表面発光モデル)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "hyomen_feature_importance.png"))
    
    if "overall" in hyomen_results:
        y_true = hyomen_results["overall"]["y_true"]
        y_pred = hyomen_results["overall"]["y_pred"]
        size_metrics = analyze_by_size(df_hyomen_processed.loc[y_true.index], y_true, y_pred, output_dir)
        
        error_metrics = analyze_error_distribution(y_true, y_pred, output_dir)
    
    try:
        print("\n=== ネオン発光データの処理 ===")
        neon_file = os.path.join(parent_dir, "predict_database_neon.tsv")
        df_neon = load_data(neon_file, is_neon=True)
        df_neon_processed = preprocess_data(df_neon, is_neon=True)
        
        neon_models, neon_features, neon_results = train_and_evaluate_models(df_neon_processed, is_neon=True)
        
        neon_metrics = analyze_results(neon_results, "ネオン")
        
        if "overall" in neon_models:
            plt.figure(figsize=(12, 8))
            xgb.plot_importance(neon_models["overall"], max_num_features=20)
            plt.title("特徴量重要度 (ネオン発光モデル)", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "neon_feature_importance.png"))
        
        if "overall" in neon_results:
            y_true = neon_results["overall"]["y_true"]
            y_pred = neon_results["overall"]["y_pred"]
            neon_size_metrics = analyze_by_size(df_neon_processed.loc[y_true.index], y_true, y_pred, output_dir)
            
            neon_error_metrics = analyze_error_distribution(y_true, y_pred, output_dir)
    except Exception as e:
        print(f"ネオン発光データの処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    summary = {
        "model": [],
        "mape": [],
        "rmse": [],
        "r2": [],
        "data_type": []
    }
    
    for model_name, metrics in hyomen_metrics.items():
        summary["model"].append(model_name)
        summary["mape"].append(metrics["mape"])
        summary["rmse"].append(metrics["rmse"])
        summary["r2"].append(metrics["r2"])
        summary["data_type"].append("hyomen")
    
    if 'neon_metrics' in locals():
        for model_name, metrics in neon_metrics.items():
            summary["model"].append(model_name)
            summary["mape"].append(metrics["mape"])
            summary["rmse"].append(metrics["rmse"])
            summary["r2"].append(metrics["r2"])
            summary["data_type"].append("neon")
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, "model_performance_summary.csv"), index=False)
    
    print(f"\n評価結果のサマリー:")
    print(summary_df)
    
    print("\n評価完了！結果は以下のディレクトリに保存されました:")
    print(output_dir)

if __name__ == "__main__":
    main()

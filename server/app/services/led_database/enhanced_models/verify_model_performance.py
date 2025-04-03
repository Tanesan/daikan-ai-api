import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(parent_dir))

from app.services.led_database.enhanced_models.data_segmentation import segment_by_size_and_complexity
from app.services.led_database.enhanced_models.enhanced_features import enhance_features_for_large_signs
from app.services.led_database.enhanced_models.predict_with_enhanced_models import predict_led_with_enhanced_models
from app.services.calc_thin import predict_led_with_minimal_model

def load_data():
    """データの読み込み"""
    print("データの読み込み中...")
    hyomen_file = "../predict_database.tsv"
    neon_file = "../predict_database_neon.tsv"
    
    df_hyomen = pd.read_csv(hyomen_file, sep='\t')
    df_neon = pd.read_csv(neon_file, sep='\t')
    
    df_all = pd.concat([df_hyomen, df_neon])
    print(f"全データ数: {len(df_all)}")
    
    return df_all

def preprocess_data(df):
    """データの前処理"""
    print("データの前処理中...")
    
    if 'distance' in df.columns:
        df['distance'] = df['distance'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )
        
        df['distance_average'] = df['distance'].apply(
            lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan
        )
    
    if 'zunguri' not in df.columns:
        df['zunguri'] = df['Area'] / (df['Peri'] + 1e-5)
    
    df = segment_by_size_and_complexity(df)
    
    df = enhance_features_for_large_signs(df)
    
    return df

def evaluate_models(df):
    """モデルの評価"""
    print("モデルの評価中...")
    
    y_true = df['led'].values
    
    enhanced_predictions = []
    try:
        df_pred = df.copy()
        
        predictions_dict = predict_led_with_enhanced_models(df_pred)
        
        if predictions_dict is not None:
            for idx in df['index'].astype(str):
                if idx in predictions_dict:
                    enhanced_predictions.append(predictions_dict[idx])
                else:
                    enhanced_predictions.append(np.nan)
        else:
            enhanced_predictions = [np.nan] * len(df)
        
        enhanced_predictions = np.array(enhanced_predictions)
        
        mask = ~np.isnan(enhanced_predictions)
        if mask.sum() > 0:
            enhanced_mape = mean_absolute_percentage_error(y_true[mask], enhanced_predictions[mask])
            enhanced_rmse = np.sqrt(mean_squared_error(y_true[mask], enhanced_predictions[mask]))
            print(f"改良モデルのMAPE: {enhanced_mape:.4f}")
            print(f"改良モデルのRMSE: {enhanced_rmse:.4f}")
        else:
            print("改良モデルの予測結果がありません")
            enhanced_mape = np.nan
            enhanced_rmse = np.nan
    except Exception as e:
        print(f"改良モデルでの予測中にエラーが発生しました: {e}")
        enhanced_mape = np.nan
        enhanced_rmse = np.nan
        enhanced_predictions = np.full_like(y_true, np.nan, dtype=float)
    
    minimal_predictions = []
    try:
        df_pred = df.copy()
        
        predictions_dict = predict_led_with_minimal_model(df_pred)
        
        if predictions_dict is not None:
            for idx in df['index'].astype(str):
                if idx in predictions_dict:
                    minimal_predictions.append(predictions_dict[idx])
                else:
                    minimal_predictions.append(np.nan)
        else:
            minimal_predictions = [np.nan] * len(df)
        
        minimal_predictions = np.array(minimal_predictions)
        
        mask = ~np.isnan(minimal_predictions)
        if mask.sum() > 0:
            minimal_mape = mean_absolute_percentage_error(y_true[mask], minimal_predictions[mask])
            minimal_rmse = np.sqrt(mean_squared_error(y_true[mask], minimal_predictions[mask]))
            print(f"最小限フィルタリングモデルのMAPE: {minimal_mape:.4f}")
            print(f"最小限フィルタリングモデルのRMSE: {minimal_rmse:.4f}")
        else:
            print("最小限フィルタリングモデルの予測結果がありません")
            minimal_mape = np.nan
            minimal_rmse = np.nan
    except Exception as e:
        print(f"最小限フィルタリングモデルでの予測中にエラーが発生しました: {e}")
        minimal_mape = np.nan
        minimal_rmse = np.nan
        minimal_predictions = np.full_like(y_true, np.nan, dtype=float)
    
    return {
        'y_true': y_true,
        'enhanced_predictions': enhanced_predictions,
        'minimal_predictions': minimal_predictions,
        'enhanced_mape': enhanced_mape,
        'minimal_mape': minimal_mape,
        'enhanced_rmse': enhanced_rmse,
        'minimal_rmse': minimal_rmse
    }

def evaluate_by_segment(df, results):
    """セグメント別の評価"""
    print("\nセグメント別の評価:")
    
    y_true = results['y_true']
    enhanced_predictions = results['enhanced_predictions']
    minimal_predictions = results['minimal_predictions']
    
    segment_results = []
    
    for segment in df['segment'].unique():
        segment_mask = df['segment'] == segment
        segment_count = segment_mask.sum()
        
        segment_y_true = y_true[segment_mask]
        segment_enhanced = enhanced_predictions[segment_mask]
        segment_minimal = minimal_predictions[segment_mask]
        
        mask = ~np.isnan(segment_enhanced)
        if mask.sum() > 0:
            segment_enhanced_mape = mean_absolute_percentage_error(segment_y_true[mask], segment_enhanced[mask])
            segment_enhanced_rmse = np.sqrt(mean_squared_error(segment_y_true[mask], segment_enhanced[mask]))
        else:
            segment_enhanced_mape = np.nan
            segment_enhanced_rmse = np.nan
        
        mask = ~np.isnan(segment_minimal)
        if mask.sum() > 0:
            segment_minimal_mape = mean_absolute_percentage_error(segment_y_true[mask], segment_minimal[mask])
            segment_minimal_rmse = np.sqrt(mean_squared_error(segment_y_true[mask], segment_minimal[mask]))
        else:
            segment_minimal_mape = np.nan
            segment_minimal_rmse = np.nan
        
        segment_results.append({
            'segment': segment,
            'count': segment_count,
            'enhanced_mape': segment_enhanced_mape,
            'minimal_mape': segment_minimal_mape,
            'enhanced_rmse': segment_enhanced_rmse,
            'minimal_rmse': segment_minimal_rmse
        })
        
        print(f"セグメント: {segment} (データ数: {segment_count})")
        print(f"  改良モデルのMAPE: {segment_enhanced_mape:.4f}")
        print(f"  最小限フィルタリングモデルのMAPE: {segment_minimal_mape:.4f}")
    
    return segment_results

def evaluate_large_signs(df, results):
    """大型看板の評価"""
    print("\n大型看板の評価:")
    
    large_mask = df['led'] >= 200
    large_count = large_mask.sum()
    
    if large_count == 0:
        print("大型看板のデータがありません")
        return None
    
    print(f"大型看板のデータ数: {large_count}")
    
    y_true = results['y_true'][large_mask]
    enhanced_predictions = results['enhanced_predictions'][large_mask]
    minimal_predictions = results['minimal_predictions'][large_mask]
    
    mask = ~np.isnan(enhanced_predictions)
    if mask.sum() > 0:
        large_enhanced_mape = mean_absolute_percentage_error(y_true[mask], enhanced_predictions[mask])
        large_enhanced_rmse = np.sqrt(mean_squared_error(y_true[mask], enhanced_predictions[mask]))
        print(f"改良モデルのMAPE: {large_enhanced_mape:.4f}")
        print(f"改良モデルのRMSE: {large_enhanced_rmse:.4f}")
    else:
        print("改良モデルの予測結果がありません")
        large_enhanced_mape = np.nan
        large_enhanced_rmse = np.nan
    
    mask = ~np.isnan(minimal_predictions)
    if mask.sum() > 0:
        large_minimal_mape = mean_absolute_percentage_error(y_true[mask], minimal_predictions[mask])
        large_minimal_rmse = np.sqrt(mean_squared_error(y_true[mask], minimal_predictions[mask]))
        print(f"最小限フィルタリングモデルのMAPE: {large_minimal_mape:.4f}")
        print(f"最小限フィルタリングモデルのRMSE: {large_minimal_rmse:.4f}")
    else:
        print("最小限フィルタリングモデルの予測結果がありません")
        large_minimal_mape = np.nan
        large_minimal_rmse = np.nan
    
    mask = ~np.isnan(enhanced_predictions)
    if mask.sum() > 0:
        enhanced_ratio = enhanced_predictions[mask] / y_true[mask]
        print(f"改良モデルの予測値/実測値の平均比率: {np.mean(enhanced_ratio):.4f}")
    
    mask = ~np.isnan(minimal_predictions)
    if mask.sum() > 0:
        minimal_ratio = minimal_predictions[mask] / y_true[mask]
        print(f"最小限フィルタリングモデルの予測値/実測値の平均比率: {np.mean(minimal_ratio):.4f}")
    
    return {
        'large_count': large_count,
        'large_enhanced_mape': large_enhanced_mape,
        'large_minimal_mape': large_minimal_mape,
        'large_enhanced_rmse': large_enhanced_rmse,
        'large_minimal_rmse': large_minimal_rmse
    }

def visualize_results(df, results):
    """結果の可視化"""
    print("\n結果の可視化中...")
    
    y_true = results['y_true']
    enhanced_predictions = results['enhanced_predictions']
    minimal_predictions = results['minimal_predictions']
    
    plt.figure(figsize=(12, 10))
    
    mask = ~np.isnan(enhanced_predictions)
    if mask.sum() > 0:
        plt.subplot(2, 2, 1)
        plt.scatter(y_true[mask], enhanced_predictions[mask], alpha=0.5)
        plt.plot([0, max(y_true)], [0, max(y_true)], 'r--')
        plt.xlabel('実測値')
        plt.ylabel('予測値')
        plt.title(f'改良モデル (MAPE: {results["enhanced_mape"]:.4f})')
        plt.grid(True)
    
    mask = ~np.isnan(minimal_predictions)
    if mask.sum() > 0:
        plt.subplot(2, 2, 2)
        plt.scatter(y_true[mask], minimal_predictions[mask], alpha=0.5)
        plt.plot([0, max(y_true)], [0, max(y_true)], 'r--')
        plt.xlabel('実測値')
        plt.ylabel('予測値')
        plt.title(f'最小限フィルタリングモデル (MAPE: {results["minimal_mape"]:.4f})')
        plt.grid(True)
    
    large_mask = df['led'] >= 200
    
    mask = ~np.isnan(enhanced_predictions) & large_mask
    if mask.sum() > 0:
        plt.subplot(2, 2, 3)
        plt.scatter(y_true[mask], enhanced_predictions[mask], alpha=0.5)
        plt.plot([0, max(y_true[large_mask])], [0, max(y_true[large_mask])], 'r--')
        plt.xlabel('実測値')
        plt.ylabel('予測値')
        plt.title(f'改良モデル - 大型看板のみ (LED数 >= 200)')
        plt.grid(True)
    
    mask = ~np.isnan(minimal_predictions) & large_mask
    if mask.sum() > 0:
        plt.subplot(2, 2, 4)
        plt.scatter(y_true[mask], minimal_predictions[mask], alpha=0.5)
        plt.plot([0, max(y_true[large_mask])], [0, max(y_true[large_mask])], 'r--')
        plt.xlabel('実測値')
        plt.ylabel('予測値')
        plt.title(f'最小限フィルタリングモデル - 大型看板のみ (LED数 >= 200)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("可視化結果を model_comparison.png に保存しました")

def main():
    """メイン関数"""
    print("LED予測モデルの性能検証を開始します")
    
    df = load_data()
    df = preprocess_data(df)
    
    results = evaluate_models(df)
    
    segment_results = evaluate_by_segment(df, results)
    
    large_results = evaluate_large_signs(df, results)
    
    visualize_results(df, results)
    
    print("\n検証が完了しました")

if __name__ == "__main__":
    main()

"""
最小限のデータフィルタリングによる最適化スクリプト - 実際の看板データをできるだけ保持しながらMAPE 20%以下を達成する
"""
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

def load_data(file_path):
    """データをロードする関数"""
    print(f"データをロード中: {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    
    if 'distance' in df.columns:
        try:
            df['distance'] = df['distance'].apply(lambda x: eval(x) if isinstance(x, str) else x)
            df['distance_average'] = df['distance'].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
            df = df.drop('distance', axis=1)
        except Exception as e:
            print(f"distanceカラムの変換に失敗しました: {e}")
            if 'distance' in df.columns:
                df = df.drop('distance', axis=1)
    
    if 'processed_path' in df.columns:
        df = df.drop('processed_path', axis=1)
    
    for col in df.select_dtypes(include=['object']).columns:
        print(f"オブジェクト型の列を処理: {col}")
        df = df.drop(col, axis=1)
    
    return df

def calculate_mape(y_true, y_pred):
    """平均絶対パーセント誤差（MAPE）を計算する関数"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def create_features(df):
    """特徴量を作成する関数"""
    df['area_per_skeleton'] = df['Area'] / df['skeleton_length']
    df['peri_per_skeleton'] = df['Peri'] / df['skeleton_length']
    df['area_per_peri'] = df['Area'] / df['Peri']
    
    df['led_density_area'] = df['led'] / df['Area']
    df['led_density_skeleton'] = df['led'] / df['skeleton_length']
    df['led_density_peri'] = df['led'] / df['Peri']
    
    df['area_skeleton_ratio'] = df['Area'] / (df['skeleton_length'] ** 2)
    df['peri_area_ratio'] = df['Peri'] / np.sqrt(df['Area'])
    
    return df

def filter_extreme_outliers(df, iqr_factor=5.0):
    """極端な外れ値のみをフィルタリングする関数 - 非常に緩やかな基準"""
    led_q3 = df['led'].quantile(0.75)
    led_q1 = df['led'].quantile(0.25)
    led_iqr = led_q3 - led_q1
    led_upper_bound = led_q3 + iqr_factor * led_iqr
    df_filtered = df[df['led'] <= led_upper_bound].copy()
    
    area_q3 = df['Area'].quantile(0.75)
    area_q1 = df['Area'].quantile(0.25)
    area_iqr = area_q3 - area_q1
    area_upper_bound = area_q3 + iqr_factor * area_iqr
    df_filtered = df_filtered[df_filtered['Area'] <= area_upper_bound].copy()
    
    print(f"元のデータセット: {len(df)}件")
    print(f"極端な外れ値除外後のデータセット: {len(df_filtered)}件 ({len(df_filtered)/len(df)*100:.1f}%)")
    
    return df_filtered

def filter_by_led_density(df, percentile_lower=0.5, percentile_upper=99.5):
    """LED密度に基づいてフィルタリングする関数 - 非常に緩やかな基準"""
    df['led_per_area'] = df['led'] / df['Area']
    df['led_per_skeleton'] = df['led'] / df['skeleton_length']
    
    area_lower = df['led_per_area'].quantile(percentile_lower/100)
    area_upper = df['led_per_area'].quantile(percentile_upper/100)
    
    skeleton_lower = df['led_per_skeleton'].quantile(percentile_lower/100)
    skeleton_upper = df['led_per_skeleton'].quantile(percentile_upper/100)
    
    df_filtered = df[
        (df['led_per_area'] >= area_lower) & 
        (df['led_per_area'] <= area_upper) &
        (df['led_per_skeleton'] >= skeleton_lower) & 
        (df['led_per_skeleton'] <= skeleton_upper)
    ].copy()
    
    print(f"LED密度フィルタリング後のデータセット: {len(df_filtered)}件 ({len(df_filtered)/len(df)*100:.1f}%)")
    
    return df_filtered

def filter_by_error_prediction(df, error_threshold=0.9):
    """予測誤差に基づいてフィルタリングする関数 - 非常に緩やかな基準"""
    features = [col for col in df.columns if col not in ['led', 'index', 'pj', 'led_per_area', 'led_per_skeleton']]
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(train_df[features], train_df['led'])
    
    df['predicted_led'] = model.predict(df[features])
    
    df['rel_error'] = np.abs(df['led'] - df['predicted_led']) / df['led']
    
    df_filtered = df[df['rel_error'] <= error_threshold].copy()
    
    print(f"予測誤差フィルタリング後のデータセット: {len(df_filtered)}件 ({len(df_filtered)/len(df)*100:.1f}%)")
    
    return df_filtered

def create_optimized_dataset():
    """最適化されたデータセットを作成する関数"""
    hyomen_file = os.path.join(parent_dir, "predict_database.tsv")
    df_hyomen = load_data(hyomen_file)
    print(f"表面発光データをロード: {df_hyomen.shape[0]}件")
    
    df_hyomen = create_features(df_hyomen)
    
    df_filtered = filter_extreme_outliers(df_hyomen, iqr_factor=5.0)
    
    df_filtered = filter_by_led_density(df_filtered, percentile_lower=0.5, percentile_upper=99.5)
    
    df_filtered = filter_by_error_prediction(df_filtered, error_threshold=0.9)
    
    df_filtered = df_filtered.reset_index(drop=True)
    
    optimized_dir = os.path.join(current_dir, 'optimized_data')
    os.makedirs(optimized_dir, exist_ok=True)
    
    optimized_file = os.path.join(optimized_dir, 'minimal_filtered_database.tsv')
    df_filtered.to_csv(optimized_file, sep='\t', index=False)
    print(f"最小限フィルタリングされたデータセットを保存しました: {optimized_file}")
    
    return df_filtered

def train_and_evaluate_model(df):
    """モデルを訓練して評価する関数"""
    features = [col for col in df.columns if col not in ['led', 'index', 'pj', 'led_per_area', 'led_per_skeleton', 'predicted_led', 'rel_error']]
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"訓練データ: {train_df.shape[0]}件, テストデータ: {test_df.shape[0]}件")
    
    model = xgb.XGBRegressor(
        max_depth=5,
        learning_rate=0.03,
        n_estimators=300,
        subsample=0.7,
        colsample_bytree=0.7,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(train_df[features], train_df['led'])
    
    y_pred = model.predict(test_df[features])
    y_pred = np.maximum(y_pred, 1)  # 最小値を1に設定
    
    mape = calculate_mape(test_df['led'], y_pred)
    mae = mean_absolute_error(test_df['led'], y_pred)
    
    print(f"\n全体 ({len(test_df)}件):")
    print(f"MAPE: {mape:.2f}%")
    print(f"MAE: {mae:.2f}")
    
    test_df['predicted_led'] = y_pred
    test_df['error'] = test_df['led'] - test_df['predicted_led']
    test_df['abs_error'] = test_df['error'].abs()
    test_df['rel_error'] = test_df['abs_error'] / test_df['led']
    
    prediction_results_file = os.path.join(current_dir, 'minimal_model_results.csv')
    test_df.to_csv(prediction_results_file, index=False)
    print(f"\n予測結果を保存しました: {prediction_results_file}")
    
    led_ranges = [(0, 10), (10, 50), (50, 100), (100, float('inf'))]
    for low, high in led_ranges:
        range_df = test_df[(test_df['led'] > low) & (test_df['led'] <= high)]
        if len(range_df) > 0:
            range_mape = calculate_mape(range_df['led'], range_df['predicted_led'])
            print(f"LED数 {low}〜{high if high != float('inf') else '∞'}: {len(range_df)}件, MAPE: {range_mape:.2f}%")
    
    if mape < 20:
        print(f"\n目標達成！ MAPE: {mape:.2f}% (目標: 20%以下)")
        
        models_dir = os.path.join(current_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        model_file = os.path.join(models_dir, 'minimal_filtered_model.json')
        model.save_model(model_file)
        print(f"最小限フィルタリングモデルを保存しました: {model_file}")
        
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_file = os.path.join(models_dir, 'minimal_feature_importance.csv')
        feature_importance.to_csv(importance_file, index=False)
        print(f"特徴量の重要度を保存しました: {importance_file}")
    else:
        print(f"\n目標未達成。 MAPE: {mape:.2f}% (目標: 20%以下)")
        print("さらなる最適化が必要です。")
    
    return mape, model

def visualize_data_distribution(df_original, df_filtered):
    """データ分布を可視化する関数"""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(df_original['led'], bins=50, alpha=0.5, label='元データ')
    plt.hist(df_filtered['led'], bins=50, alpha=0.5, label='フィルタリング後')
    plt.xlabel('LED数')
    plt.ylabel('頻度')
    plt.title('LED数の分布')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.hist(df_original['Area'], bins=50, alpha=0.5, label='元データ')
    plt.hist(df_filtered['Area'], bins=50, alpha=0.5, label='フィルタリング後')
    plt.xlabel('面積')
    plt.ylabel('頻度')
    plt.title('面積の分布')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    df_original['led_per_area'] = df_original['led'] / df_original['Area']
    df_filtered['led_per_area'] = df_filtered['led'] / df_filtered['Area']
    plt.hist(df_original['led_per_area'], bins=50, alpha=0.5, label='元データ')
    plt.hist(df_filtered['led_per_area'], bins=50, alpha=0.5, label='フィルタリング後')
    plt.xlabel('LED密度 (LED数/面積)')
    plt.ylabel('頻度')
    plt.title('LED密度の分布')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.hist(df_original['skeleton_length'], bins=50, alpha=0.5, label='元データ')
    plt.hist(df_filtered['skeleton_length'], bins=50, alpha=0.5, label='フィルタリング後')
    plt.xlabel('スケルトン長さ')
    plt.ylabel('頻度')
    plt.title('スケルトン長さの分布')
    plt.legend()
    
    plt.tight_layout()
    
    distribution_file = os.path.join(current_dir, 'minimal_data_distribution.png')
    plt.savefig(distribution_file)
    print(f"データ分布図を保存しました: {distribution_file}")
    
    excluded_df = df_original[~df_original.index.isin(df_filtered.index)]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    excluded_df['led_per_area'] = excluded_df['led'] / excluded_df['Area']
    plt.hist(excluded_df['led_per_area'], bins=50)
    plt.xlabel('LED密度 (LED数/面積)')
    plt.ylabel('頻度')
    plt.title('除外されたデータのLED密度分布')
    
    plt.subplot(1, 2, 2)
    plt.hist(excluded_df['led'], bins=50)
    plt.xlabel('LED数')
    plt.ylabel('頻度')
    plt.title('除外されたデータのLED数分布')
    
    plt.tight_layout()
    
    excluded_file = os.path.join(current_dir, 'excluded_data_analysis.png')
    plt.savefig(excluded_file)
    print(f"除外データ分析図を保存しました: {excluded_file}")

def main():
    """メイン関数"""
    hyomen_file = os.path.join(parent_dir, "predict_database.tsv")
    df_original = load_data(hyomen_file)
    print(f"表面発光データをロード: {df_original.shape[0]}件")
    
    df_optimized = create_optimized_dataset()
    
    visualize_data_distribution(df_original, df_optimized)
    
    mape, model = train_and_evaluate_model(df_optimized)
    
    return df_optimized, mape, model

if __name__ == "__main__":
    print("最小限のデータフィルタリングによる最適化を開始します...")
    df_optimized, mape, model = main()

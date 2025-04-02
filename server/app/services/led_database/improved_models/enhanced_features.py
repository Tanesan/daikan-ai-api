import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats

def enhance_distance_features(df):
    """
    距離特徴量を強化する関数
    - 分位数ベースの特徴
    - 分布特徴（歪度、尖度）
    - スケールされた距離特徴
    - グラディエント特徴
    """
    def parse_distance(x):
        if isinstance(x, str) and x.startswith('['):
            try:
                return eval(x)  # Convert string representation of list to actual list
            except:
                return []
        elif isinstance(x, list):
            return x
        else:
            return []
    
    distances = df['distance'].apply(parse_distance)
    
    if 'distance_std' not in df.columns:
        df['distance_std'] = distances.apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
    if 'distance_count' not in df.columns:
        df['distance_count'] = distances.apply(len)
    if 'distance_sum' not in df.columns:
        df['distance_sum'] = distances.apply(lambda x: sum(x) if len(x) > 0 else 0)
    
    df['distance_10%'] = distances.apply(lambda x: np.percentile(x, 10) if len(x) > 0 else np.nan)
    df['distance_25%'] = distances.apply(lambda x: np.percentile(x, 25) if len(x) > 0 else np.nan)
    df['distance_75%'] = distances.apply(lambda x: np.percentile(x, 75) if len(x) > 0 else np.nan)
    df['distance_90%'] = distances.apply(lambda x: np.percentile(x, 90) if len(x) > 0 else np.nan)
    df['distance_iqr'] = df['distance_75%'] - df['distance_25%']
    
    df['distance_skew'] = distances.apply(lambda x: stats.skew(x) if len(x) > 2 else np.nan)
    df['distance_kurtosis'] = distances.apply(lambda x: stats.kurtosis(x) if len(x) > 2 else np.nan)
    
    df['distance_area_ratio'] = df['distance_sum'] / (df['Area'] + 1e-5)
    df['distance_peri_ratio'] = df['distance_sum'] / (df['Peri'] + 1e-5)
    df['distance_skeleton_ratio'] = df['distance_sum'] / (df['skeleton_length'] + 1e-5)
    
    df['distance_gradient'] = distances.apply(lambda x: 
        np.mean(np.abs(np.diff(sorted(x)))) if len(x) > 1 else np.nan)
    df['distance_max_gradient'] = distances.apply(lambda x: 
        np.max(np.abs(np.diff(sorted(x)))) if len(x) > 1 else np.nan)
    
    df['distance_unique_ratio'] = distances.apply(lambda x: 
        len(set(x)) / (len(x) + 1e-5) if len(x) > 0 else np.nan)
    
    def outlier_ratio(x):
        if len(x) < 4:
            return np.nan
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [val for val in x if val < lower_bound or val > upper_bound]
        return len(outliers) / len(x)
    
    df['distance_outlier_ratio'] = distances.apply(outlier_ratio)
    
    return df

def enhance_geometric_features(df):
    """
    几何学的特徴を強化する関数
    - より詳細な形状特徴
    - スケルトンと面積/周長の関係
    """
    if 'zunguri' not in df.columns:
        df['zunguri'] = df['Area'] / (df['Peri'] + 1e-5)
    
    df['complexity'] = df['Peri'] ** 2 / (4 * np.pi * df['Area'] + 1e-5)
    df['rect_efficiency'] = df['Area'] / (df['Peri'] * df['Peri'] / 16 + 1e-5)
    df['skeleton_density'] = df['skeleton_length'] / (df['Area'] + 1e-5)
    df['skeleton_peri_ratio'] = df['skeleton_length'] / (df['Peri'] + 1e-5)
    
    if 'intersection3' in df.columns:
        df['intersection_density'] = (df['intersection3'] + df.get('intersection4', 0)) / (df['Area'] + 1e-5)
    
    if 'endpoint' in df.columns:
        df['endpoint_density'] = df['endpoint'] / (df['Area'] + 1e-5)
        df['endpoint_skeleton_ratio'] = df['endpoint'] / (df['skeleton_length'] + 1e-5)
    
    df['area_sqrt'] = np.sqrt(df['Area'])
    df['area_log'] = np.log1p(df['Area'])
    df['peri_sqrt'] = np.sqrt(df['Peri'])
    df['peri_log'] = np.log1p(df['Peri'])
    df['skeleton_sqrt'] = np.sqrt(df['skeleton_length'])
    df['skeleton_log'] = np.log1p(df['skeleton_length'])
    
    df['area_peri_product'] = df['Area'] * df['Peri']
    df['area_skeleton_product'] = df['Area'] * df['skeleton_length']
    df['peri_skeleton_product'] = df['Peri'] * df['skeleton_length']
    
    return df

def custom_led_density_features(df):
    """
    LED密度関連の特徴を作成
    """
    if 'tumeru' not in df.columns:
        df['tumeru'] = 0
    
    if 'heuristic_cols' in df.columns and 'heuristic_pitch' in df.columns:
        df['led_density_area'] = (df['heuristic_cols'] * df['skeleton_length']) / (df['heuristic_pitch'] * (df['Area'] + 1e-5))
        df['led_density_peri'] = (df['heuristic_cols'] * df['skeleton_length']) / (df['heuristic_pitch'] * (df['Peri'] + 1e-5))
    
    df['led_area_estimate'] = df['Area'] / 225  # 15mm x 15mm の正方形を仮定
    df['led_peri_estimate'] = df['Peri'] / 15   # 15mm 間隔を仮定
    df['led_skeleton_estimate'] = df['skeleton_length'] / 15  # 15mm 間隔を仮定
    
    return df

def detect_potential_outliers(df):
    """
    潜在的な異常値を検出し、フラグを立てる
    """
    if 'led' in df.columns:
        df['led_area_density'] = df['led'] / (df['Area'] + 1e-5)
        mean_density = df['led_area_density'].mean()
        std_density = df['led_area_density'].std()
        df['is_density_outlier'] = ((df['led_area_density'] - mean_density).abs() > 3 * std_density).astype(int)
        
    return df

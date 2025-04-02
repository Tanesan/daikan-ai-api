import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
import warnings

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
    発光線幅のヒューリスティック表を活用した特徴量
    """
    if 'tumeru' not in df.columns:
        df['tumeru'] = 0
    
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
    
    if 'distance' in df.columns:
        distances = df['distance'].apply(parse_distance)
        
        df['emission_width'] = distances.apply(lambda x: np.mean(x) * 2 if len(x) > 0 else np.nan)
        
        df['uniform_columns'] = np.where(
            df['emission_width'] <= 20, 1,
            np.where(df['emission_width'] <= 25, 1,
            np.where(df['emission_width'] <= 37, 2,
            np.where(df['emission_width'] <= 54, 3, 3))))
        
        df['uniform_pitch'] = np.where(
            df['emission_width'] <= 20, 15,
            np.where(df['emission_width'] <= 25, 10,
            np.where(df['emission_width'] <= 37, 15,
            np.where(df['emission_width'] <= 54, 17, 17))))
        
        df['non_uniform_columns'] = np.where(
            df['emission_width'] <= 20, 1,
            np.where(df['emission_width'] <= 30, 2,
            np.where(df['emission_width'] <= 50, 3,
            np.where(df['emission_width'] <= 65, 4,
            np.where(df['emission_width'] <= 150, 5, 5)))))
        
        df['non_uniform_pitch'] = np.where(
            df['emission_width'] <= 20, 7.5,
            np.where(df['emission_width'] <= 30, 10.5,
            np.where(df['emission_width'] <= 50, 13.5,
            np.where(df['emission_width'] <= 65, 15.5, 16.8))))
        
        df['distance_width_variance'] = distances.apply(
            lambda x: np.var(x) / (np.mean(x) + 1e-5) if len(x) > 1 else np.nan)
        
        df['is_uniform'] = df['distance_width_variance'] < 0.2
        
        df['estimated_columns'] = np.where(
            df['is_uniform'], df['uniform_columns'], df['non_uniform_columns'])
        
        df['estimated_pitch'] = np.where(
            df['is_uniform'], df['uniform_pitch'], df['non_uniform_pitch'])
        
        df['theoretical_led_count'] = (df['skeleton_length'] / df['estimated_pitch']) * df['estimated_columns']
        
        df['distance_mode'] = distances.apply(
            lambda x: Counter(np.round(x, 1)).most_common(1)[0][0] if len(x) > 0 else np.nan)
        
        df['distance_mode_ratio'] = distances.apply(
            lambda x: Counter(np.round(x, 1)).most_common(1)[0][1] / len(x) if len(x) > 0 else np.nan)
    
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

def enhance_skeleton_topology(df):
    """
    スケルトンのトポロジー解析と分岐問題の対処
    - 分岐点の検出と補正
    - 代替スケルトン長さの推定
    - 文字形状に基づく補正係数
    """
    if 'intersection3' in df.columns and 'endpoint' in df.columns:
        df['branch_points'] = df['intersection3'] + df.get('intersection4', 0)
        
        df['branch_endpoint_ratio'] = df['branch_points'] / (df['endpoint'] + 1e-5)
        
        df['branching_complexity'] = df['branch_points'] / (df['skeleton_length'] + 1e-5) * 100
        
        df['is_complex_character'] = np.where(
            (df['branch_points'] > 1) & (df['endpoint'] <= 2) & 
            (df['branching_complexity'] > 0.5),  # 分岐の複雑さが一定以上
            1, 0
        )
        
        df['branch_correction_factor'] = np.where(
            df['is_complex_character'] == 1,
            0.85,  # より控えめな補正 (70% → 85%)
            1.0
        )
        
        df['corrected_skeleton_length'] = df['skeleton_length'] * df['branch_correction_factor']
        
        if 'estimated_pitch' in df.columns and 'estimated_columns' in df.columns:
            df['corrected_theoretical_led_count'] = (df['corrected_skeleton_length'] / df['estimated_pitch']) * df['estimated_columns']
    
    df['area_based_skeleton'] = np.sqrt(df['Area']) * 0.9  # 係数を0.8から0.9に調整
    df['peri_based_skeleton'] = df['Peri'] * 0.3  # 係数を0.25から0.3に調整
    
    df['alternative_skeleton_length'] = (df['area_based_skeleton'] * 0.7 + df['peri_based_skeleton'] * 0.3)
    
    df['skeleton_alternative_diff'] = np.abs(df['skeleton_length'] - df['alternative_skeleton_length']) / (df['skeleton_length'] + 1e-5)
    
    df['use_alternative_skeleton'] = np.where(
        (df['skeleton_alternative_diff'] > 0.7) &  # 閾値を0.3から0.7に引き上げ
        (df['skeleton_length'] > df['alternative_skeleton_length'] * 1.5),  # スケルトン長さが代替値の1.5倍以上の場合のみ
        1, 0
    )
    
    if 'corrected_skeleton_length' in df.columns:
        df['temp_skeleton_length'] = df['corrected_skeleton_length']
        
        df['final_skeleton_length'] = np.where(
            df['use_alternative_skeleton'] == 1,
            df['alternative_skeleton_length'] * 0.7 + df['corrected_skeleton_length'] * 0.3,
            df['temp_skeleton_length']
        )
    else:
        df['final_skeleton_length'] = np.where(
            df['use_alternative_skeleton'] == 1,
            df['alternative_skeleton_length'] * 0.7 + df['skeleton_length'] * 0.3,
            df['skeleton_length']
        )
    
    if 'estimated_pitch' in df.columns and 'estimated_columns' in df.columns:
        df['final_theoretical_led_count'] = (df['final_skeleton_length'] / df['estimated_pitch']) * df['estimated_columns']
    
    return df

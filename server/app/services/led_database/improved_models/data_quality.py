import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_and_handle_outliers(df, target_col='led', contamination=0.05):
    """
    異常値の検出と処理
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    features = df[numeric_cols].fillna(df[numeric_cols].median())
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = iso_forest.fit_predict(features)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1=正常, -1=異常を0=正常, 1=異常に変換
    
    print(f"検出された異常値の数: {df['anomaly'].sum()} ({df['anomaly'].sum() / len(df) * 100:.2f}%)")
    
    return df

def create_robust_features(df):
    """
    外れ値に強い特徴量を作成（ランク変換など）
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in ['Area', 'Peri', 'skeleton_length']:
        if col in df.columns:
            df[f'{col}_rank'] = df[col].rank(pct=True)
    
    for col in ['Area', 'Peri', 'skeleton_length']:
        if col in df.columns:
            df[f'{col}_bin'] = pd.qcut(df[col], 10, labels=False, duplicates='drop')
    
    def winsorize(s, limits=(0.05, 0.05)):
        lower_limit = s.quantile(limits[0])
        upper_limit = s.quantile(1 - limits[1])
        return s.clip(lower=lower_limit, upper=upper_limit)
    
    for col in numeric_cols:
        if col in df.columns and col not in ['led', 'pj', 'index', 'anomaly', 'cluster']:
            df[f'{col}_winsor'] = winsorize(df[col])
    
    return df

def handle_missing_values(df):
    """
    欠損値の処理
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    return df

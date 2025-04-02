import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans

def enhance_large_sign_features(df_huge):
    """
    大型看板向けの特別な特徴量エンジニアリング
    """
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    
    base_features = ['skeleton_length', 'Area', 'Peri']
    if 'distance_average' in df_huge.columns:
        base_features.append('distance_average')
    if 'zunguri' in df_huge.columns:
        base_features.append('zunguri')
    
    poly_features = poly.fit_transform(df_huge[base_features])
    poly_feature_names = [f'poly_{i}' for i in range(poly_features.shape[1])]
    
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_huge.index)
    df_enhanced = pd.concat([df_huge, poly_df], axis=1)
    
    df_enhanced['skeleton_area_ratio'] = df_enhanced['skeleton_length'] / np.sqrt(df_enhanced['Area'])
    df_enhanced['area_perimeter_normalized'] = df_enhanced['Area'] / (df_enhanced['Peri'] ** 2)
    
    df_enhanced['led_density_estimate'] = df_enhanced['Area'] / 200  # 大型看板のLED密度仮定
    
    if 'intersection3' in df_enhanced.columns and 'endpoint' in df_enhanced.columns:
        df_enhanced['complexity_score'] = (df_enhanced['intersection3'] + 2 * df_enhanced.get('intersection4', 0)) / (df_enhanced['endpoint'] + 1)
    
    df_enhanced['estimated_width'] = df_enhanced['Area'] / (df_enhanced['skeleton_length'] + 1e-5)
    
    df_enhanced['estimated_columns'] = np.where(
        df_enhanced['estimated_width'] <= 20, 1,
        np.where(df_enhanced['estimated_width'] <= 37, 2,
        np.where(df_enhanced['estimated_width'] <= 54, 3,
        np.where(df_enhanced['estimated_width'] <= 65, 4, 5))))
    
    df_enhanced['estimated_pitch'] = np.where(
        df_enhanced['estimated_columns'] == 1, 
        np.where(df_enhanced['estimated_width'] <= 20, 15, 10),
        np.where(df_enhanced['estimated_columns'] == 2, 13,
        np.where(df_enhanced['estimated_columns'] == 3, 15,
        np.where(df_enhanced['estimated_columns'] == 4, 15.5, 16.8))))
    
    df_enhanced['theoretical_led_count'] = (df_enhanced['skeleton_length'] / df_enhanced['estimated_pitch']) * df_enhanced['estimated_columns']
    
    return df_enhanced

def split_large_signs_by_complexity(df_huge):
    """
    大型看板を複雑さによって分割
    """
    if 'complexity' not in df_huge.columns:
        df_huge['complexity'] = df_huge['Peri'] ** 2 / (4 * np.pi * df_huge['Area'] + 1e-5)
    
    df_huge['size_segment'] = pd.qcut(df_huge['Area'], 3, labels=['large', 'very_large', 'extreme'], duplicates='drop')
    
    df_huge['complexity_segment'] = pd.qcut(df_huge['complexity'], 2, labels=['simple', 'complex'], duplicates='drop')
    
    df_huge['large_sign_segment'] = 'huge_' + df_huge['size_segment'].astype(str) + '_' + df_huge['complexity_segment'].astype(str)
    
    return df_huge

def build_specialized_large_sign_model(df_huge, X_cols, y_col='led'):
    """
    大型看板専用のモデルを構築
    """
    X = df_huge[X_cols]
    y = df_huge[y_col]
    
    X = X.fillna(X.median())
    
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'auto',
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 5
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("大型看板モデルの特徴量重要度:")
    print(feature_importance.head(10))
    
    return model, feature_importance

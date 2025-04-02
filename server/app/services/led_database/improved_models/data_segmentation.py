import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def advanced_segmentation(df, n_clusters=4):
    """
    クラスタリングを使用した高度なデータセグメンテーション
    """
    cluster_features = df[['Area', 'Peri', 'skeleton_length', 'distance_average', 'zunguri']].copy()
    
    cluster_features = cluster_features.fillna(cluster_features.median())
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    
    return df, kmeans, scaler

def get_segment_models_mapping(df):
    """
    セグメントに基づくモデルマッピングを生成
    """
    is_huge = ((df['Area'] > 20000) & (df['zunguri'] >= 15)) | (df['Area'] > 40000)
    is_neon = (df['distance_average'] <= 6.3) & ~is_huge
    
    segments = []
    
    df['segment'] = np.where(is_huge, 'huge', np.where(is_neon, 'neon', 'hyomen'))
    
    if 'cluster' in df.columns:
        df['detailed_segment'] = df['segment'] + '_cluster_' + df['cluster'].astype(str)
    
    return df

def get_specific_models_for_large_signs(df):
    """
    大型看板用の特別なセグメント
    """
    is_huge = ((df['Area'] > 20000) & (df['zunguri'] >= 15)) | (df['Area'] > 40000)
    df_huge = df[is_huge].copy()
    
    if len(df_huge) > 30:  # 十分なデータがある場合
        df_huge['area_quartile'] = pd.qcut(df_huge['Area'], 3, labels=['large', 'very_large', 'extreme'])
        df_huge['shape_type'] = pd.qcut(df_huge['zunguri'], 2, labels=['complex', 'simple'], duplicates='drop')
        
        df_huge['large_segment'] = 'huge_' + df_huge['area_quartile'].astype(str) + '_' + df_huge['shape_type'].astype(str)
        
        df.loc[is_huge, 'large_segment'] = df_huge['large_segment']
    else:
        df.loc[is_huge, 'large_segment'] = 'huge'
    
    return df

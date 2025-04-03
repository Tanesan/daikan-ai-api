import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def segment_by_size_and_complexity(df):
    """
    看板を以下のセグメントに分割：
    - small: Area <= 5000
    - medium: 5000 < Area <= 20000
    - large: 20000 < Area <= 40000 
    - very_large: Area > 40000
    """
    df['size_segment'] = np.where(df['Area'] <= 5000, 'small',
                         np.where(df['Area'] <= 20000, 'medium',
                         np.where(df['Area'] <= 40000, 'large', 'very_large')))
    
    if 'zunguri' not in df.columns:
        df['zunguri'] = df['Area'] / (df['Peri'] + 1e-5)
    
    df['complexity'] = np.where(df['zunguri'] < 15, 'complex', 'simple')
    
    is_neon = (df['distance_average'] <= 6.3) & (df['size_segment'] != 'very_large')
    df['emission_type'] = np.where(is_neon, 'neon', 'hyomen')
    
    df['segment'] = df['emission_type'] + '_' + df['size_segment'] + '_' + df['complexity']
    
    very_large_idx = df['size_segment'] == 'very_large'
    if very_large_idx.sum() > 0:
        df.loc[very_large_idx, 'segment'] = 'very_large_special'
    
    return df

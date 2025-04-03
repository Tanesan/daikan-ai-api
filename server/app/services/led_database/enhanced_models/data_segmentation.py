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
    df_copy = df.copy()
    
    df_copy.loc[:, 'size_segment'] = np.where(df_copy['Area'] <= 5000, 'small',
                                    np.where(df_copy['Area'] <= 20000, 'medium',
                                    np.where(df_copy['Area'] <= 40000, 'large', 'very_large')))
    
    if 'zunguri' not in df_copy.columns:
        df_copy.loc[:, 'zunguri'] = df_copy['Area'] / (df_copy['Peri'] + 1e-5)
    
    df_copy.loc[:, 'complexity'] = np.where(df_copy['zunguri'] < 15, 'complex', 'simple')
    
    is_neon = (df_copy['distance_average'] <= 6.3) & (df_copy['size_segment'] != 'very_large')
    df_copy.loc[:, 'emission_type'] = np.where(is_neon, 'neon', 'hyomen')
    
    df_copy.loc[:, 'segment'] = df_copy['emission_type'] + '_' + df_copy['size_segment'] + '_' + df_copy['complexity']
    
    very_large_idx = df_copy['size_segment'] == 'very_large'
    if very_large_idx.sum() > 0:
        df_copy.loc[very_large_idx, 'segment'] = 'very_large_special'
    
    return df_copy

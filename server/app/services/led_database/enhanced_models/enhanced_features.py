import numpy as np
import pandas as pd

def enhance_features_for_large_signs(df):
    """
    大型看板向けの特徴量エンジニアリング
    """
    df['area_per_skeleton'] = df['Area'] / df['skeleton_length']
    df['peri_per_skeleton'] = df['Peri'] / df['skeleton_length']
    df['area_per_peri'] = df['Area'] / df['Peri']
    
    df['estimated_width'] = df['Area'] / (df['skeleton_length'] + 1e-5)
    
    df['estimated_columns'] = np.where(
        df['estimated_width'] <= 20, 1,
        np.where(df['estimated_width'] <= 37, 2,
        np.where(df['estimated_width'] <= 54, 3,
        np.where(df['estimated_width'] <= 65, 4,
        np.where(df['estimated_width'] <= 90, 5,
        np.where(df['estimated_width'] <= 120, 6,
        np.where(df['estimated_width'] <= 150, 7, 8)))))))
    
    df['estimated_pitch'] = np.where(
        df['estimated_columns'] == 1, np.where(df['estimated_width'] <= 20, 15, 10),
        np.where(df['estimated_columns'] == 2, 10.5,
        np.where(df['estimated_columns'] == 3, 13.5,
        np.where(df['estimated_columns'] == 4, 15.5,
        np.where(df['estimated_columns'] == 5, 14.0, 12.0)))))
    
    df['scaling_factor'] = np.where(df['Area'] > 40000, 2.8, 
                           np.where(df['Area'] > 20000, 1.5, 1.0))
    
    df['theoretical_led_count'] = (df['skeleton_length'] / df['estimated_pitch']) * df['estimated_columns']
    
    if 'intersection3' in df.columns and 'endpoint' in df.columns:
        df['branch_correction_factor'] = ((df['intersection3'] + 2 * df.get('intersection4', 0)) / 
                                         (df['endpoint'] + 1))
        
        df['corrected_skeleton_length'] = df['skeleton_length'] * (1 + 0.1 * df['branch_correction_factor'])
        
        df['branch_adjusted_led'] = df['theoretical_led_count'] * (1 + 0.2 * df['branch_correction_factor'])
    else:
        df['branch_correction_factor'] = 1.0
        df['corrected_skeleton_length'] = df['skeleton_length']
        df['branch_adjusted_led'] = df['theoretical_led_count']
    
    df['final_skeleton_length'] = np.where(
        df['Area'] > 40000, df['corrected_skeleton_length'],
        np.where(df['Area'] > 20000, 
                (df['skeleton_length'] + df['corrected_skeleton_length']) / 2,
                df['skeleton_length'])
    )
    
    return df

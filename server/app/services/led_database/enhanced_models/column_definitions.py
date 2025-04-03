import pandas as pd

FEATURE_COLUMNS = [
    'Area',
    'Peri',
    'skeleton_length',
    'intersection3',
    'intersection4',
    'intersection5',
    'intersection6',
    'endpoint',
    'distance_average',
    'distance_min',
    'distance_max',
    'distance_median',
    'distance_mode',
    'zunguri',
    'area_per_skeleton',
    'peri_per_skeleton',
    'area_per_peri',
    'estimated_width',
    'estimated_columns',
    'estimated_pitch',
    'scaling_factor',
    'theoretical_led_count',
    'branch_correction_factor',
    'corrected_skeleton_length',
    'branch_adjusted_led',
    'final_skeleton_length'
]

REQUIRED_COLUMNS = [
    'Area',
    'Peri',
    'skeleton_length',
    'intersection3',
    'intersection4',
    'endpoint',
    'distance_average'
]

def standardize_dataframe(df, for_training=False):
    """
    データフレームの列を標準化する
    
    Args:
        df: 入力データフレーム
        for_training: 学習用かどうか（Trueの場合、'led'列も含める）
    
    Returns:
        標準化されたデータフレーム
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"必須カラムが不足しています: {missing_columns}")
    
    if for_training and 'led' not in df.columns:
        raise ValueError("学習用データフレームには'led'列が必要です")
    
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    
    if for_training:
        columns_to_use = FEATURE_COLUMNS + ['led']
    else:
        columns_to_use = FEATURE_COLUMNS
    
    available_columns = [col for col in columns_to_use if col in df.columns]
    
    return df[available_columns]

def get_feature_columns():
    """
    特徴量の列名リストを取得
    
    Returns:
        特徴量の列名リスト
    """
    return FEATURE_COLUMNS.copy()

def get_required_columns():
    """
    必須の列名リストを取得
    
    Returns:
        必須の列名リスト
    """
    return REQUIRED_COLUMNS.copy()

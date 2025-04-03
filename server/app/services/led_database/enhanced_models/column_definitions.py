"""
Column definitions module for LED prediction models.

This module defines standard column lists and provides functionality to
standardize DataFrame columns for consistent model training and prediction.
"""
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
    Standardize DataFrame columns for consistent model training and prediction.
    
    Ensures all required columns are present and adds default values for
    missing feature columns. For training data, also ensures the 'led' column
    is present.
    
    Args:
        df: Input DataFrame with sign parameters
        for_training: Whether the DataFrame is for training (True) or prediction (False)
    
    Returns:
        DataFrame with standardized columns
        
    Raises:
        ValueError: If required columns are missing or 'led' column is missing for training
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
    Get the list of feature column names.
    
    Returns:
        List of feature column names used in the models
    """
    return FEATURE_COLUMNS.copy()

def get_required_columns():
    """
    Get the list of required column names.
    
    Returns:
        List of column names that are required for prediction
    """
    return REQUIRED_COLUMNS.copy()

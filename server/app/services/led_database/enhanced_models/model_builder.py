import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import os
import joblib
import matplotlib.pyplot as plt
from app.services.led_database.enhanced_models.column_definitions import standardize_dataframe

def weighted_rmse_obj(preds, dtrain):
    """
    Custom objective function based on RMSE that increases penalty when prediction < actual (residual < 0).
    Enhances penalty to avoid underestimation of LED counts, especially for large signs.
    """
    labels = dtrain.get_label()
    residual = preds - labels
    
    alpha = 5.0 * np.log1p(labels) / np.log1p(labels.mean())  # Higher penalty for larger values
    
    weight = np.where(residual < 0, 1.0 + alpha, 1.0)
    
    grad = 2.0 * weight * residual
    hess = 2.0 * weight
    
    return grad, hess

def build_model_for_segment(X_train, y_train, X_test, y_test, segment_name):
    """
    Build a model for a specific segment.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        segment_name: Name of the segment
        
    Returns:
        Trained model and MAPE score
    """
    if 'very_large' in segment_name:
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'auto',
            'learning_rate': 0.03,
            'max_depth': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 1.5,
            'min_child_weight': 3
        }
    elif 'large' in segment_name:
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'auto',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.3,
            'reg_lambda': 1.0,
            'min_child_weight': 3
        }
    else:
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'auto',
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 1
        }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    
    if 'very_large' in segment_name or 'large' in segment_name:
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            obj=weighted_rmse_obj,
            evals=watchlist,
            early_stopping_rounds=20,
            verbose_eval=100
        )
    else:
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=watchlist,
            early_stopping_rounds=20,
            verbose_eval=100
        )
    
    y_pred = model.predict(dtest)
    
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"{segment_name} model MAPE: {mape:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{segment_name} model (MAPE: {mape:.4f})')
    plt.grid(True)
    
    os.makedirs('../enhanced_models/plots', exist_ok=True)
    plt.savefig(f'../enhanced_models/plots/{segment_name}_model_performance.png')
    
    return model, mape

def train_all_models(df, output_dir='../enhanced_models/models'):
    """
    Train and save models for all segments.
    
    Args:
        df: DataFrame containing features and labels
        output_dir: Directory to save models
        
    Returns:
        DataFrame with model results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_results = pd.DataFrame(columns=['segment', 'mape', 'model_path', 'scaler_path'])
    
    for segment in df['segment'].unique():
        segment_df = df[df['segment'] == segment].copy()
        
        if len(segment_df) < 30:
            print(f"Segment {segment} has insufficient data ({len(segment_df)} < 30). Skipping.")
            continue
        
        standardized_df = standardize_dataframe(segment_df, for_training=True)
        
        X = standardized_df.drop(['led'], axis=1)
        y = standardized_df['led']
        
        X = X.select_dtypes(include=['number'])
        
        X = X.fillna(X.median())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model, mape = build_model_for_segment(X_train, y_train, X_test, y_test, segment)
        
        model_path = os.path.join(output_dir, f"{segment}_model.json")
        model.save_model(model_path)
        
        new_row = pd.DataFrame({
            'segment': [segment],
            'mape': [mape],
            'model_path': [model_path],
        })
        model_results = pd.concat([model_results, new_row], ignore_index=True)
        
        print(f"Saved model for segment {segment}: {model_path}")
    
    model_results.to_csv(os.path.join(output_dir, 'model_results.csv'), index=False)
    print(f"Saved model results: {os.path.join(output_dir, 'model_results.csv')}")
    
    return model_results

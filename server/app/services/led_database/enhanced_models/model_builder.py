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
    RMSEをベースにしつつ、予測値 < 実測値（residual < 0）の場合にペナルティを大きくする。
    特に大型看板の過少予測を避けるために、ペナルティを強化。
    """
    labels = dtrain.get_label()
    residual = preds - labels
    
    alpha = 5.0 * np.log1p(labels) / np.log1p(labels.mean())  # 値が大きいほどペナルティ大
    
    weight = np.where(residual < 0, 1.0 + alpha, 1.0)
    
    grad = 2.0 * weight * residual
    hess = 2.0 * weight
    
    return grad, hess

def build_model_for_segment(X_train, y_train, X_test, y_test, segment_name):
    """
    特定のセグメント用のモデルを構築
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
    print(f"{segment_name} モデルのMAPE: {mape:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
    plt.xlabel('実測値')
    plt.ylabel('予測値')
    plt.title(f'{segment_name} モデル (MAPE: {mape:.4f})')
    plt.grid(True)
    
    os.makedirs('../enhanced_models/plots', exist_ok=True)
    plt.savefig(f'../enhanced_models/plots/{segment_name}_model_performance.png')
    
    return model, mape

def train_all_models(df, output_dir='../enhanced_models/models'):
    """
    全セグメント用のモデルをトレーニングして保存する
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_results = pd.DataFrame(columns=['segment', 'mape', 'model_path', 'scaler_path'])
    
    for segment in df['segment'].unique():
        segment_df = df[df['segment'] == segment].copy()
        
        if len(segment_df) < 30:
            print(f"セグメント {segment} のデータ数が不足しています ({len(segment_df)} < 30)。スキップします。")
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
        
        print(f"セグメント {segment} のモデルを保存しました: {model_path}")
    
    model_results.to_csv(os.path.join(output_dir, 'model_results.csv'), index=False)
    print(f"モデル結果を保存しました: {os.path.join(output_dir, 'model_results.csv')}")
    
    return model_results

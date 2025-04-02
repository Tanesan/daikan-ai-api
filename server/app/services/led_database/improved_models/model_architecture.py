import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

def weighted_rmse_obj(preds, dtrain):
    """
    XGBoostの重み付きRMSE目的関数（既存のものを再利用）
    """
    labels = dtrain.get_label()
    residual = preds - labels
    alpha = 3.0  # 負の誤差に対するペナルティ係数
    
    weight = np.where(residual < 0, 1.0 + alpha, 1.0)
    
    grad = 2.0 * weight * residual
    hess = 2.0 * weight
    
    return grad, hess

def weighted_rmse_eval(preds, dtrain):
    """
    XGBoostの重み付きRMSE評価関数（既存のものを再利用）
    """
    labels = dtrain.get_label()
    residual = preds - labels
    alpha = 2.0
    negative_indicator = (residual < 0).astype(float)
    weight = 1.0 + alpha * negative_indicator
    weighted_mse = np.mean(weight * (residual ** 2))
    weighted_rmse = np.sqrt(weighted_mse)
    return "weighted_rmse", weighted_rmse

def build_segment_model(X_train, y_train, X_val, y_val, segment_name, feature_selection=True):
    """
    各セグメント用のモデルを構築
    feature_selection: 特徴量選択を行うかどうか
    """
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    X_train = X_train[numeric_cols]
    X_val = X_val[numeric_cols]
    
    X_train = X_train.fillna(X_train.median())
    X_val = X_val.fillna(X_train.median())  # 訓練データの中央値で検証データも埋める
    
    print(f"数値型特徴量の数: {len(numeric_cols)}")
    
    if feature_selection and X_train.shape[1] > 5:  # 十分な特徴量がある場合
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train, y_train)
        
        importance = xgb_model.feature_importances_
        indices = np.argsort(importance)[-min(20, len(importance)):]  # 上位20個または全部
        selected_features = X_train.columns[indices]
        
        if len(selected_features) < 3:  # 最低限の特徴量数を確保
            correlations = []
            for col in X_train.columns:
                corr = abs(X_train[col].corr(y_train))
                correlations.append(corr)
            
            indices = np.argsort(correlations)[-5:]
            selected_features = X_train.columns[indices]
        
        X_train_selected = X_train[selected_features]
        X_val_selected = X_val[selected_features]
    else:
        selected_features = X_train.columns
        X_train_selected = X_train
        X_val_selected = X_val
    
    params = {
        'tree_method': 'auto',
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.01,
        'reg_lambda': 1.0,
        'min_child_weight': 3
    }
    
    dtrain = xgb.DMatrix(X_train_selected, label=y_train)
    dval = xgb.DMatrix(X_val_selected, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        obj=weighted_rmse_obj,
        custom_metric=weighted_rmse_eval,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False
    )
    
    y_pred = model.predict(dval)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"Segment: {segment_name}, MAPE: {mape:.4f}, RMSE: {rmse:.4f}, Features: {len(selected_features)}")
    
    return model, selected_features, mape

def build_ensemble_model(X_train, y_train, X_val, y_val, n_folds=5):
    """
    交差検証ベースのスタッキングアンサンブルモデルを構築
    """
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    X_train = X_train[numeric_cols]
    X_val = X_val[numeric_cols]
    
    X_train = X_train.fillna(X_train.median())
    X_val = X_val.fillna(X_train.median())
    
    base_params = {
        'tree_method': 'auto',
        'learning_rate': 0.1,
        'objective': 'reg:squarederror'
    }
    
    base_models = [
        xgb.XGBRegressor(**base_params, max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.01, reg_lambda=1.0),
        xgb.XGBRegressor(**base_params, max_depth=6, subsample=0.7, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=0.1),
        xgb.XGBRegressor(**base_params, max_depth=8, subsample=0.9, colsample_bytree=0.7, reg_alpha=0.001, reg_lambda=10)
    ]
    
    ensemble = StackingRegressor(
        estimators=[(f"model_{i}", model) for i, model in enumerate(base_models)],
        final_estimator=xgb.XGBRegressor(**base_params, max_depth=3),
        cv=KFold(n_splits=n_folds, shuffle=True, random_state=42)
    )
    
    ensemble.fit(X_train, y_train)
    
    y_pred = ensemble.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"Ensemble Model - MAPE: {mape:.4f}, RMSE: {rmse:.4f}")
    
    return ensemble, mape

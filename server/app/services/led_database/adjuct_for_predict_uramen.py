from ast import literal_eval
from collections import Counter
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# SHAP, Permutation Importance 用
import shap
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# シード固定
np.random.seed(42)
def get_cols_pitch_for_uniform(dist_val, area):
    dist_val *= 2
    """
    均一線幅の場合の判定ロジック
    dist_val: ある1つの distance の値 (float)
    戻り値: (col, pitch)
    """
    if dist_val <= 20:
        return (1, 15.0)
    elif dist_val <= 25:
        return (1, 10.0)
    elif dist_val <= 37:
        return (2, 15.0)
    elif dist_val <= 54:
        return (3, 17.0)
    elif dist_val <= 70:
        return (4, 19.0)
    elif dist_val <= 100:
        return (5, 18.0)
    elif dist_val <= 130:
        return (7, 12.0)
    else:
        return (8, 11.0)
def get_cols_pitch_for_mixed(dist_val, area):
    dist_val *= 2
    """
    混在配列の場合の判定ロジック
    dist_val: ある1つの distance の値 (float)
    戻り値: (col, pitch)
    """
    if dist_val <= 20:
        return (1, 7.5)    # 1列 / 7.5mm
    elif dist_val <= 30:
        return (2, 10.5)  # 2列 / 10~11mm → ここでは 10.5
    elif dist_val <= 50:
        return (3, 13.5)  # 3列 / 13~14mm → ここでは 13.5
    elif dist_val <= 65:
        return (4, 15.5)
    elif dist_val <= 130:
        if area >= 50000:
            return (5, 14.0)
        else:
            return (5, 16.8)
    elif dist_val <= 130:
            return (6, 15.8)
    elif dist_val <= 150:
            return (7, 12.0)
    else:
        return (8, 11.0)

def weighted_rmse_eval(preds, dtrain):
    """
    RMSEをベースにしつつ、予測値 < 実測値（residual < 0）の場合にペナルティを大きくする。
    alpha を大きくするほど、負の誤差に対するペナルティが強まる。
    """
    labels = dtrain.get_label()
    residual = preds - labels
    alpha = 2.0  # この値を調整することで負の誤差の重みづけを変えられる
    negative_indicator = (residual < 0).astype(float)
    weight = 1.0 + alpha * negative_indicator
    weighted_mse = np.mean(weight * (residual ** 2))
    weighted_rmse = np.sqrt(weighted_mse)
    return "weighted_rmse", weighted_rmse

def weighted_rmse_obj(preds: np.ndarray, dtrain: xgb.DMatrix):
    labels = dtrain.get_label()
    residual = preds - labels
    alpha = 3.0  # これを大きくすると負の誤差のペナルティがより強くなる

    # 負の誤差に対して重み w = 1 + alpha
    # (residual < 0) なら 1 + alpha, そうでなければ 1.0
    weight = np.where(residual < 0, 1.0 + alpha, 1.0)

    # ロス: w * (residual^2)
    # => grad = 2 * w * residual
    # => hess = 2 * w
    grad = 2.0 * weight * residual
    hess = 2.0 * weight

    return grad, hess

def apply_heuristic_pitch_adjustment(df):
    """
    df: 以下の列を含む DataFrame を想定
        - pj: 看板ID (同じIDが同一看板)
        - heuristic_cols: 列数(ヒューリスティック計算後)
        - heuristic_pitch: ピッチ(ヒューリスティック計算後)

    ロジック:
      - まず 'heuristic_size' = heuristic_cols * heuristic_pitch を計算
      - 同一看板内で、他の文字の 'heuristic_size' が自分より5%以上小さい
        (＝他の文字の size が self_size * 0.95 より小さい) 行が1つでもある場合、
        → 自分の pitch を -3 する（下限0など必要なら適宜実装）
    """

    # step1: 'heuristic_size' 列を作る
    df['heuristic_size'] = df['heuristic_cols'] * df['heuristic_pitch']

    # step2: groupby で pj ごとに処理
    def adjust_pitch_for_sign(sign_df):
        """
        同一 pj 内のデータフレームに対して、ピッチを調整した結果を返す。
        """
        sign_df = sign_df.copy()  # groupごとにコピー

        # 新しい列 'final_pitch' を作り、初期値は 'heuristic_pitch' にしておく
        sign_df['tumeru'] = 0

        for i in sign_df.index:
            self_size = sign_df.loc[i, 'heuristic_size']

            # "他の文字が 5%以上詰めていた" ＝ その文字の size < self_size * 0.95
            # (どれか1つでも該当すれば詰めているとみなす)
            others = sign_df.drop(i)  # 自分以外
            condition = others['heuristic_size'] < (self_size * 0.95)

            if condition.any():
                # ピッチを -3
                # new_pitch = sign_df.loc[i, 'final_pitch'] - 3.0
                # 0 未満にはならないようにクリップする場合は max(0, new_pitch)
                sign_df.loc[i, 'tumeru'] = 1

        return sign_df

    # groupby + apply で pj ごとに処理した結果を一つにまとめる
    df = df.groupby('pj', group_keys=False).apply(adjust_pitch_for_sign)

    return df

def determine_led_arrangement(row, std_threshold=5.0):
    """
    row: df の1行 (Series)
      - row['distance'] がリストとして格納されている想定
      - row['distance_std'] なども入っている想定
    std_threshold: この値より小さければ「均一線幅」とみなす仮の基準

    戻り値: (avg_col, avg_pitch)
      - 全distanceに対して列数・ピッチを求め、平均を返す
    """
    dist_std = row.get('distance_std', np.nan)
    distances = row.get('distance', [])

    # distance が空 (or NaN) の場合
    if not isinstance(distances, list) or len(distances) == 0:
        return (0, 0.0)

    # 均一線幅かどうかを std_threshold で判定
    is_uniform = (dist_std < std_threshold) if pd.notna(dist_std) else False

    col_list = []
    pitch_list = []

    # すべての distance に対して処理
    for dist_val in distances:
        if is_uniform:
            col, pitch = get_cols_pitch_for_uniform(dist_val, row.get('area', 0))
        else:
            col, pitch = get_cols_pitch_for_mixed(dist_val, row.get('area', 0))

        col_list.append(col)
        pitch_list.append(pitch)

    # 平均を計算 (列数は整数にするか、またはfloatにするかは要件次第)
    avg_col = np.mean(col_list)
    avg_pitch = np.mean(pitch_list)

    # 列数を四捨五入したい場合は以下のように:
    # avg_col = int(round(avg_col))

    return (avg_col, avg_pitch)


def add_heuristic_cols_and_pitch(df, std_threshold=5.0):
    """
    上記 determine_led_arrangement を df に適用し、
    (avg_col, avg_pitch) を2列に追加する。
    """
    result = df.apply(lambda row: determine_led_arrangement(row, std_threshold=std_threshold), axis=1)
    df['heuristic_cols']  = result.apply(lambda x: x[0])
    df['heuristic_pitch'] = result.apply(lambda x: x[1])
    return df

def train_xgboost_with_optuna(X, y, X_test, y_test, n_trials=40):
    """
    与えられた X, y について、
    Optuna でパラメータ探索 → ベストモデルを返す
    """
    # DMatrix 化
    dtrain_all = xgb.DMatrix(X, label=y)
    dvalid_all = xgb.DMatrix(X_test, label=y_test)

    # objective 関数
    def objective(trial):
        params = {
            'eval_metric': 'rmse',  # 学習中モニター用
            'tree_method': 'auto',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        }

        # 便宜上、学習データをそのまま watchlist に (本来は validation 分割するのが望ましい)
        booster = xgb.train(
            params,
            dtrain_all,
            obj=weighted_rmse_obj,
            num_boost_round=300,
            custom_metric=weighted_rmse_eval,
            verbose_eval=False
        )
        # ここでは MAPE を計算するため、学習データに対する予測を取る
        y_pred = booster.predict(dtrain_all)
        mape_value = mean_absolute_percentage_error(y, y_pred)
        return mape_value

    # Optuna 実行
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ベストパラメータを取得
    best_trial = study.best_trial
    best_params = {
        'tree_method': 'auto'
    }
    best_params.update(best_trial.params)
    print("Best MAPE:", best_trial.value)
    print("Best params:", best_trial.params)

    # ベストパラメータで最終モデルを再学習
    model = xgb.train(
        params=best_params,
        dtrain=dtrain_all,
        num_boost_round=300,
        verbose_eval=True,
        obj=weighted_rmse_obj,
        custom_metric=weighted_rmse_eval,
        evals=[(dtrain_all, "train"), (dvalid_all, "val")],
        early_stopping_rounds=20
    )
    return model

def preprocess(df):
    """
    distance 配列から各種統計量を取り出す前処理。
    すでにある程度動いている想定に、例外ハンドリングを少し追加。
    """

    def filter_distance(dist_list):
        if not isinstance(dist_list, list):
            return []
        # 0.5 以下の値は除外
        return [d for d in dist_list if d > 0.5]

    df['distance'] = df['distance'].apply(filter_distance)

    # 各種統計量 (std, count, sum, percentile)
    df['distance_std'] = df['distance'].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
    df['distance_count'] = df['distance'].apply(len)
    df['distance_sum'] = df['distance'].apply(sum)
    df['distance_50%'] = df['distance'].apply(lambda x: np.percentile(x, 50) if len(x) > 0 else np.nan)
    df['distance_75%'] = df['distance'].apply(lambda x: np.percentile(x, 75) if len(x) > 0 else np.nan)
    df['distance_average'] = df['distance'].apply(lambda x: np.mean(x))
    return df


def add_num_objects(df):
    """
    ・distance の最頻値/2番目の最頻値
    ・max が全体の 1~5% を占めるかどうかの判定
    """

    def process_distances(distances):
        # distが空なら、全て 0 または NaN などで返す
        if not distances:
            return {
                'num_objects': 0,
                'max_val': 0,
                'max_in_1_5_percent': False,
                'mode_1': np.nan,
                'mode_2': np.nan
            }

        data = Counter(distances)

        # 全要素数
        total_count = sum(data.values())
        # 最大値
        max_val = max(distances)

        # max_val が全体の 1~5% に収まるかチェック
        max_count = data[max_val]
        max_ratio = max_count / total_count * 100
        is_in_range = (1 <= max_ratio <= 5)

        # 上位の最頻値を取得
        # most_common(n) で上位 n 個の (値, 頻度) ペアを取れる
        most_common_list = data.most_common()  # [(value, count), (value, count), ...] 順に多い
        mode_1 = most_common_list[0][0] if len(most_common_list) > 0 else np.nan
        mode_2 = most_common_list[1][0] if len(most_common_list) > 1 else np.nan

        # 1% 以下の雑多な値を除外する例(必要なら)
        # filtered = {key: data[key] for key in data if data[key] / total_count > 0.01}

        return {
            'max_val': max_val,
            'max_in_1_5_percent': is_in_range,
            'mode_1': mode_1,
            'mode_2': mode_2
        }

    # apply で各行の distance に対して処理
    temp = df['distance'].apply(process_distances).apply(pd.Series)

    # dfに列として追加
    df['distance_max'] = temp['max_val']
    df['max_in_1_5_percent'] = temp['max_in_1_5_percent']
    df['distance_mode_1'] = temp['mode_1']
    df['distance_mode_2'] = temp['mode_2']

    return df


# ================================================================
# 1. ダミーデータの作成
#    実際には csv 読み込みや画像処理の結果を読み込む等に置き換えてください
# ================================================================

df = pd.read_csv("predict_database_uramen.tsv", sep="\t")

# ================================================================
# 2. 学習データとテストデータに分割
# ================================================================
# 特徴量と目的変数に分ける
df = df.dropna(subset=['distance'])
df['distance'] = df['distance'].apply(literal_eval)
df = preprocess(df)
df = add_num_objects(df)
df = add_heuristic_cols_and_pitch(df)
df = apply_heuristic_pitch_adjustment(df)
df = df[df['distance_count'] != 1]
df["zunguri"] = df['Area'] / (df['Peri'] + 1e-5)
df['log_Area'] = np.log1p(df['Area'])  # 対数変換
df['sqrt_Area'] = np.sqrt(df['Area'])  # 平方根変換
df['heuristic_pitch'] = df['heuristic_pitch'] + 1 * df['tumeru']
df["ruled_number"] = df["skeleton_length"] / 15
df["ruled_number_15"] = df["Area"] / 15 / 15
df["adj"] =  df['heuristic_cols'] * df['heuristic_pitch']

X_uramen = df[['processed_path', 'skeleton_length', "ruled_number", "ruled_number_15",
         'distance_mode_2', "zunguri", "distance_count",
                'Peri']]

y_uramen = df['led']
X_uramen_features = X_uramen.drop('processed_path', axis=1)


X_train, X_test, y_train, y_test, proj_train, proj_test = train_test_split(
    X_uramen_features, y_uramen, X_uramen['processed_path'],
    test_size=0.2, random_state=42
)

print("=== Train Special Model ===")
model_special = train_xgboost_with_optuna(X_train, y_train, X_test, y_test, n_trials=15)

dtest_s = xgb.DMatrix(X_test)
y_pred_special = model_special.predict(dtest_s)
mape_special = mean_absolute_percentage_error(y_test, y_pred_special)

print(f"MAPE (special model) : {mape_special:.3f}")
# 7) SHAP 可視化 (オプション)
#    TreeExplainerは xgb.train() で得た booster を直接渡せる

explainer_special = shap.TreeExplainer(model_special)
shap_values_s = explainer_special.shap_values(X_test)
shap.summary_plot(shap_values_s, X_test, feature_names=X_test.columns)


results_special_df = X_test.copy()
results_special_df['y_true'] = y_test.values
results_special_df['y_pred'] = y_pred_special
results_special_df['ape']    = np.abs(results_special_df['y_true'] - results_special_df['y_pred']) / results_special_df['y_true']
results_special_df['mape']   = results_special_df['ape'] * 100
results_special_df['sa']   = np.abs(results_special_df['y_true'] - results_special_df['y_pred'])
results_special_df['model']  = 'special'
results_special_df['processed_path'] = proj_test.values


plt.figure(figsize=(6, 6))
plt.scatter(results_special_df['y_true'], results_special_df['y_pred'], alpha=0.7, c='blue')
plt.plot([0, results_special_df['y_true'].max()], [0, results_special_df['y_true'].max()], 'r--')  # 対角線
plt.xlabel("Actual (y_true)")
plt.ylabel("Predicted (y_pred)")
plt.title("Actual vs. Predicted (uramen)")
plt.grid(True)
plt.show()


# 結合して csv などに出力してもOK
results_df = pd.concat([results_special_df], axis=0)
results_df.to_csv("results_two_models.tsv", sep='\t', index=False)
print("\n--- Combined Results ---")
print(results_df.head(10))
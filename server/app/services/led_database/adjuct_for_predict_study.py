from ast import literal_eval
from collections import Counter

import joblib
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

df = pd.read_csv("predict_database.tsv", sep="\t")

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
df["ruled_number"] = df["Area"] / ((df["heuristic_pitch"] - (df['heuristic_cols'] - 1))**2)
df["ruled_number_15"] = df["Area"] / 15 / 15
df["adj"] =  df['heuristic_cols'] * df['heuristic_pitch']
df.loc[df["Area"] >= 16000, "ruled_number"] = df["ruled_number_15"]
# 例: 特定の条件を満たすものは「specialモデル」で学習したい
#     ここでは仮に "distance_count == 1" が条件とする
condition = (df['distance_average'] <= 6.3) & ~(
    ((df['Area'] > 25000) & (df['zunguri'] >= 20)) |
    (df['Area'] > 50000)
)

# special データ (条件を満たす)
df_neon = df[condition].copy()
# main データ (条件を満たさない)
df_hyomen = df[~condition].copy()

is_so_huge = (
    (df_hyomen['Area'] > 25000) & (df_hyomen['zunguri'] >= 20)
) | (df_hyomen['Area'] > 50000)
df_huge_hyomen = df_hyomen[is_so_huge].copy()
df_hyomen = df_hyomen[~is_so_huge].copy()

print(f"df_special: {len(df_neon)} samples")
print(f"df_main   : {len(df_hyomen)} samples")
X_huge_hyomen = df_huge_hyomen[['processed_path', 'skeleton_length', 'endpoint',
         'distance_mode_2', 'heuristic_pitch', "ruled_number_15",
        'Area', 'Peri']]

y_huge_hyomen = df_huge_hyomen['led']

X_hyomen = df_hyomen[['processed_path', 'skeleton_length', 'intersection3', 'endpoint',
         'distance_mode_2', 'heuristic_pitch', "ruled_number_15", "adj",
        'Area', 'Peri', 'distance_std', 'distance_average', 'ruled_number']]

y_hyomen = df_hyomen['led']

df_neon['too_small'] = df_neon['Area'].apply(lambda x: 2 if x > 100 else 0)
df_neon['heuristic_pitch'] -= df_neon['too_small']
df_neon['ruled_number'] = df_neon["skeleton_length"] / df_neon['heuristic_pitch']
X_neon = df_neon[['processed_path', 'skeleton_length', 'distance_average','zunguri', 'ruled_number', 'too_small']]
results_df = pd.concat([df_hyomen, df_neon], axis=0)
y_neon = df_neon['led']
# 1. Project ID 以外の特徴量を取得
X_huge_hyomen_features = X_huge_hyomen.drop('processed_path', axis=1)
X_hyomen_features = X_hyomen.drop('processed_path', axis=1)
X_neon_features = X_neon.drop('processed_path', axis=1)


scaler_h = StandardScaler()
X_hyomen_features = scaler_h.fit_transform(X_hyomen_features)
scaler_n = StandardScaler()
X_neon_features = scaler_n.fit_transform(X_neon_features)
scaler_hh = StandardScaler()
X_huge_features = scaler_hh.fit_transform(X_huge_hyomen_features)



# hyomen
best_hyomen_params = {'learning_rate': 0.14189970387806142, 'max_depth': 10, 'subsample': 0.8280972385043891, 'colsample_bytree': 0.6722294857485422, 'reg_alpha': 0.034605355995371116, 'reg_lambda': 0.003679941045678635, 'min_child_weight': 7}


# neon
best_neon_params = {'learning_rate': 0.21661621529916492, 'max_depth': 10, 'subsample': 0.9004837160703227, 'colsample_bytree': 0.6346358246267731, 'reg_alpha': 0.014806496994843677, 'reg_lambda': 0.014789059706982749, 'min_child_weight': 1}

# huge
best_huge_params = {'learning_rate': 0.24951403960709495, 'max_depth': 10, 'subsample': 0.9925823438580981, 'colsample_bytree': 0.5219281882850777, 'reg_alpha': 1.2075411328576848e-08, 'reg_lambda': 7.372562302828144e-08, 'min_child_weight': 1}




dtrain_hyomen = xgb.DMatrix(X_hyomen_features, label=y_hyomen)
dtrain_neon = xgb.DMatrix(X_neon_features, label=y_neon)
dtrain_huge = xgb.DMatrix(X_huge_features, label=y_huge_hyomen)

model_huge = xgb.train(
        best_huge_params,
        dtrain_huge,
        num_boost_round=300,
        obj=weighted_rmse_obj,
        verbose_eval=False
)


model_hyomen = xgb.train(
        best_hyomen_params,
        dtrain_hyomen,
        num_boost_round=300,
        obj=weighted_rmse_obj,
        verbose_eval=False
)

model_neon = xgb.train(
        best_neon_params,
        dtrain_neon,
        num_boost_round=300,
        obj=weighted_rmse_obj,
        verbose_eval=False
)

model_hyomen.save_model("hyomen_model.json")
model_neon.save_model("hyomen_thin_model.json")
model_huge.save_model("hyomen_huge_model.json")
joblib.dump(scaler_h, "hyomen_scaler.pkl")
joblib.dump(scaler_n, "hyomen_thin_scaler.pkl")
joblib.dump(scaler_hh, "hyomen_huge_scaler.pkl")

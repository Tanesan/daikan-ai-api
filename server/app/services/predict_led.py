import os
import sys
import pandas as pd
import numpy as np
from ast import literal_eval

from numba.core.callconv import ErrorModel
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from collections import Counter
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from ast import literal_eval
import logging
# スケーラーとモデルをグローバルに保持
scalers = {}
models = {}
mape_scores = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
improved_models_path = os.path.join(current_dir, 'led_database', 'improved_models')
minimal_model_path = os.path.join(improved_models_path, 'models', 'minimal_filtered_model.json')

MINIMAL_MODEL_AVAILABLE = os.path.exists(minimal_model_path)
if MINIMAL_MODEL_AVAILABLE:
    logger.info("最小限フィルタリングモデルが利用可能です")
else:
    logger.warning("最小限フィルタリングモデルが見つかりません")
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
    df = adjust_pitch_for_sign(df)
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


# モデルとスケーラーを読み込む
def load_model_scaler(condition):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    led_database_dir = os.path.join(ROOT_DIR, 'led_database')

    model_filename = os.path.join(led_database_dir, f"{condition}_model.json")
    scaler_filename = os.path.join(led_database_dir, f"{condition}_scaler.pkl")

    # モデルを読み込み
    model = xgb.XGBRegressor()
    model.load_model(model_filename)

    # スケーラーを読み込み
    scaler = joblib.load(scaler_filename)

    return model, scaler


# Web APIで予測を行う関数
def predict_with_minimal_model(df, scale_ratio=1.0):
    """
    最小限フィルタリングモデルを使用してLEDの数を予測する関数
    
    Args:
        df: 入力データフレーム
        scale_ratio: スケール比率（デフォルト: 1.0）
    
    Returns:
        予測されたLED数（整数）またはNone（エラー時）
    """
    if not MINIMAL_MODEL_AVAILABLE:
        logger.warning("最小限フィルタリングモデルが利用できないため、従来のモデルで予測します")
        return None
    
    try:
        df_enhanced = df.copy()
        
        df_enhanced['scale_ratio'] = scale_ratio
        
        if 'distance' in df_enhanced.columns:
            if isinstance(df_enhanced['distance'].iloc[0], list):
                df_enhanced['distance_average'] = df_enhanced['distance'].apply(
                    lambda x: sum(x) / len(x) if x and len(x) > 0 else 0
                )
            else:
                logger.warning("distanceがリスト形式ではありません。distance_averageを0に設定します。")
                df_enhanced['distance_average'] = 0
        else:
            logger.warning("distanceカラムが見つかりません。distance_averageを0に設定します。")
            df_enhanced['distance_average'] = 0
        
        df_enhanced['area_per_skeleton'] = df_enhanced['Area'] / df_enhanced['skeleton_length']
        df_enhanced['peri_per_skeleton'] = df_enhanced['Peri'] / df_enhanced['skeleton_length']
        df_enhanced['area_per_peri'] = df_enhanced['Area'] / df_enhanced['Peri']
        
        if 'led' not in df_enhanced.columns:
            df_enhanced['led'] = 0
        df_enhanced['led_density_area'] = df_enhanced['led'] / df_enhanced['Area']
        df_enhanced['led_density_skeleton'] = df_enhanced['led'] / df_enhanced['skeleton_length']
        df_enhanced['led_density_peri'] = df_enhanced['led'] / df_enhanced['Peri']
        
        df_enhanced['area_skeleton_ratio'] = df_enhanced['Area'] / (df_enhanced['skeleton_length'] ** 2)
        df_enhanced['peri_area_ratio'] = df_enhanced['Peri'] / np.sqrt(df_enhanced['Area'])
        
        model = xgb.XGBRegressor()
        model.load_model(minimal_model_path)
        
        expected_feature_order = [
            'skeleton_length', 'scale_ratio', 'intersection3', 'intersection4', 
            'intersection5', 'intersection6', 'endpoint', 'Area', 'Peri', 
            'distance_average', 'area_per_skeleton', 'peri_per_skeleton', 
            'area_per_peri', 'led_density_area', 'led_density_skeleton', 
            'led_density_peri', 'area_skeleton_ratio', 'peri_area_ratio'
        ]
        
        available_features = [f for f in expected_feature_order if f in df_enhanced.columns]
        
        X = df_enhanced[available_features]
        logger.info(f"予測に使用する特徴量: {X.columns.tolist()}")
        prediction = model.predict(X)
        
        return round(prediction[0])
    except Exception as e:
        logger.error(f"最小限フィルタリングモデルでの予測中にエラーが発生しました: {e}")
        logger.exception(e)
        return None

def predict_led(area, peri, distance, skeleton_length, scale_ratio, intersection3, intersection4, intersection5,
                intersection6, endpoint, condition):
    # テストデータの準備
    df = pd.DataFrame([{
        'Area': area,
        'Peri': peri,
        'distance': distance,
        'skeleton_length': skeleton_length,
        'intersection3': intersection3,
        'intersection4': intersection4,
        'intersection5': intersection5,
        'intersection6': intersection6,
        'endpoint': endpoint,
    }])
    
    if MINIMAL_MODEL_AVAILABLE:
        minimal_prediction = predict_with_minimal_model(df, scale_ratio=scale_ratio)
        if minimal_prediction is not None:
            logger.info("最小限フィルタリングモデルで予測しました")
            return minimal_prediction

    if condition.name.lower() == "uramen":
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
        df["adj"] = df['heuristic_cols'] * df['heuristic_pitch']
        X_uramen_features = df[['skeleton_length', "ruled_number", "ruled_number_15",
         'distance_mode_2', "zunguri", "distance_count",
                'Peri']]
        if len(df) > 0:
            X_test_scaled = scaler_uramen.transform(X_uramen_features)
            dtest = xgb.DMatrix(X_test_scaled)
            y_pred = model_uramen.predict(dtest)
            logger.info("裏面のモデルで実行しました。")
            return round(y_pred[0])
        else:
            return 0

    if condition.name.lower() == "hyomen" or "neon":
        df = df.dropna(subset=['distance'])
        # df['distance'] = df['distance'].apply(literal_eval)
        df = preprocess(df)
        df = add_num_objects(df)
        df = add_heuristic_cols_and_pitch(df)
        df = apply_heuristic_pitch_adjustment(df)
        df = df[df['distance_count'] != 1]
        df["zunguri"] = df['Area'] / (df['Peri'] + 1e-5)
        df['log_Area'] = np.log1p(df['Area'])  # 対数変換
        df['sqrt_Area'] = np.sqrt(df['Area'])  # 平方根変換
        df['heuristic_pitch'] = df['heuristic_pitch'] + 1 * df['tumeru']
        df["ruled_number"] = df["Area"] / ((df["heuristic_pitch"] - (df['heuristic_cols'] - 1)) ** 2)
        df["ruled_number_15"] = df["Area"] / 15 / 15
        df["adj"] = df['heuristic_cols'] * df['heuristic_pitch']
        condition = (df['distance_average'] <= 6.3) & ~(
                ((df['Area'] > 20000) & (df['zunguri'] >= 15)) |  # Area基準を25000→20000に、zunguri基準を20→15に緩和
                (df['Area'] > 40000)  # Area単体の基準も50000→40000に緩和
        )
        df_neon = df[condition].copy()
        df_hyomen = df[~condition].copy()
        is_so_huge = (
                             (df_hyomen['Area'] > 20000) & (df_hyomen['zunguri'] >= 15)
                     ) | (df_hyomen['Area'] > 40000)
        df_huge_hyomen = df_hyomen[is_so_huge].copy()
        df_hyomen = df_hyomen[~is_so_huge].copy()
        y_pred = None
        if len(df_hyomen) > 0:
            X_hyomen_features = df_hyomen[['skeleton_length', 'intersection3', 'endpoint',
                                  'distance_mode_2', 'heuristic_pitch', "ruled_number_15", "adj",
                                  'Area', 'Peri', 'distance_std', 'distance_average', 'ruled_number']]
            X_test_scaled = scaler_hyomen.transform(X_hyomen_features)
            dtest = xgb.DMatrix(X_test_scaled)
            y_pred = model_hyomen.predict(dtest)
            logger.info("表面のモデルで実行しました。")
        if len(df_neon) > 0:
            df_neon['too_small'] = df_neon['Area'].apply(lambda x: 2 if x > 100 else 0)
            df_neon['heuristic_pitch'] -= df_neon['too_small']
            df_neon['ruled_number'] = df_neon["skeleton_length"] / df_neon['heuristic_pitch']
            df_neon.to_csv("test_neon.csv", index=False)
            X_neon_features = df_neon[['skeleton_length', 'distance_average','zunguri', 'ruled_number', 'too_small']]
            X_test_scaled = scaler_neon.transform(X_neon_features)
            dtest = xgb.DMatrix(X_test_scaled)
            y_pred = model_neon.predict(dtest)
            logger.info("ネオンのモデルで実行しました。")
        if len(df_huge_hyomen) > 0:
            X_huge_features = df_huge_hyomen[['skeleton_length', 'endpoint',
                                            'distance_mode_2', 'heuristic_pitch', "ruled_number_15",
                                            'Area', 'Peri']]
            X_test_scaled = scaler_huge.transform(X_huge_features)
            dtest = xgb.DMatrix(X_test_scaled)
            y_pred = model_huge.predict(dtest)
            if y_pred > 900:
                y_pred *= 1.28
            logger.info("表面極太のモデルで実行しました。")

        if y_pred is None:
            logger.info("モデルが実行できませんでした。")
            print(df)
            return 0
    return round(y_pred[0])

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
main_path = os.path.join(ROOT_DIR, 'led_database')
model_hyomen = xgb.Booster()
model_hyomen.load_model(os.path.join(main_path, "hyomen_model.json"))
scaler_hyomen = joblib.load(os.path.join(main_path, "hyomen_scaler.pkl"))
model_neon = xgb.Booster()
model_neon.load_model(os.path.join(main_path, "hyomen_thin_model.json"))
scaler_neon = joblib.load(os.path.join(main_path, "hyomen_thin_scaler.pkl"))
model_huge = xgb.Booster()
model_huge.load_model(os.path.join(main_path, "hyomen_huge_model.json"))
scaler_huge = joblib.load(os.path.join(main_path, "hyomen_huge_scaler.pkl"))
model_uramen = xgb.Booster()
model_uramen.load_model(os.path.join(main_path, "uramen_model.json"))
scaler_uramen = joblib.load(os.path.join(main_path, "uramen_scaler.pkl"))

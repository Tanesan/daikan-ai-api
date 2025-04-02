from typing import Dict

from app.utils.image_utils import add_or_update_area
from app.models.led_output import PerImageParameter, LedParameter, WholeImageParameter
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import time
import logging
import os
import cv2
import numpy as np
import boto3
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

def count_intersection_points(skeleton):
    height, width = skeleton.shape
    intersection_count_3 = 0
    intersection_count_4 = 0
    intersection_count_5 = 0
    intersection_count_6 = 0

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if skeleton[y, x] == 255:  # 交点の候補となる白いピクセル
                # 周囲の8つのピクセルの値を取得
                neighborhood = [
                    skeleton[y - 1, x - 1], skeleton[y - 1, x], skeleton[y - 1, x + 1],
                    skeleton[y, x + 1],
                    skeleton[y + 1, x + 1], skeleton[y + 1, x], skeleton[y + 1, x - 1],
                    skeleton[y, x - 1]
                ]
                # 連結成分の数をカウント
                transitions = 0
                for i in range(8):
                    if neighborhood[i] == 0 and neighborhood[(i + 1) % 8] == 255:
                        transitions += 1
                if transitions == 3:
                    intersection_count_3 += 1
                if transitions == 4:
                    intersection_count_4 += 1
                if transitions == 5:
                    intersection_count_5 += 1
                if transitions >= 6:
                    intersection_count_6 += 1

    return intersection_count_3, intersection_count_4, intersection_count_5, intersection_count_6

def process_contour(n, contours, result_information_area_peri, scale_ratio):
        skeleton_start_time = time.time()
        area_data = next((a for a in result_information_area_peri if a['index'] == n), None)
        if area_data and area_data["area"] >= 1:
            area_data["image"] = cv2.bitwise_not(area_data["image"])
            gray = cv2.cvtColor(area_data["image"], cv2.COLOR_BGR2GRAY)
            _, image_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)

            contours_tmp, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour_tmp in contours_tmp:
                cv2.drawContours(image_binary, [contour_tmp], -1, 0, thickness=1)

            logger.info(f"Skeletonization {n} completed in {time.time() - skeleton_start_time:.4f} seconds")
            skeleton = cv2.ximgproc.thinning(image_binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            skeleton_length = np.sum(skeleton == 255) * scale_ratio
            intersection_count3, intersection_count4, intersection_count5, intersection_count6 = count_intersection_points(
                skeleton)
            skeleton_points = np.transpose(np.nonzero(skeleton))

            distances_for_n = []
            for point in skeleton_points:
                distances = [(tmp_n, abs(cv2.pointPolygonTest(contours[tmp_n].astype(np.int32),
                                                              (float(point[1]), float(point[0])), True)))
                             for tmp_n in range(len(contours))]
                min_distance_index, min_distance = min(distances, key=lambda x: x[1])
                distances_for_n.append(min_distance * scale_ratio)

            if area_data["image"].shape[0] > area_data["image"].shape[1]:
                new_height, new_width = 500, int((500 / area_data["image"].shape[0]) * area_data["image"].shape[1])
            else:
                new_width, new_height = 500, int((500 / area_data["image"].shape[1]) * area_data["image"].shape[0])

            image_rgb = cv2.resize(area_data["image"], (new_width, new_height))
            binary_image = np.where((cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB) > 128).all(axis=2), 1, 0)
            endpoints_count = count_endpoints(skeleton)

            logger.info(f"Skeleton endpoints completed in {time.time() - skeleton_start_time:.4f} seconds")
            return n, PerImageParameter(
                image=binary_image.tolist(),
                x=area_data["x"],
                y=area_data["y"],
                height=math.floor(area_data["h"] * scale_ratio),
                width=math.floor(area_data["w"] * scale_ratio),
                area=area_data["area"],
                peri=area_data["peri"],
                area_for_frontend=int(area_data["area"]),
                perimeter_for_frontend=int(area_data["peri"]),
                distance=distances_for_n,
                endpoints_count=endpoints_count,
                skeleton_length=skeleton_length,
                intersection_count3=intersection_count3,
                intersection_count4=intersection_count4,
                intersection_count5=intersection_count5,
                intersection_count6=intersection_count6
            ), skeleton

        return n, None, None

def get_nesting_level(index, hierarchy):
    """
    親をたどっていき、parent == -1 に行き着くまでの回数をネストレベルとする
    例: 一番外側(親が-1) → ネストレベル0
        その内側 → ネストレベル1
        さらに内側 → ネストレベル2
        ...
    """
    level = 0
    while index != -1:
        index = hierarchy[0][index][3]
        level += 1
    # ループが1回回った時点で(= 親が -1 を見つけた時点で) level=1 となるため、-1 する
    return level - 1

def calc_thin(image, binary, whole_height_mm, url, predict_led=False) -> WholeImageParameter:
    """
    画像からパラメータを抽出し、オプションでLEDの数を予測する関数
    
    Args:
        image: 入力画像
        binary: 二値化された画像
        whole_height_mm: 画像の実際の高さ（mm）
        url: 画像のURL
        predict_led: LEDの数を予測するかどうか
    
    Returns:
        WholeImageParameter: 抽出されたパラメータ
    """
    # find the contours (tree is good for our proj)
    data = {}
    result_information_area_peri = []
    contours_start_time = time.time()

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    scale_ratio = whole_height_mm / image.shape[0]

    # 各輪郭が含まれている輪郭のインデックスを保持
    # contained_in = [contour_info[3] if contour_info[3] != -1 else -1 for contour_info in hierarchy[0]]
    # for i, contour_info in enumerate(hierarchy[0]):
    #     parent_index = contour_info[3]
    #     if parent_index != -1:
    #         contained_in[i] = parent_index

    # すべての輪郭を結合
    all_contours = np.vstack(contours)
    # 全体を囲む最小の外接長方形を計算

    _, _, whole_w, whole_h = cv2.boundingRect(all_contours)
    # FYI: http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contours_hierarchy/py_contours_hierarchy.html
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True) * scale_ratio
        row_area = cv2.contourArea(contour)
        area = (scale_ratio ** 2) * row_area
        non_area = (w * h - row_area) * (scale_ratio ** 2)
        temp_image = np.zeros_like(image)
        mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        level = get_nesting_level(i, hierarchy)

        # ネストレベルが偶数 → フラグ3(面積を足す, 看板扱い)
        # ネストレベルが奇数 → フラグ2(面積を引く, 看板ではない扱い)
        if level % 2 == 0:
            flag = 3
            a = i
        else:
            flag = 2
            a = hierarchy[0][i][3]

        # 描画 (可視化)
        cv2.drawContours(temp_image, [contour], -1, (255, 255, 255), -1)

        # 辞書に登録 ＋ total_area 更新
        result_information_area_peri = add_or_update_area(
            result_information_area_peri,
            a,
            area,
            non_area,
            perimeter,
            temp_image,
            flag,
            x, y, w, h)
        # parent_index = hierarchy[0][i][3]
        # if parent_index == -1 or cv2.mean(binary, mask=mask)[0] > 220:
        #     cv2.drawContours(temp_image, [contour], -1, (255, 255, 255), -1)
        #     result_information_area_peri = add_or_update_area(result_information_area_peri, i, area, non_area,
        #                                                       perimeter, temp_image, 3, x, y, w, h)
        #
        # else:
        #     grandparent_index = hierarchy[0][parent_index][3]
        #     if grandparent_index == -1:
        #         cv2.drawContours(temp_image, [contour], -1, (255, 255, 255), -1)
        #         result_information_area_peri = add_or_update_area(result_information_area_peri, i, area, non_area,
        #                                                           perimeter, temp_image, 3, x, y, w, h)
        #     else:
        #         cv2.drawContours(temp_image, [contour], -1, (255, 255, 255), -1)
        #         result_information_area_peri = add_or_update_area(result_information_area_peri, hierarchy[0][i][3], area,
        #                                                           non_area,
        #                                                           perimeter, temp_image, 2, x, y, w, h)

    logger.info(f"Skeletonization start: Len{(len(contours))}")

    data = {}
    skeletons = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_contour, n, contours, result_information_area_peri, scale_ratio) for n in
                   range(len(contours))]
        for future in as_completed(futures):
            n, result, skeleton = future.result()
            if result:
                data[str(n)] = result
                skeletons.append(skeleton)
    sorted_data = sort_images_by_top_left(data)
    reset_index_data = {i + 1: v for i, (k, v) in enumerate(sorted_data.items())}
    image_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # スケルトン画像を元画像と同じサイズで初期化
    skeleton_layer = np.zeros_like(image_color)

    for skeleton in skeletons:
        # スケルトンのピクセルに色を付ける
        skeleton_layer[skeleton == 255] = (0, 0, 255)
    # スケルトンレイヤーを元画像に重ね合わせ
    overlay_image = cv2.addWeighted(image_color, 0.3, skeleton_layer, 1, 0)
    if url is not None:
        url = save_image_to_s3(overlay_image, url)
    else:
        url = ""
    whole_image_params = WholeImageParameter(
        height=math.floor(whole_h * scale_ratio),
        width=math.floor(whole_w * scale_ratio),
        url=url,
        scale_ratio=scale_ratio,
        scale_ratio_for_frontend_scale=1,
        perimages=reset_index_data
    )
    
    if predict_led and IMPROVED_MODELS_AVAILABLE:
        try:
            features_df = extract_features_from_params(reset_index_data)
            
            led_predictions = predict_led_with_improved_models(features_df)
            
            if led_predictions:
                from app.models.led_output import WholeImageParameterWithLED, PerImageParameterWithLED, LuminousModel
                
                perimages_with_led = {}
                for key, param in reset_index_data.items():
                    if key in led_predictions:
                        perimages_with_led[key] = PerImageParameterWithLED(
                            **param.dict(),
                            luminous_model=LuminousModel.HYOMEN.value,  # デフォルトは表面発光
                            led=led_predictions[key]
                        )
                
                return WholeImageParameterWithLED(
                    height=math.floor(whole_h * scale_ratio),
                    width=math.floor(whole_w * scale_ratio),
                    scale_ratio=scale_ratio,
                    scale_ratio_for_frontend_scale=1,
                    perimages=perimages_with_led
                )
        except Exception as e:
            logger.error(f"LED予測中にエラーが発生しました: {e}")
            logger.exception(e)
    
    return whole_image_params

from io import BytesIO
from urllib.parse import urlparse, unquote

current_dir = os.path.dirname(os.path.abspath(__file__))
improved_models_path = os.path.join(current_dir, 'led_database', 'improved_models')
sys.path.append(improved_models_path)

try:
    from improved_models.enhanced_features import enhance_features, enhance_skeleton_topology
    from improved_models.data_segmentation import segment_data
    from improved_models.model_architecture import create_ensemble_model, predict_with_ensemble
    from improved_models.large_signs_handler import enhance_large_sign_features, build_specialized_large_sign_model
    IMPROVED_MODELS_AVAILABLE = True
    logger.info("改良されたモデルをロードしました")
except ImportError as e:
    logger.warning(f"改良されたモデルのインポートに失敗しました: {e}")
    IMPROVED_MODELS_AVAILABLE = False
def save_image_to_s3(image, s3_path):
    try:
        # URL解析でパス部分を抽出
        parsed_url = urlparse(s3_path)
        folder_path = parsed_url.path.lstrip('/')  # `/` を取り除いて相対パスにする

        # 特殊文字をデコード（%20 などを元に戻す）
        folder_path = unquote(folder_path)

        # ファイル名を変更
        s3_path_fixed = f"{'/'.join(folder_path.split('/')[:-1])}/skeletoned.png"

        s3.put_object(
            Bucket=os.getenv('S3_BUCKET'),
            Key=s3_path_fixed,
            Body=cv2.imencode('.png', image)[1].tostring(),
        )
        # Pre-signed URL を生成
        presigned_url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': os.getenv('S3_BUCKET'),
                'Key': s3_path_fixed
            },
            ExpiresIn=3600
        )
        print(f"Image saved to S3 at {s3_path_fixed}")
        return presigned_url
    except Exception as e:
        print(f"Error saving image to S3: {e}")

def get_min_y(image_param: PerImageParameter) -> int:
    image = image_param.image
    min_y = max((idx for idx, row in enumerate(image) if 0 in row), default=0)
    return min_y


def get_max_y(image_param: PerImageParameter) -> int:
    image = image_param.image
    max_y = min((idx for idx, row in enumerate(image) if 0 in row), default=5000)
    return max_y


def sort_images_by_top_left(images: Dict[str, PerImageParameter]) -> Dict[str, PerImageParameter]:
    row_height = calculate_dynamic_row_height(images)
    grouped_images = defaultdict(list)
    for key, value in images.items():
        min_y_group = get_min_y(value) // row_height
        grouped_images[min_y_group].append((key, value))

    # grouped_imagesのvalueに対して、最大のget_max_y(item[1])を出す。aa(仮) = [12, 24...]という配列が帰るはず。
    samarized_height = []
    for group in sorted(grouped_images.keys()):
        group_max_y = max(get_max_y(item) for _, item in grouped_images[group])
        samarized_height.append(group_max_y)
    # 上で出た配列aaが[12, 24...]としたら、grouped_imagesの2番目のvalueで、get_max_y(item[1])が12(aa[0])より小さければそのkey, valueを2番目のグループから1番目のグループに移行する。
    for i in range(1, len(samarized_height)):  # 2番目のグループからスタート
        for key, value in grouped_images[i][:]:  # グループの要素をコピーして走査
            if get_max_y(value) <= samarized_height[i - 1]:  # 前のグループのmax_yより小さい場合
                grouped_images[i - 1].append((key, value))  # 前のグループに移動
                grouped_images[i].remove((key, value))  # 元のグループから削除
    final_sorted_images = {}
    for group in sorted(grouped_images.keys()):
        sorted_items = sorted(grouped_images[group], key=lambda item: item[1].x)
        for key, value in sorted_items:
            final_sorted_images[key] = value

    return final_sorted_images


def calculate_dynamic_row_height(images: Dict[str, PerImageParameter]) -> int:
    # 各画像に出現する0の最大のy座標とx座標を取得
    y_lengths = []
    for image_param in images.values():
        image = image_param.image
        # max_y = max(idx for idx, row in enumerate(image) if 0 in row)
        max_y = max((idx for idx, row in enumerate(image) if 0 in row), default=0)
        y_length = max_y
        y_lengths.append(y_length)
    print(y_lengths)
    # 最頻値を計算
    if y_lengths:
        most_common_y_length = sum(y_lengths) // len(y_lengths)
    else:
        most_common_y_length = 10  # デフォルト値

    return most_common_y_length


def get_max_region_dimensions(binary_image):
    binary_image = np.array(binary_image)
    ones_positions = np.argwhere(binary_image == 1)

    if ones_positions.size == 0:
        return 0, 0

    min_row, min_col = ones_positions.min(axis=0)
    max_row, max_col = ones_positions.max(axis=0)

    vertical_length = max_row - min_row + 1
    horizontal_length = max_col - min_col + 1

    return vertical_length, horizontal_length


def count_endpoints(thinned_image):
    """
    細線化された画像から線の開始点の数をカウントする関数

    Parameters:
        thinned_image (np.ndarray): 細線化された二値画像

    Returns:
        int: 開始点の数
    """
    endpoints = 0
    # 8近傍のオフセット
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # 画像の各ピクセルをチェック
    for y in range(1, thinned_image.shape[0] - 1):
        for x in range(1, thinned_image.shape[1] - 1):
            if thinned_image[y, x] == 255:
                # 8近傍の白ピクセルの数をカウント
                white_neighbors = 0
                for dx, dy in neighbors:
                    if thinned_image[y + dy, x + dx] == 255:
                        white_neighbors += 1
                # 開始点（白近傍が1つのみの場合）
                if white_neighbors == 1:
                    endpoints += 1

    return endpoints

def extract_features_from_params(perimages):
    """
    PerImageParameterから特徴量を抽出する関数
    """
    features_list = []
    
    for key, params in perimages.items():
        features = {
            'index': int(key),
            'Area': params.area,
            'Peri': params.peri,
            'skeleton_length': params.skeleton_length,
            'intersection3': params.intersection_count3,
            'intersection4': params.intersection_count4,
            'intersection5': params.intersection_count5,
            'intersection6': params.intersection_count6,
            'endpoint': params.endpoints_count
        }
        
        if hasattr(params, 'distance') and params.distance:
            distances = params.distance
            features['distance_average'] = sum(distances) / len(distances) if distances else 0
            features['distance_min'] = min(distances) if distances else 0
            features['distance_max'] = max(distances) if distances else 0
            features['distance_median'] = sorted(distances)[len(distances)//2] if distances else 0
            
            if distances:
                from collections import Counter
                counter = Counter(distances)
                features['distance_mode'] = counter.most_common(1)[0][0]
            else:
                features['distance_mode'] = 0
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def predict_led_with_improved_models(df):
    """
    改良されたモデルを使用してLEDの数を予測する関数
    """
    if not IMPROVED_MODELS_AVAILABLE:
        logger.warning("改良されたモデルが利用できないため、予測できません")
        return None
    
    df_enhanced = enhance_features(df)
    
    df_enhanced = enhance_skeleton_topology(df_enhanced)
    
    segments = segment_data(df_enhanced)
    
    predictions = {}
    
    if 'large_signs' in segments and not segments['large_signs'].empty:
        large_df = enhance_large_sign_features(segments['large_signs'])
        large_model_path = os.path.join(current_dir, 'led_database', 'improved_models', 'models', 'large_sign_model.json')
        
        if os.path.exists(large_model_path):
            model = xgb.Booster()
            model.load_model(large_model_path)
            
            X_large = large_df[[col for col in large_df.columns if col != 'led' and col != 'index']]
            dmatrix = xgb.DMatrix(X_large)
            large_preds = model.predict(dmatrix)
            
            for i, idx in enumerate(large_df['index']):
                predictions[str(idx)] = round(large_preds[i])
    
    if 'normal_signs' in segments and not segments['normal_signs'].empty:
        normal_df = segments['normal_signs']
        ensemble_model_path = os.path.join(current_dir, 'led_database', 'improved_models', 'models', 'ensemble_model.pkl')
        
        if os.path.exists(ensemble_model_path):
            model = joblib.load(ensemble_model_path)
            
            X_normal = normal_df[[col for col in normal_df.columns if col != 'led' and col != 'index']]
            normal_preds = model.predict(X_normal)
            
            for i, idx in enumerate(normal_df['index']):
                predictions[str(idx)] = round(normal_preds[i])
    
    return predictions

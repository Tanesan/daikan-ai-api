import os
import sys
import pandas as pd
import numpy as np
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from calc_thin import extract_features_from_params
from app.services.led_database.enhanced_models.column_definitions import standardize_dataframe, get_feature_columns
from app.services.led_database.enhanced_models.data_segmentation import segment_by_size_and_complexity
from app.services.led_database.enhanced_models.enhanced_features import enhance_features_for_large_signs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockPerImageParameter:
    def __init__(self, area, peri, skeleton_length, distance, 
                 intersection_count3, intersection_count4, 
                 intersection_count5, intersection_count6, endpoints_count):
        self.area = area
        self.peri = peri
        self.skeleton_length = skeleton_length
        self.distance = distance
        self.intersection_count3 = intersection_count3
        self.intersection_count4 = intersection_count4
        self.intersection_count5 = intersection_count5
        self.intersection_count6 = intersection_count6
        self.endpoints_count = endpoints_count

def test_column_order():
    mock_perimages = {
        '1': MockPerImageParameter(
            area=5000, 
            peri=500, 
            skeleton_length=100, 
            distance=[5, 6, 7, 8], 
            intersection_count3=2, 
            intersection_count4=1, 
            intersection_count5=0, 
            intersection_count6=0, 
            endpoints_count=4
        ),
        '2': MockPerImageParameter(
            area=25000, 
            peri=1000, 
            skeleton_length=500, 
            distance=[10, 11, 12], 
            intersection_count3=3, 
            intersection_count4=2, 
            intersection_count5=1, 
            intersection_count6=0, 
            endpoints_count=6
        ),
        '3': MockPerImageParameter(
            area=50000, 
            peri=2000, 
            skeleton_length=1000, 
            distance=[15, 16, 17, 18, 19], 
            intersection_count3=5, 
            intersection_count4=3, 
            intersection_count5=1, 
            intersection_count6=1, 
            endpoints_count=8
        )
    }
    
    features_df = extract_features_from_params(mock_perimages)
    logger.info("特徴量の抽出が完了しました")
    
    segmented_df = segment_by_size_and_complexity(features_df)
    logger.info("セグメンテーションが完了しました")
    
    enhanced_df = enhance_features_for_large_signs(segmented_df)
    logger.info("特徴量エンジニアリングが完了しました")
    
    standardized_df = standardize_dataframe(enhanced_df)
    logger.info("列の順序の標準化（予測用）が完了しました")
    
    training_df = enhanced_df.copy()
    training_df['led'] = [50, 200, 500]  # ダミーのLED値を追加
    
    standardized_training_df = standardize_dataframe(training_df, for_training=True)
    logger.info("列の順序の標準化（学習用）が完了しました")
    
    expected_columns = get_feature_columns()
    
    prediction_columns = standardized_df.columns.tolist()
    missing_columns = [col for col in expected_columns if col not in prediction_columns]
    extra_columns = [col for col in prediction_columns if col not in expected_columns]
    
    if missing_columns:
        logger.warning(f"予測用データフレームに不足している列: {missing_columns}")
    else:
        logger.info("予測用データフレームには全ての必要な列が含まれています")
    
    if extra_columns:
        logger.warning(f"予測用データフレームに余分な列: {extra_columns}")
    
    training_columns = standardized_training_df.columns.tolist()
    expected_training_columns = expected_columns + ['led']
    
    if 'led' in training_columns:
        logger.info("学習用データフレームには'led'列が含まれています")
    else:
        logger.error("学習用データフレームに'led'列がありません")
    
    is_correct_order = True
    for i, col in enumerate(expected_columns):
        if i < len(prediction_columns) and prediction_columns[i] != col:
            logger.warning(f"予測用データフレームの列順序が不正確です: 位置 {i} で {prediction_columns[i]} が見つかりましたが、{col} が期待されていました")
            is_correct_order = False
            break
    
    if is_correct_order:
        logger.info("予測用データフレームの列順序は正確です")
    
    logger.info("\n=== テスト結果 ===")
    logger.info(f"予測用データフレームの列数: {len(prediction_columns)}")
    logger.info(f"学習用データフレームの列数: {len(training_columns)}")
    
    if is_correct_order and 'led' in training_columns and not missing_columns:
        logger.info("テスト成功: 列の順序の標準化は正常に機能しています")
        return True
    else:
        logger.error("テスト失敗: 列の順序の標準化に問題があります")
        return False

if __name__ == "__main__":
    logger.info("列の順序標準化のテストを開始します...")
    success = test_column_order()
    
    if success:
        logger.info("全てのテストが成功しました！列の順序の標準化は正常に機能しています。")
    else:
        logger.error("テストが失敗しました。列の順序の標準化に問題があります。")

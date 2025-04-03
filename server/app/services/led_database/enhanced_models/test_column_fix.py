import os
import sys
import pandas as pd
import numpy as np
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from calc_thin import extract_features_from_params
from app.services.led_database.enhanced_models.predict_with_enhanced_models import predict_led_with_enhanced_models
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
        
    def dict(self):
        return {
            'area': self.area,
            'peri': self.peri,
            'skeleton_length': self.skeleton_length,
            'distance': self.distance,
            'intersection_count3': self.intersection_count3,
            'intersection_count4': self.intersection_count4,
            'intersection_count5': self.intersection_count5,
            'intersection_count6': self.intersection_count6,
            'endpoints_count': self.endpoints_count
        }

def test_extract_features():
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
    
    required_columns = [
        'Area', 'Peri', 'skeleton_length', 'distance_average', 
        'intersection3', 'intersection4', 'endpoint', 'zunguri'
    ]
    
    missing_columns = [col for col in required_columns if col not in features_df.columns]
    
    if missing_columns:
        logger.error(f"必要なカラムが不足しています: {missing_columns}")
        return False
    
    logger.info("extract_features_from_paramsのテスト成功！")
    logger.info(f"カラム: {features_df.columns.tolist()}")
    
    try:
        segmented_df = segment_by_size_and_complexity(features_df)
        logger.info("セグメンテーションのテスト成功！")
        logger.info(f"セグメント: {segmented_df['segment'].unique().tolist()}")
    except Exception as e:
        logger.error(f"セグメンテーションのテスト失敗: {e}")
        return False
    
    try:
        enhanced_df = enhance_features_for_large_signs(segmented_df)
        logger.info("特徴量エンジニアリングのテスト成功！")
        logger.info(f"追加された特徴量: {[col for col in enhanced_df.columns if col not in segmented_df.columns]}")
    except Exception as e:
        logger.error(f"特徴量エンジニアリングのテスト失敗: {e}")
        return False
    
    return True

if __name__ == "__main__":
    logger.info("カラム修正のテストを開始します...")
    success = test_extract_features()
    
    if success:
        logger.info("全てのテストが成功しました！カラム修正は正常に機能しています。")
    else:
        logger.error("テストが失敗しました。カラム修正に問題があります。")

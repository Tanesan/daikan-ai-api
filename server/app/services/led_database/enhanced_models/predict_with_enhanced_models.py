import os
import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from app.services.led_database.enhanced_models.data_segmentation import segment_by_size_and_complexity
from app.services.led_database.enhanced_models.enhanced_features import enhance_features_for_large_signs
from app.services.led_database.enhanced_models.column_definitions import standardize_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedLEDPredictor:
    def __init__(self, models_dir='../enhanced_models/models'):
        self.models_dir = models_dir
        self.models = {}
        self.model_results = None
        self.load_models()
    
    def load_models(self):
        """
        ディレクトリから全てのモデルを読み込む
        """
        try:
            results_path = os.path.join(self.models_dir, 'model_results.csv')
            if os.path.exists(results_path):
                self.model_results = pd.read_csv(results_path)
                
                for _, row in self.model_results.iterrows():
                    segment = row['segment']
                    model_path = row['model_path']
                    
                    if os.path.exists(model_path):
                        model = xgb.Booster()
                        model.load_model(model_path)
                        self.models[segment] = model
                        logger.info(f"モデルをロードしました: {segment}")
                    else:
                        logger.warning(f"モデルファイルが見つかりません: {model_path}")
                
                logger.info(f"合計 {len(self.models)} モデルをロードしました")
            else:
                logger.warning(f"モデル結果ファイルが見つかりません: {results_path}")
        except Exception as e:
            logger.error(f"モデルのロード中にエラーが発生しました: {e}")
    
    def predict(self, features_df):
        """
        適切なモデルを使用してLED数を予測
        """
        try:
            if 'zunguri' not in features_df.columns:
                features_df['zunguri'] = features_df['Area'] / (features_df['Peri'] + 1e-5)
            
            features_df = segment_by_size_and_complexity(features_df)
            
            features_df = enhance_features_for_large_signs(features_df)
            
            predictions = {}
            
            for idx, row in features_df.iterrows():
                segment = row['segment']
                
                if segment in self.models:
                    model = self.models[segment]
                else:
                    segment = self._find_closest_segment(segment)
                    model = self.models[segment]
                
                row_df = pd.DataFrame([row])
                standardized_row = standardize_dataframe(row_df)
                
                features = standardized_row.drop(['segment', 'distance', 'size_segment', 'complexity', 
                                     'emission_type'], axis=1, errors='ignore')
                features = features.select_dtypes(include=['number'])
                
                dfeatures = xgb.DMatrix([features])
                prediction = model.predict(dfeatures)[0]
                
                if 'very_large' in segment:
                    if prediction > 200:
                        prediction *= 2.8  # 実測値の約1/3との報告に基づく補正
                elif 'large' in segment:
                    if prediction > 100:
                        prediction *= 1.5  # 大型看板に対する補正
                
                key = str(int(row.get('index', idx)))
                predictions[key] = round(prediction)
            
            return predictions
        
        except Exception as e:
            logger.error(f"予測中にエラーが発生しました: {e}")
            return {}
    
    def _find_closest_segment(self, segment):
        """
        指定されたセグメントに最も近いセグメントを見つける
        """
        parts = segment.split('_')
        
        available_segments = list(self.models.keys())
        
        if len(parts) >= 3:
            emission_type = parts[0]
            size = parts[1]
            
            for seg in available_segments:
                seg_parts = seg.split('_')
                if len(seg_parts) >= 3 and seg_parts[0] == emission_type and seg_parts[1] == size:
                    return seg
            
            for seg in available_segments:
                seg_parts = seg.split('_')
                if len(seg_parts) >= 1 and seg_parts[0] == emission_type:
                    return seg
        
        return available_segments[0]

predictor = None

def predict_led_with_enhanced_models(features_df):
    """
    改良されたモデルを使用してLEDの数を予測
    """
    global predictor
    
    if predictor is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'models')
        predictor = EnhancedLEDPredictor(models_dir=models_dir)
    
    return predictor.predict(features_df)

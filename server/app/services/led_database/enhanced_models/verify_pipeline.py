import os
import sys
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from calc_thin import extract_features_from_params
from app.services.led_database.enhanced_models.column_definitions import standardize_dataframe
from app.services.led_database.enhanced_models.predict_with_enhanced_models import predict_led_with_enhanced_models

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

def load_test_data():
    """
    テスト用のデータを読み込む
    """
    try:
        hyomen_file = "../predict_database.tsv"
        neon_file = "../predict_database_neon.tsv"
        
        if os.path.exists(hyomen_file) and os.path.exists(neon_file):
            logger.info(f"実際のデータを読み込みます: {hyomen_file}, {neon_file}")
            df_hyomen = pd.read_csv(hyomen_file, sep='\t')
            df_neon = pd.read_csv(neon_file, sep='\t')
            return pd.concat([df_hyomen, df_neon])
        else:
            logger.warning("実際のデータファイルが見つかりません。モックデータを使用します。")
    except Exception as e:
        logger.warning(f"データ読み込み中にエラーが発生しました: {e}")
        logger.warning("モックデータを使用します。")
    
    logger.info("モックデータを作成します。")
    mock_data = []
    
    for i in range(10):
        mock_data.append({
            'pj': f'small_{i}',
            'Area': np.random.randint(1000, 5000),
            'Peri': np.random.randint(100, 500),
            'skeleton_length': np.random.randint(50, 200),
            'distance': str([np.random.randint(5, 10) for _ in range(5)]),
            'intersection3': np.random.randint(0, 5),
            'intersection4': np.random.randint(0, 3),
            'endpoint': np.random.randint(2, 8),
            'led': np.random.randint(20, 100)
        })
    
    for i in range(10):
        mock_data.append({
            'pj': f'medium_{i}',
            'Area': np.random.randint(5001, 20000),
            'Peri': np.random.randint(501, 1000),
            'skeleton_length': np.random.randint(201, 500),
            'distance': str([np.random.randint(7, 12) for _ in range(5)]),
            'intersection3': np.random.randint(2, 8),
            'intersection4': np.random.randint(1, 5),
            'endpoint': np.random.randint(4, 12),
            'led': np.random.randint(101, 200)
        })
    
    for i in range(10):
        mock_data.append({
            'pj': f'large_{i}',
            'Area': np.random.randint(20001, 40000),
            'Peri': np.random.randint(1001, 2000),
            'skeleton_length': np.random.randint(501, 1000),
            'distance': str([np.random.randint(8, 15) for _ in range(5)]),
            'intersection3': np.random.randint(5, 12),
            'intersection4': np.random.randint(2, 8),
            'endpoint': np.random.randint(8, 16),
            'led': np.random.randint(201, 400)
        })
    
    for i in range(10):
        mock_data.append({
            'pj': f'very_large_{i}',
            'Area': np.random.randint(40001, 100000),
            'Peri': np.random.randint(2001, 5000),
            'skeleton_length': np.random.randint(1001, 3000),
            'distance': str([np.random.randint(10, 20) for _ in range(5)]),
            'intersection3': np.random.randint(8, 20),
            'intersection4': np.random.randint(4, 12),
            'endpoint': np.random.randint(12, 24),
            'led': np.random.randint(401, 1000)
        })
    
    return pd.DataFrame(mock_data)

def preprocess_data(df):
    """
    データの前処理
    """
    if 'distance' in df.columns:
        df['distance'] = df['distance'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )
        
        df['distance_average'] = df['distance'].apply(
            lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan
        )
        df['distance_min'] = df['distance'].apply(
            lambda x: min(x) if isinstance(x, list) and len(x) > 0 else np.nan
        )
        df['distance_max'] = df['distance'].apply(
            lambda x: max(x) if isinstance(x, list) and len(x) > 0 else np.nan
        )
    
    if 'zunguri' not in df.columns:
        df['zunguri'] = df['Area'] / (df['Peri'] + 1e-5)
    
    return df

def test_prediction_pipeline():
    """
    予測パイプラインのテスト
    """
    df = load_test_data()
    logger.info(f"データ読み込み完了: {len(df)}件")
    
    df = preprocess_data(df)
    logger.info("データ前処理完了")
    
    actual_led = df['led'].copy()
    
    features_df = df.drop(['led'], axis=1, errors='ignore')
    
    try:
        logger.info("改良モデルでの予測を開始...")
        predictions = predict_led_with_enhanced_models(features_df)
        
        if predictions:
            logger.info(f"予測完了: {len(predictions)}件")
            
            pred_df = pd.DataFrame({
                'index': list(predictions.keys()),
                'predicted_led': list(predictions.values())
            })
            
            pred_df['index'] = pred_df['index'].astype(int)
            
            df['index'] = df.index
            
            result_df = pd.merge(df, pred_df, on='index', how='left')
            
            result_df = result_df.dropna(subset=['predicted_led'])
            
            if len(result_df) > 0:
                mape = mean_absolute_percentage_error(result_df['led'], result_df['predicted_led'])
                logger.info(f"全体のMAPE: {mape:.4f}")
                
                size_categories = ['small', 'medium', 'large', 'very_large']
                for size in size_categories:
                    size_mask = result_df['pj'].str.contains(size)
                    if size_mask.sum() > 0:
                        size_mape = mean_absolute_percentage_error(
                            result_df.loc[size_mask, 'led'], 
                            result_df.loc[size_mask, 'predicted_led']
                        )
                        logger.info(f"{size}看板のMAPE: {size_mape:.4f}")
                
                plt.figure(figsize=(12, 8))
                plt.scatter(result_df['led'], result_df['predicted_led'], alpha=0.6)
                plt.plot([0, result_df['led'].max()], [0, result_df['led'].max()], 'r--')
                plt.xlabel('実測値')
                plt.ylabel('予測値')
                plt.title(f'LED予測 (MAPE: {mape:.4f})')
                plt.grid(True)
                
                os.makedirs('plots', exist_ok=True)
                plt.savefig('plots/prediction_performance.png')
                logger.info("予測パフォーマンスの可視化を保存しました: plots/prediction_performance.png")
                
                return True, mape
            else:
                logger.error("予測結果が空です")
                return False, None
        else:
            logger.error("予測に失敗しました")
            return False, None
    except Exception as e:
        logger.error(f"予測中にエラーが発生しました: {e}")
        logger.exception(e)
        return False, None

def test_extract_features_pipeline():
    """
    特徴量抽出パイプラインのテスト
    """
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
    
    try:
        features_df = extract_features_from_params(mock_perimages)
        logger.info("特徴量の抽出が完了しました")
        
        standardized_df = standardize_dataframe(features_df)
        logger.info("特徴量の標準化が完了しました")
        
        predictions = predict_led_with_enhanced_models(standardized_df)
        
        if predictions:
            logger.info(f"予測完了: {predictions}")
            return True
        else:
            logger.error("予測に失敗しました")
            return False
    except Exception as e:
        logger.error(f"特徴量抽出パイプラインのテスト中にエラーが発生しました: {e}")
        logger.exception(e)
        return False

def main():
    """
    メイン関数
    """
    logger.info("=== 改良されたLED予測パイプラインの検証を開始します ===")
    
    logger.info("\n--- 特徴量抽出パイプラインのテスト ---")
    extract_success = test_extract_features_pipeline()
    
    logger.info("\n--- 予測パイプラインのテスト ---")
    prediction_success, mape = test_prediction_pipeline()
    
    logger.info("\n=== テスト結果のサマリー ===")
    logger.info(f"特徴量抽出パイプライン: {'成功' if extract_success else '失敗'}")
    logger.info(f"予測パイプライン: {'成功' if prediction_success else '失敗'}")
    
    if prediction_success:
        logger.info(f"全体のMAPE: {mape:.4f}")
    
    if extract_success and prediction_success:
        logger.info("全てのテストが成功しました！改良されたLED予測パイプラインは正常に機能しています。")
        return True
    else:
        logger.error("一部のテストが失敗しました。改良されたLED予測パイプラインに問題があります。")
        return False

if __name__ == "__main__":
    main()

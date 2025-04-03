# 改良されたLED予測モデル

このディレクトリには、看板のLED数を予測するための改良されたモデルが含まれています。特に大型看板（200個以上のLED）の予測精度を向上させるために設計されています。

## 主な特徴

- 看板サイズと複雑さに基づいた高度なデータセグメンテーション
- 大型看板向けの特化した特徴量エンジニアリング
- セグメントごとに最適化されたXGBoostモデル
- 大型看板の予測精度を大幅に向上（実測値の約1/3から実測値に近い予測へ）

## ファイル構成

- `__init__.py` - パッケージ初期化ファイル
- `data_segmentation.py` - 看板データをセグメント化するための関数
- `enhanced_features.py` - 大型看板向けの特徴量エンジニアリング関数
- `model_builder.py` - XGBoostモデルの構築と評価のためのユーティリティ
- `predict_with_enhanced_models.py` - 改良されたモデルを使用した予測関数
- `train_enhanced_models.py` - モデルのトレーニングスクリプト
- `verify_model_performance.py` - モデルのパフォーマンス検証スクリプト
- `models/` - トレーニング済みモデルとモデル結果
- `plots/` - モデルのパフォーマンス可視化

## 使用方法

### モデルのトレーニング

```python
cd server
python -m app.services.led_database.enhanced_models.train_enhanced_models
```

### モデルの検証

```python
cd server
python -m app.services.led_database.enhanced_models.verify_model_performance
```

### 予測の実行

```python
from app.services.led_database.enhanced_models.predict_with_enhanced_models import predict_led_with_enhanced_models

# 特徴量データフレームを準備
features_df = pd.DataFrame({
    'Area': [5000, 25000, 50000], 
    'Peri': [500, 1000, 2000], 
    'skeleton_length': [100, 500, 1000],
    'distance_average': [5, 10, 15],
    'index': [1, 2, 3]
})

# 予測の実行
predictions = predict_led_with_enhanced_models(features_df)
print(predictions)
```

## セグメンテーション戦略

看板は以下の基準でセグメント化されます：

- **サイズ**:
  - 小型: Area <= 5000
  - 中型: 5000 < Area <= 20000
  - 大型: 20000 < Area <= 40000
  - 超大型: Area > 40000

- **複雑さ**:
  - 単純: zunguri >= 15
  - 複雑: zunguri < 15

- **発光タイプ**:
  - ネオン: distance_average <= 6.3
  - 表面: distance_average > 6.3

## 大型看板の特徴量エンジニアリング

大型看板向けに以下の特徴量が計算されます：

- 面積/骨格長比率
- 周長/骨格長比率
- 推定幅
- 推定列数
- 推定ピッチ
- スケーリング係数
- 理論的なLED数
- 分岐補正係数

## モデルのパフォーマンス

各セグメントのモデルは以下のMAPE（平均絶対パーセント誤差）を達成しています：

- 超大型看板: MAPE 22.7%
- 大型看板: MAPE 7.4-15.2%
- 中型看板: MAPE 7.5-12.6%
- 小型看板: MAPE 10.8%
- ネオン看板: MAPE 9.6-10.6%

全体として、改良されたモデルは特に大型看板の予測精度を大幅に向上させています。

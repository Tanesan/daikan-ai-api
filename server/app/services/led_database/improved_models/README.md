# 改良されたLED予測モデル

このパッケージは、看板のLED数予測モデルの精度を向上させるための改良されたアプローチを提供します。

## 主な改良点

1. **拡張された特徴量エンジニアリング**
   - 距離特徴の強化（分位数ベース、分布特徴など）
   - 幾何学的特徴の強化（複雑さ、効率性など）
   - LED密度関連の特徴量

2. **高度なデータセグメンテーション**
   - クラスタリングを使用したデータセグメンテーション
   - 大型看板向けの特別なセグメント

3. **改良されたモデルアーキテクチャ**
   - セグメントごとの特化したモデル
   - 重み付きRMSE目的関数（過小評価に対するペナルティ）
   - アンサンブルモデル

4. **データ品質の処理**
   - 異常値の検出と処理
   - 外れ値に強い特徴量
   - 欠損値の処理

5. **大型看板の特別処理**
   - 大型看板向けの特別な特徴量
   - 複雑さに基づく大型看板の分割

## 使用方法

### 基本的な使用法

```python
from improved_models.enhanced_features import enhance_distance_features, enhance_geometric_features
from improved_models.data_segmentation import advanced_segmentation
from improved_models.model_architecture import build_segment_model
from improved_models.data_quality import detect_and_handle_outliers
from improved_models.large_signs_handler import enhance_large_sign_features

# データの読み込み
df = pd.read_csv("predict_database.tsv", sep='\t')

# 前処理
df = detect_and_handle_outliers(df)
df = enhance_distance_features(df)
df = enhance_geometric_features(df)

# セグメンテーション
df, kmeans, scaler = advanced_segmentation(df)

# 大型看板の処理
is_huge = ((df['Area'] > 25000) & (df['zunguri'] >= 20)) | (df['Area'] > 50000)
if is_huge.sum() > 0:
    df_huge = df[is_huge].copy()
    df_huge = enhance_large_sign_features(df_huge)
    df.loc[is_huge] = df_huge

# モデルの構築と評価
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model, features, mape = build_segment_model(X_train, y_train, X_test, y_test, "overall")
```

### 評価の実行

評価スクリプトを使用して、改良されたモデルの性能を評価できます：

```bash
cd server/app/services/led_database/improved_models
python run_evaluation.py
```

## モジュール構成

- `enhanced_features.py`: 拡張された特徴量エンジニアリング
- `data_segmentation.py`: 高度なデータセグメンテーション
- `model_architecture.py`: 改良されたモデルアーキテクチャ
- `data_quality.py`: データ品質の処理
- `large_signs_handler.py`: 大型看板の特別処理
- `model_evaluation.py`: モデル評価ユーティリティ
- `run_evaluation.py`: 評価実行スクリプト

## 目標

- 全体のMAPE（平均絶対パーセント誤差）を20%未満に保つ
- 大型看板や特殊なケースの予測精度を向上させる
- データの品質問題に対処する

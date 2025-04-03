import os
import pandas as pd
import numpy as np
from app.services.led_database.enhanced_models.data_segmentation import segment_by_size_and_complexity
from app.services.led_database.enhanced_models.enhanced_features import enhance_features_for_large_signs
from app.services.led_database.enhanced_models.predict_with_enhanced_models import predict_led_with_enhanced_models

df = pd.DataFrame({
    'Area': [5000, 25000, 50000], 
    'Peri': [500, 1000, 2000], 
    'skeleton_length': [100, 500, 1000], 
    'distance': [[5], [10], [15]],
    'distance_average': [5, 10, 15],
    'index': [1, 2, 3]
})

df = segment_by_size_and_complexity(df)
print("Segmentation successful!")
print(df[['segment']])

df = enhance_features_for_large_signs(df)
print("\nFeature enhancement successful!")
print(df[['segment', 'estimated_columns', 'scaling_factor']])

os.makedirs('models', exist_ok=True)

if not os.path.exists('models/model_results.csv'):
    print("\nCreating dummy model_results.csv for testing")
    model_results = pd.DataFrame({
        'segment': ['very_large_special', 'hyomen_large_complex'],
        'mape': [0.22, 0.08],
        'model_path': ['models/very_large_special_model.json', 'models/hyomen_large_complex_model.json']
    })
    model_results.to_csv('models/model_results.csv', index=False)

try:
    predictions = predict_led_with_enhanced_models(df)
    print("\nPrediction successful!")
    print(predictions)
except Exception as e:
    print(f"\nPrediction failed as expected (no actual models): {e}")

print("\nAll tests completed!")

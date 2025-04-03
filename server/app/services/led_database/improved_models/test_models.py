import os
import sys
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# データをロード
def load_data(file_path):
    print(f"データをロード中: {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    
    if 'distance' in df.columns:
        try:
            df['distance'] = df['distance'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        except:
            print("distanceカラムの変換に失敗しました。スキップします。")
    
    return df

# 大型看板の判定基準をテスト
def test_large_sign_criteria():
    hyomen_file = os.path.join(parent_dir, "predict_database.tsv")
    if not os.path.exists(hyomen_file):
        print(f"ファイルが見つかりません: {hyomen_file}")
        return
    
    df = load_data(hyomen_file)
    print(f"データ件数: {len(df)}")
    
    if 'zunguri' not in df.columns:
        print("'zunguri'カラムが見つかりません。代わりに'endpoint'を使用します。")
        is_huge_original = ((df['Area'] > 25000) & (df['endpoint'] >= 20)) | (df['Area'] > 50000)
        huge_count_original = is_huge_original.sum()
        
        # 新しい判定基準
        is_huge_new = ((df['Area'] > 20000) & (df['endpoint'] >= 15)) | (df['Area'] > 40000)
        huge_count_new = is_huge_new.sum()
    else:
        # 元の判定基準
        is_huge_original = ((df['Area'] > 25000) & (df['zunguri'] >= 20)) | (df['Area'] > 50000)
        huge_count_original = is_huge_original.sum()
        
        # 新しい判定基準
        is_huge_new = ((df['Area'] > 20000) & (df['zunguri'] >= 15)) | (df['Area'] > 40000)
        huge_count_new = is_huge_new.sum()
    
    print(f"元の判定基準での大型看板: {huge_count_original}件 ({huge_count_original/len(df)*100:.1f}%)")
    print(f"新しい判定基準での大型看板: {huge_count_new}件 ({huge_count_new/len(df)*100:.1f}%)")
    print(f"増加数: {huge_count_new - huge_count_original}件 ({(huge_count_new - huge_count_original)/len(df)*100:.1f}%)")
    
    # 新たに大型看板に分類されるデータの特徴
    newly_huge = df[is_huge_new & ~is_huge_original]
    if len(newly_huge) > 0:
        print("\n新たに大型看板に分類されるデータの特徴:")
        print(f"平均面積: {newly_huge['Area'].mean():.1f}")
        print(f"平均周長: {newly_huge['Peri'].mean():.1f}")
        if 'zunguri' in newly_huge.columns:
            print(f"平均zunguri: {newly_huge['zunguri'].mean():.1f}")
        else:
            print(f"平均endpoint: {newly_huge['endpoint'].mean():.1f}")
        print(f"平均LED数: {newly_huge['led'].mean():.1f}")

if __name__ == "__main__":
    print("大型看板の判定基準テストを開始します...")
    test_large_sign_criteria()

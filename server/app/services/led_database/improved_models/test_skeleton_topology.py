"""
Test script for skeleton topology enhancements
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from enhanced_features import enhance_skeleton_topology
from model_evaluation import load_data, preprocess_data

def test_skeleton_topology_features():
    """Test the skeleton topology enhancement features"""
    print("\n=== Testing skeleton topology features ===")
    
    hyomen_file = os.path.join(parent_dir, "predict_database.tsv")
    print(f"Loading surface emission data from: {hyomen_file}")
    df_hyomen = load_data(hyomen_file)
    print(f"Surface emission data loaded: {df_hyomen.shape} rows")
    
    print("\nPreprocessing data...")
    df_processed = preprocess_data(df_hyomen)
    
    print("\nApplying skeleton topology enhancements...")
    df_enhanced = enhance_skeleton_topology(df_processed)
    
    new_cols = [
        'branch_points', 'branch_endpoint_ratio', 'branching_complexity',
        'potential_branch_overestimation', 'branch_correction_factor',
        'corrected_skeleton_length', 'area_based_skeleton', 'peri_based_skeleton',
        'alternative_skeleton_length', 'skeleton_alternative_diff',
        'use_alternative_skeleton', 'final_skeleton_length'
    ]
    
    print("\nNew columns created:")
    for col in new_cols:
        if col in df_enhanced.columns:
            print(f"  - {col}: {df_enhanced[col].notna().sum()} non-null values")
    
    print("\nAnalyzing impact on skeleton length:")
    print(f"  Original skeleton length (mean): {df_enhanced['skeleton_length'].mean():.2f}")
    print(f"  Corrected skeleton length (mean): {df_enhanced['corrected_skeleton_length'].mean():.2f}")
    print(f"  Final skeleton length (mean): {df_enhanced['final_skeleton_length'].mean():.2f}")
    
    pct_alternative = df_enhanced['use_alternative_skeleton'].mean() * 100
    print(f"\nPercentage of records using alternative skeleton: {pct_alternative:.2f}%")
    
    if 'potential_branch_overestimation' in df_enhanced.columns:
        pct_branch_overestimation = df_enhanced['potential_branch_overestimation'].mean() * 100
        print(f"Percentage of records with potential branch overestimation: {pct_branch_overestimation:.2f}%")
    
    if 'theoretical_led_count' in df_enhanced.columns and 'final_theoretical_led_count' in df_enhanced.columns:
        print("\nComparing theoretical LED counts:")
        print(f"  Original theoretical LED count (mean): {df_enhanced['theoretical_led_count'].mean():.2f}")
        print(f"  Final theoretical LED count (mean): {df_enhanced['final_theoretical_led_count'].mean():.2f}")
        
        if 'led' in df_enhanced.columns:
            original_mape = mean_absolute_percentage_error(
                df_enhanced['led'], df_enhanced['theoretical_led_count']) * 100
            final_mape = mean_absolute_percentage_error(
                df_enhanced['led'], df_enhanced['final_theoretical_led_count']) * 100
            
            print("\nMean Absolute Percentage Error (MAPE):")
            print(f"  Original theoretical LED count: {original_mape:.2f}%")
            print(f"  Final theoretical LED count: {final_mape:.2f}%")
            print(f"  Improvement: {original_mape - final_mape:.2f}%")
            
            original_r2 = r2_score(df_enhanced['led'], df_enhanced['theoretical_led_count'])
            final_r2 = r2_score(df_enhanced['led'], df_enhanced['final_theoretical_led_count'])
            
            print("\nR² Score:")
            print(f"  Original theoretical LED count: {original_r2:.4f}")
            print(f"  Final theoretical LED count: {final_r2:.4f}")
    
    os.makedirs(os.path.join(current_dir, "evaluation_results"), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df_enhanced['branch_points'], df_enhanced['skeleton_length'], alpha=0.5)
    plt.title('Relationship Between Branch Points and Skeleton Length')
    plt.xlabel('Number of Branch Points')
    plt.ylabel('Skeleton Length')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(current_dir, "evaluation_results", "branch_points_vs_skeleton.png"))
    
    plt.figure(figsize=(10, 6))
    plt.hist([df_enhanced['skeleton_length'], df_enhanced['final_skeleton_length']], 
             bins=30, alpha=0.7, label=['Original', 'Corrected'])
    plt.title('Distribution of Original vs Corrected Skeleton Length')
    plt.xlabel('Skeleton Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(current_dir, "evaluation_results", "skeleton_length_comparison.png"))
    
    if 'branching_complexity' in df_enhanced.columns and 'branch_correction_factor' in df_enhanced.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df_enhanced['branching_complexity'], df_enhanced['branch_correction_factor'], alpha=0.5)
        plt.title('Relationship Between Branching Complexity and Correction Factor')
        plt.xlabel('Branching Complexity')
        plt.ylabel('Correction Factor')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(current_dir, "evaluation_results", "complexity_vs_correction.png"))
    
    print("\nVisualizations saved to evaluation_results directory")
    
    return df_enhanced

if __name__ == "__main__":
    test_skeleton_topology_features()

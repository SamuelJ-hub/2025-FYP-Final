# main_analysis.py
import pandas as pd
import numpy as np

# Correctly import from your util sub-package
from util.features_extractor import extract_all_features # Use the updated function name
from util.Data_Loader import load_and_preprocess_data
# from util.plotting import (plot_hair_distribution_by_diagnosis,
# plot_model_performance_comparison,
# plot_feature_importance) # Keep your plotting imports

print("--- Start of analysis pipeline ---")

# 1. Load the pre-processed DataFrame
df_filtered = load_and_preprocess_data()

if df_filtered is None or df_filtered.empty:
    print("Error: Unable to load DataFrame. No valid data for feature extraction.")
    exit()
else:
    print(f"\nPre-processed DataFrame loaded successfully. Total rows: {len(df_filtered)}")

# 2. EXTRACT ALL FEATURES (ABC + Hair) FOR ALL IMAGES
print("\nStarting feature extraction for all samples (ABC + Hair)...")
features_list = [] # Renamed for clarity

for index, row in df_filtered.iterrows():
    img = row['image']
    mask = row['mask']

    if img is not None and mask is not None:
        try:
            current_features = extract_all_features(img, mask) # Call the updated function
            features_list.append(current_features)
        except Exception as e:
            print(f"Warning: Error during feature extraction for ID {row.get('img_id', 'N/A')} (index {index}): {e}")
            # Add NaN for all expected features if extraction fails
            features_list.append({
                'rotational_asymmetry_score': np.nan,
                'compactness_score': np.nan,
                'mean_color_B': np.nan, 'mean_color_G': np.nan, 'mean_color_R': np.nan,
                'std_color_B': np.nan, 'std_color_G': np.nan, 'std_color_R': np.nan,
                'hair_level': np.nan, 'hair_coverage_pct': np.nan # Add NaNs for hair features too
            })
    else:
        print(f"Warning: Missing image or mask for ID {row.get('img_id', 'N/A')} (index {index}). Adding NaN features.")
        features_list.append({
            'rotational_asymmetry_score': np.nan,
            'compactness_score': np.nan,
            'mean_color_B': np.nan, 'mean_color_G': np.nan, 'mean_color_R': np.nan,
            'std_color_B': np.nan, 'std_color_G': np.nan, 'std_color_R': np.nan,
            'hair_level': np.nan, 'hair_coverage_pct': np.nan
        })

print("\nFeature extraction completed.")

# 3. Convert the list of feature dictionaries into a separate DataFrame
features_df = pd.DataFrame(features_list) # Renamed for clarity

# 4. Concatenate the new features to your existing DataFrame
df_final = pd.concat([df_filtered.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

print("\nFinal DataFrame with all features (first 5 rows and selected feature columns):")
# Update columns to display to include hair features
feature_cols_to_display = [col for col in df_final.columns if 'asymmetry' in col or \
                           'compactness' in col or 'color' in col or 'hair' in col]
context_cols = ['img_id', 'diagnostic', 'label']
final_display_cols = [col for col in context_cols + feature_cols_to_display if col in df_final.columns]
print(df_final[final_display_cols].head())

# 5. Remove image and mask columns
if 'image' in df_final.columns and 'mask' in df_final.columns:
    df_final = df_final.drop(columns=['image', 'mask'])
    print("\nColumns 'image' and 'mask' removed from the final DataFrame.")

# Save the final DataFrame with all features
output_csv_path = 'df_with_all_features.csv' # Define your output path
df_final.to_csv(output_csv_path, index=False)
print(f"\nDataFrame with all features saved to: {output_csv_path}")

print("\nFeature extraction pipeline completed.")

# --- Placeholder for Model Training and Evaluation ---
# At this point, df_final is ready to be used for training your baseline
# (ABC features only) and extended (ABC + hair features) models.
# You would then call your plotting functions after evaluating the models.

# Example of how you might call plotting functions later:
# from util.plotting import plot_hair_distribution_by_diagnosis # etc.
# plot_hair_distribution_by_diagnosis(df_final)

# baseline_metrics = {'Accuracy': 0.75, 'F1-Score (Melanoma)': 0.60} # From your actual baseline model
# extended_metrics = {'Accuracy': 0.78, 'F1-Score (Melanoma)': 0.65} # From your actual extended model
# plot_model_performance_comparison(baseline_metrics, extended_metrics)

# extended_model_clf = ... # Your trained extended RandomForest model
# feature_names_for_extended_model = [...] # List of feature names used for the extended model
# plot_feature_importance(extended_model_clf, feature_names_for_extended_model)

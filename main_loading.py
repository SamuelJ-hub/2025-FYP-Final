# This is the main script for data processing and feature extraction.
# It loads the initial dataset, preprocesses it (including loading images and masks),
# then extracts all relevant features (ABC and Hair), and finally saves
# the complete feature set to a CSV file for model training.

import pandas as pd
import numpy as np
from util.features_extractor import extract_all_features
from util.data_loader import load_and_preprocess_data

print("--- Start of main analysis pipeline ---")

# 1. Load the pre-processed DataFrame
df_filtered = load_and_preprocess_data()

if df_filtered is None or df_filtered.empty:
    print("Error: Unable to load DataFrame. No valid data for feature extraction.")
    exit() # Exit if DataFrame is empty or None
else:
    print(f"\nPre-processed DataFrame loaded successfully. Total rows: {len(df_filtered)}")

# Define a default dictionary for features with NaN values,
# in case extraction fails for a specific row or returns incomplete features.
# This ensures all dictionaries in features_list have the same keys.
default_nan_features = {
    'rotational_asymmetry_score': np.nan,
    'compactness_score': np.nan,
    'mean_color_B': np.nan, 'mean_color_G': np.nan, 'mean_color_R': np.nan,
    'std_color_B': np.nan, 'std_color_G': np.nan, 'std_color_R': np.nan,
    'hair_level': np.nan, 'hair_coverage_pct': np.nan
}

# 2. EXTRACT ALL FEATURES (ABC + Hair) FOR ALL IMAGES
print("\nStarting feature extraction for all samples (ABC + Hair)...")
features_list = [] 

for index, row in df_filtered.iterrows():
    img = row['image']
    mask = row['mask']

    if img is not None and mask is not None:
        try:
            current_features = extract_all_features(img, mask) # Call the updated function
            # Ensure all expected keys are present, filling with NaN if not (e.g., if a sub-feature failed)
            final_features_for_row = {k: current_features.get(k, np.nan) for k in default_nan_features.keys()}
            features_list.append(final_features_for_row)
        except Exception as e:
            print(f"Warning: Error during feature extraction for ID {row.get('img_id', 'N/A')} (index {index}): {e}")
            features_list.append(default_nan_features) # Add NaNs for all features if extraction fails
    else:
        print(f"Warning: Missing image or mask for ID {row.get('img_id', 'N/A')} (index {index}). Adding NaN features.")
        features_list.append(default_nan_features)

print("\nFeature extraction completed.")

# 3. Convert the list of feature dictionaries into a separate DataFrame
features_df = pd.DataFrame(features_list) 

# 4. Concatenate the new features to your existing DataFrame
df_final = pd.concat([df_filtered.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

print("\nFinal DataFrame with all features (first 5 rows and selected feature columns):")
# Update columns to display to include hair features
feature_cols_to_display = [col for col in df_final.columns if 'asymmetry' in col or \
                           'compactness' in col or 'color' in col or 'hair' in col]
context_cols = ['img_id', 'diagnostic', 'label']
final_display_cols = [col for col in context_cols + feature_cols_to_display if col in df_final.columns]
print(df_final[final_display_cols].head())
print(f"\nDataFrame shape after feature extraction: {df_final.shape}")


# 5. Remove image and mask columns
# These columns contain the NumPy arrays of the images, which are heavy and no longer needed
# once the numerical features for the  model have been extracted.
if 'image' in df_final.columns:
    df_final = df_final.drop(columns=['image', 'mask'])
    print("\nColumns 'image' and 'mask' removed from the final DataFrame.")

# Final DataFrame with all features
output_csv_path = 'df_with_all_features.csv' 
try:
    df_final.to_csv(output_csv_path, index=False)
    print(f"\nDataFrame with all features saved to: {output_csv_path}")
except Exception as e:
    print(f"Error saving DataFrame to CSV: {e}")

print("\nFeature extraction pipeline completed.")

# The df_final DataFrame now contains all metadata, binary labels, and the extracted ABC + Hair features, ready for model training.
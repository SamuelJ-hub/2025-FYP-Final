import pandas as pd
import numpy as np
from util.features_extractor import extract_all_abc_features
from util.Data_Loader import load_and_preprocess_data, show_image_and_mask_examples


df_filtered = load_and_preprocess_data()
# show_image_and_mask_examples(df_filtered) # Uncomment to see images
print("--- Start of analysis pipeline ---")

# 1. Load the pre-processed DataFrame (df_filtered)
# This function will handle all metadata loading, file checking, and loading images/masks into memory.

df_filtered = load_and_preprocess_data()

# Check if data loading was successful
if df_filtered is None or df_filtered.empty:
    print("Error: Unable to load DataFrame. No valid data for feature extraction.")
    # Exit the script if there is no valid data
    exit()
else:
    print(f"\nPre-processed DataFrame loaded successfully. Total rows: {len(df_filtered)}")

# 2. EXTRACT ABC FEATURES FOR ALL IMAGES
print("\nStarting ABC feature extraction for all samples...")
abc_features_list = []

# Iterate over each row in df_filtered to access image and mask
for index, row in df_filtered.iterrows():
    img = row['image']
    mask = row['mask']

    # Perform feature extraction only if image and mask were loaded correctly
    # (df_filtered.dropna() in Data_Loader should have already removed problematic cases, but a check here adds robustness)
    if img is not None and mask is not None:
        try:
            current_features = extract_all_abc_features(img, mask)
            abc_features_list.append(current_features)
        except Exception as e:
            # Catch specific feature extraction errors for a row
            print(f"Warning: Error during feature extraction for ID {row.get('img_id', 'N/A')} (index {index}): {e}")
            # Add NaN for features if extraction fails for this row
            abc_features_list.append({
                'rotational_asymmetry_score': np.nan,
                'compactness_score': np.nan,
                'mean_color_B': np.nan, 'mean_color_G': np.nan, 'mean_color_R': np.nan,
                'std_color_B': np.nan, 'std_color_G': np.nan, 'std_color_R': np.nan
            })
    else:
        # This case should be rare if df_filtered.dropna() works correctly, but it's a safety net.
        print(f"Warning: Missing image or mask for ID {row.get('img_id', 'N/A')} (index {index}). Adding NaN features.")
        abc_features_list.append({
            'rotational_asymmetry_score': np.nan,
            'compactness_score': np.nan,
            'mean_color_B': np.nan, 'mean_color_G': np.nan, 'mean_color_R': np.nan,
            'std_color_B': np.nan, 'std_color_G': np.nan, 'std_color_R': np.nan
        })

print("\nABC feature extraction completed.")

# 3. Convert the list of feature dictionaries into a separate DataFrame
abc_features_df = pd.DataFrame(abc_features_list)

# 4. Concatenate the new features to your existing DataFrame (df_filtered)
# We use reset_index(drop=True) on both to ensure indices align correctly
df_final = pd.concat([df_filtered.reset_index(drop=True), abc_features_df.reset_index(drop=True)], axis=1)

print("\nFinal DataFrame with ABC features (first 5 rows and selected feature columns):")
feature_cols_to_display = [col for col in df_final.columns if 'asymmetry' in col or 'compactness' in col or 'color' in col]
# add some metadata columns for context
context_cols = ['img_id', 'diagnostic', 'label']
# Filter for columns that actually exist in df_final
final_display_cols = [col for col in context_cols + feature_cols_to_display if col in df_final.columns]
print(df_final[final_display_cols].head())

# 5. Remove the 'image' and 'mask' columns
# These columns contain the NumPy arrays of the images, which are heavy and no longer needed
# once the numerical features for the ML model have been extracted.
if 'image' in df_final.columns:
    df_final = df_final.drop(columns=['image', 'mask'])
    print("\nColumns 'image' and 'mask' removed from the final DataFrame.")

print("\nFeature extraction pipeline completed.")
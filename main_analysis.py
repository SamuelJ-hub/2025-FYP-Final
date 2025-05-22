import pandas as pd
import numpy as np
from features_extractor import extract_all_abc_features


abc_features_list = []
for index, row in df_filtered.iterrows():
    img = row['image']
    mask = row['mask']

    if img is not None and mask is not None:
        current_features = extract_all_abc_features(img, mask)
        abc_features_list.append(current_features)
    else:
        # Handle cases where the image or mask was not loaded.
        # It is important that each row of the final DataFrame has the same number of columns.
        # You can add an empty dictionary or one with NaN values for the features.
        # Make sure the key names match those produced by extract_all_abc_features.
        abc_features_list.append({
            'rotational_asymmetry_score': np.nan,
            'compactness_score': np.nan,
            'mean_color_B': np.nan, 'mean_color_G': np.nan, 'mean_color_R': np.nan,
            'std_color_B': np.nan, 'std_color_G': np.nan, 'std_color_R': np.nan
        })

# Convert the list of feature dictionaries into a separate DataFrame
abc_features_df = pd.DataFrame(abc_features_list)

# Concatenate the new features to your existing DataFrame
df_final = pd.concat([df_filtered.reset_index(drop=True), abc_features_df.reset_index(drop=True)], axis=1)

print("\nDataFrame with ABC features (first 5 rows and selected feature columns):")
# You can print only some of the new columns to check
feature_cols_to_display = [col for col in df_final.columns if 'asymmetry' in col or 'compactness' in col or 'color' in col]
print(df_final[feature_cols_to_display].head())

# Remove the 'image' and 'mask' columns if they are no longer needed for the ML model
# This frees up memory and prepares the DataFrame with only numerical data.
if 'image' in df_final.columns:
    df_final = df_final.drop(columns=['image', 'mask'])

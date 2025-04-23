# -*- coding: utf-8 -*-
# Required libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# --- 1. CONFIGURATION (VERIFY THESE!) ---

csv_file_path = r'C:\Users\ruben\2025-FYP-Final\data\metadata.csv' # Path to metadata file
image_dir = r'C:\Users\ruben\Desktop\Project_Data_Science\files\final_lesion_data' # Path to folder containing images/masks subfolders

# --- Column Names in metadata.csv ---
img_id_column = 'img_id'       # <<--- VERIFY THIS --- >> (Column with 'filename.png')
diagnosis_column_name = 'diagnostic' # <<--- VERIFY THIS --- >> (Column with diagnosis)

# --- Diagnosis Labels ---
# <<--- VERIFY THIS --- >> (Exact labels in your CSV)
melanoma_labels = ['MEL']
# <<--- VERIFY THIS --- >> (Exact labels in your CSV)
non_melanoma_labels = ['NEV', 'BCC', 'ACK', 'SEK', 'SCC']

# --- File Naming and Extensions ---
images_subdir = 'images'      # Image subfolder name
image_extension = '.png'      # Image file extension

masks_subdir = 'masks'        # Mask subfolder name
mask_suffix = '_mask'         # Mask filename suffix (use '' if none)
mask_extension = '.png'       # Mask file extension

# --- 2. LOAD METADATA ---
try:
    df = pd.read_csv(csv_file_path)
    print(f"Loaded {csv_file_path}, found {len(df)} rows.")
    # Check essential columns exist
    essential_columns = [img_id_column, diagnosis_column_name]
    if not all(col in df.columns for col in essential_columns):
        print(f"Error: Missing essential columns. Need: {essential_columns}. Found: {list(df.columns)}")
        exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- 3. COUNT INITIAL DIAGNOSES ---
print(f"\nCounting diagnoses in column: '{diagnosis_column_name}'")
melanoma_count_initial = df[df[diagnosis_column_name].isin(melanoma_labels)].shape[0]
non_melanoma_count_initial = df[df[diagnosis_column_name].isin(non_melanoma_labels)].shape[0]
print(f"Initial count for {melanoma_labels}: {melanoma_count_initial}")
print(f"Initial count for {non_melanoma_labels}: {non_melanoma_count_initial}")

# --- 4. CONSTRUCT FULL FILE PATHS ---
print("\nConstructing file paths...")
try:
    # Helper function to remove '.png' from the img_id column value
    def get_base_id(image_id_with_ext):
        if isinstance(image_id_with_ext, str) and image_id_with_ext.endswith('.png'):
            return image_id_with_ext[:-4]
        return image_id_with_ext

    df['base_id'] = df[img_id_column].apply(get_base_id)
    df['image_path'] = df['base_id'].apply(lambda base_id: os.path.join(image_dir, images_subdir, f"{base_id}{image_extension}"))
    df['mask_path'] = df['base_id'].apply(lambda base_id: os.path.join(image_dir, masks_subdir, f"{base_id}{mask_suffix}{mask_extension}"))
    print("Example constructed paths:")
    print(df[['image_path', 'mask_path']].head())
except Exception as e:
    print(f"Error during path construction: {e}")
    exit()

# --- 5. FILTER DATAFRAME ---
all_relevant_labels = melanoma_labels + non_melanoma_labels
df_filtered = df[df[diagnosis_column_name].isin(all_relevant_labels)].copy()
print(f"\nFiltered DataFrame for labels: {all_relevant_labels}. Kept {len(df_filtered)} rows.")

# --- 6. ADD BINARY LABEL ---
df_filtered['label'] = df_filtered[diagnosis_column_name].apply(lambda x: 'melanoma' if x in melanoma_labels else 'non_melanoma')
print("\nCounts for binary label:")
print(df_filtered['label'].value_counts())

# --- 7. CHECK FILE EXISTENCE ---
print("\nChecking file existence...")
df_filtered['image_exists'] = df_filtered['image_path'].apply(os.path.exists)
df_filtered['mask_exists'] = df_filtered['mask_path'].apply(os.path.exists)
print("Image existence:", df_filtered['image_exists'].value_counts())
print("Mask existence:", df_filtered['mask_exists'].value_counts())

# Keep only rows where both image and mask files exist
df_filtered = df_filtered[df_filtered['image_exists'] & df_filtered['mask_exists']].copy()
print(f"Kept {len(df_filtered)} rows where both image and mask exist.")
df_filtered.reset_index(drop=True, inplace=True)

# --- 8. LOAD IMAGES AND MASKS ---
def load_image_and_mask(row):
    try:
        img = cv2.imread(os.path.normpath(row['image_path']))
        mask = cv2.imread(os.path.normpath(row['mask_path']), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: return None, None
        return img, mask
    except Exception as e:
        print(f"Error loading {row.get('base_id', 'N/A')}: {e}")
        return None, None

print("\nLoading images and masks...")
if not df_filtered.empty:
    df_filtered[['image', 'mask']] = df_filtered.apply(load_image_and_mask, axis=1, result_type='expand')
    initial_rows_for_loading = len(df_filtered)
    df_filtered.dropna(subset=['image', 'mask'], inplace=True)
    loaded_rows = len(df_filtered)
    print(f"Loaded data for {loaded_rows} out of {initial_rows_for_loading} rows.")
    if loaded_rows > 0: df_filtered.reset_index(drop=True, inplace=True)
else:
    loaded_rows = 0
    print("Skipping image loading (no files found or matched criteria).")

# --- 9. DISPLAY EXAMPLES ---
def show_image_and_mask(index, dataframe):
    required_cols = [img_id_column, 'image', 'mask', 'label'] # Use variable for img_id col name
    if dataframe is None or dataframe.empty or not all(col in dataframe.columns for col in required_cols) or index >= len(dataframe):
        print(f"Cannot display index {index}, issue with DataFrame or index.")
        return
    img_id = dataframe.loc[index, img_id_column] # Use variable
    img = dataframe.loc[index, 'image']
    mask = dataframe.loc[index, 'mask']
    label = dataframe.loc[index, 'label']
    if img is not None and mask is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title(f"ID: {img_id}\nLabel: {label}"); plt.axis('off')
        plt.subplot(1, 2, 2); plt.imshow(mask, cmap='gray'); plt.title(f"Mask ID: {img_id}"); plt.axis('off') # Use base img_id for mask title too
        plt.tight_layout(); plt.show()
    else: print(f"Data missing for index {index}, ID {img_id}.")

if loaded_rows > 0:
    print("\nDisplaying examples...")
    num_examples_to_show = min(5, loaded_rows)
    for i in range(num_examples_to_show):
        show_image_and_mask(i, df_filtered)
else:
    print("\nSkipping display (no images loaded).")

# --- END OF SCRIPT ---
print("\nScript finished.")
# df_filtered DataFrame should now contain loaded images and masks if successful.
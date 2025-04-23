# -*- coding: utf-8 -*-
# Import necessary libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# --- 1. CONFIGURATION (VERIFY THESE!) ---

# Path to the metadata CSV file
csv_file_path = r'C:\Users\ruben\2025-FYP-Final\data\metadata.csv' # CONFIRMADO

# Path to the main folder holding the 'images' and 'masks' subfolders
image_dir = r'C:\Users\ruben\Desktop\Project_Data_Science\files\final_lesion_data' # CONFIRMADO

# --- Column Names in your metadata.csv ---
img_id_column = 'img_id'       # <-- CONFIRMADO (Contiene 'nombre.png')
diagnosis_column_name = 'diagnostic' # <-- PARECE CORRECTO (Basado en ejecución anterior)

# --- Diagnosis Labels (VERIFY these are the exact labels in your 'diagnostic' column!) ---
# <<--- VERIFY THIS --- >>
melanoma_labels = ['MEL']
# <<--- VERIFY THIS --- >>
non_melanoma_labels = ['NEV', 'BCC', 'ACK', 'SEK', 'SCC'] # Revisa si hay otros o si alguno es diferente

# --- File Naming and Extensions ---
images_subdir = 'images'      # CONFIRMADO
image_extension = '.png'      # CONFIRMADO (La extensión REAL del archivo)

masks_subdir = 'masks'        # CONFIRMADO
mask_suffix = '_mask'         # CONFIRMADO
mask_extension = '.png'       # CONFIRMADO (La extensión REAL del archivo)

# --- 2. LOAD METADATA ---
try:
    df = pd.read_csv(csv_file_path)
    print(f"Successfully loaded {csv_file_path}, found {len(df)} rows.")

    essential_columns = [img_id_column, diagnosis_column_name]
    missing_cols = [col for col in essential_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing essential columns: {missing_cols}")
        print(f"Available columns are: {list(df.columns)}")
        exit()

    # print("\nUnique values in diagnostic column:", df[diagnosis_column_name].unique())

except FileNotFoundError:
    print(f"Error: File not found at '{csv_file_path}'. Please check the path.")
    exit()
except Exception as e:
    print(f"An error occurred loading the CSV: {e}")
    exit()

# --- 3. COUNT INITIAL DIAGNOSES ---
print(f"\nCounting diagnoses based on column: '{diagnosis_column_name}'")
melanoma_count_initial = df[df[diagnosis_column_name].isin(melanoma_labels)].shape[0]
non_melanoma_count_initial = df[df[diagnosis_column_name].isin(non_melanoma_labels)].shape[0]
print(f"Initial count for labels {melanoma_labels}: {melanoma_count_initial}")
print(f"Initial count for labels {non_melanoma_labels}: {non_melanoma_count_initial}")

# --- 4. CONSTRUCT FULL FILE PATHS (WITH CORRECTION FOR .png in img_id) ---
print("\nConstructing image and mask file paths...")
try:
    # Function to get base ID by removing '.png'
    def get_base_id(image_id_with_ext):
        if isinstance(image_id_with_ext, str) and image_id_with_ext.endswith('.png'):
             # Remove the last 4 characters ('.png')
            return image_id_with_ext[:-4]
        # Return original value if it's not a string or doesn't end with .png
        # Consider adding more robust error handling if needed
        return image_id_with_ext

    # Apply the function to get base ID first
    df['base_id'] = df[img_id_column].apply(get_base_id)

    # Now construct paths using the base_id
    df['image_path'] = df['base_id'].apply(lambda base_id: os.path.join(image_dir, images_subdir, f"{base_id}{image_extension}"))
    df['mask_path'] = df['base_id'].apply(lambda base_id: os.path.join(image_dir, masks_subdir, f"{base_id}{mask_suffix}{mask_extension}"))

    # Optional: Print first few paths to verify
    print("Example constructed paths:")
    print(df[['image_path', 'mask_path']].head())

except KeyError as e:
     print(f"Error creating file paths. Missing column: {e}. Check column name settings.")
     exit()
except Exception as e:
    print(f"An unexpected error occurred during path construction: {e}")
    exit()

# --- 5. FILTER DATAFRAME ---
all_relevant_labels = melanoma_labels + non_melanoma_labels
df_filtered = df[df[diagnosis_column_name].isin(all_relevant_labels)].copy()
print(f"\nFiltered DataFrame to keep only labels: {all_relevant_labels}. Kept {len(df_filtered)} rows.")

# --- 6. ADD BINARY LABEL ---
df_filtered['label'] = df_filtered[diagnosis_column_name].apply(lambda x: 'melanoma' if x in melanoma_labels else 'non_melanoma')
print("\nCounts after filtering and adding binary label:")
print(df_filtered['label'].value_counts())

# --- 7. CHECK FILE EXISTENCE ---
print("\nChecking if image and mask files exist (this may take a moment)...")
df_filtered['image_exists'] = df_filtered['image_path'].apply(os.path.exists)
df_filtered['mask_exists'] = df_filtered['mask_path'].apply(os.path.exists)

print("\nImage file existence check:")
print(df_filtered['image_exists'].value_counts())
print("\nMask file existence check:")
print(df_filtered['mask_exists'].value_counts())

missing_files_count = len(df_filtered[~(df_filtered['image_exists'] & df_filtered['mask_exists'])])
if missing_files_count > 0:
    print(f"\nWarning: Found {missing_files_count} rows where image or mask file does not exist at the constructed path.")
    df_filtered = df_filtered[df_filtered['image_exists'] & df_filtered['mask_exists']].copy()
    print(f"Removed rows with missing files. Remaining rows: {len(df_filtered)}")

df_filtered.reset_index(drop=True, inplace=True)

# --- 8. LOAD IMAGES AND MASKS ---
def load_image_and_mask(row):
    try:
        img_path_norm = os.path.normpath(row['image_path'])
        mask_path_norm = os.path.normpath(row['mask_path'])
        img = cv2.imread(img_path_norm)
        mask = cv2.imread(mask_path_norm, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: return None, None
        return img, mask
    except Exception as e:
        # Use .get() for potentially missing columns if df wasn't filtered yet
        img_id_val = row.get(img_id_column, row.get('base_id', 'N/A'))
        print(f"Error loading files for img_id {img_id_val}: {e}")
        return None, None

print("\nLoading images and masks into DataFrame (this may take time and memory)...")
# Check if df_filtered is empty before applying
if not df_filtered.empty:
    df_filtered[['image', 'mask']] = df_filtered.apply(load_image_and_mask, axis=1, result_type='expand')
    initial_rows = len(df_filtered) # Re-check length before dropna
    df_filtered.dropna(subset=['image', 'mask'], inplace=True)
    loaded_rows = len(df_filtered)
    print(f"\nSuccessfully loaded images/masks for {loaded_rows} out of {initial_rows} checked rows.")
    if loaded_rows > 0:
        df_filtered.reset_index(drop=True, inplace=True)
    else:
         print("\nWarning: No images/masks loaded successfully after filtering for existence.")
else:
    loaded_rows = 0
    print("\nSkipping image loading as no files were found or matched the criteria.")


# --- 9. DISPLAY EXAMPLES ---
def show_image_and_mask(index, dataframe):
    # Add checks for column existence before accessing .loc
    required_cols = [img_id_column, 'image', 'mask', 'label']
    if not all(col in dataframe.columns for col in required_cols):
        print("Error: Missing required columns in DataFrame for display.")
        return
    if index >= len(dataframe):
        print(f"Index {index} out of bounds.")
        return

    img_id = dataframe.loc[index, img_id_column]
    img = dataframe.loc[index, 'image']
    mask = dataframe.loc[index, 'mask']
    label = dataframe.loc[index, 'label']

    if img is not None and mask is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Image ID: {img_id}\nLabel: {label}")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask ID: {img_id}") # Assuming mask ID is same as image ID base
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Image or mask is None for index {index}, ID {img_id}")

if loaded_rows > 0:
    print("\nDisplaying first few image/mask examples...")
    num_examples_to_show = min(5, loaded_rows)
    for i in range(num_examples_to_show):
        show_image_and_mask(i, df_filtered)
else:
    print("\nSkipping example display as no images/masks were loaded.")

# --- END OF SCRIPT ---
print("\nScript finished.")
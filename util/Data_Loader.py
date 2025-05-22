import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2

# --- 1. CONFIGURATION (VERIFY THESE!) ---
# Adjust these paths and names to match your project's structure

csv_file_path = r'/Users/samuel/Desktop/ITU/Project in Data Science/2025-FYP-Final/data/metadata.csv' # Path to metadata file
image_dir = r'/Users/samuel/Desktop/ITU/Project in Data Science/final_lesion_data' # Path to folder containing images and masks subfolders

# --- Column Names in metadata.csv ---
img_id_column = 'img_id'       
diagnosis_column_name = 'diagnostic' 

# --- Diagnosis Labels ---
# These should match the values in your 'diagnostic' column for melanoma and non-melanoma
melanoma_labels = ['MEL']
non_melanoma_labels = ['NEV', 'BCC', 'ACK', 'SEK', 'SCC']

# --- File Naming and Extensions ---
images_subdir = 'images'      # Image subfolder name within image_dir
image_extension = '.png'      # Image file extension (e.g., '.jpg', '.png')

masks_subdir = 'masks'        # Mask subfolder name within image_dir
mask_suffix = '_mask'         # Mask filename suffix (e.g., '_segmentation', or '' if no suffix)
mask_extension = '.png'       # Mask file extension (e.g., '.png')

# --- Helper function for loading images and masks ---
def load_image_and_mask(row):
    """
    Loads an image and its corresponding mask from the paths specified in the DataFrame row.
    Returns the image and mask as NumPy arrays, or (None, None) if loading fails.
    """
    try:
        # os.path.normpath is useful for consistent path handling across operating systems
        img = cv2.imread(os.path.normpath(row['image_path']))
        # cv2.IMREAD_GRAYSCALE ensures the mask is loaded as a single-channel grayscale image
        mask = cv2.imread(os.path.normpath(row['mask_path']), cv2.IMREAD_GRAYSCALE)
        
        # If either is None, loading failed
        if img is None or mask is None: 
            # Print a warning but proceed; the row will be dropped later with dropna
            print(f"Warning: Failed to load for ID {row.get('base_id', 'N/A')}. Image valid: {img is not None}, Mask valid: {mask is not None}")
            return None, None
        return img, mask
    except Exception as e:
        # Catch any other unexpected errors during loading
        print(f"Unexpected error during loading for ID {row.get('base_id', 'N/A')}: {e}")
        return None, None

# --- Function to display images and masks (can be called on demand) ---
def show_image_and_mask_examples(dataframe, num_examples=5):
    """
    Displays a specified number of image and mask examples from the DataFrame.
    """
    # Check if the DataFrame is valid and contains the necessary data
    required_cols = [img_id_column, 'image', 'mask', 'label'] 
    if dataframe is None or dataframe.empty or not all(col in dataframe.columns for col in required_cols):
        print("Cannot display examples: Invalid DataFrame or missing columns.")
        return

    loaded_rows = len(dataframe)
    if loaded_rows == 0:
        print("No images/masks loaded for display.")
        return

    print(f"\nDisplaying {min(num_examples, loaded_rows)} image and mask examples:")
    # Iterate only up to the requested number of examples or available rows
    for i in range(min(num_examples, loaded_rows)):
        img_id = dataframe.loc[i, img_id_column] 
        img = dataframe.loc[i, 'image']
        mask = dataframe.loc[i, 'mask']
        label = dataframe.loc[i, 'label']

        # Ensure image and mask are not None before attempting to display
        if img is not None and mask is not None:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            # OpenCV loads in BGR, Matplotlib expects RGB for display
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"ID: {img_id}\nLabel: {label}")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray') # Masks are typically monochrome
            plt.title(f"Mask ID: {img_id}")
            plt.axis('off')
            plt.tight_layout() # Adjust subplots to prevent overlap
            plt.show()
        else: 
            print(f"Image/mask data missing for index {i}, ID {img_id}. Skipping display.")

# --- MAIN SCRIPT EXECUTION ---
if __name__ == "__main__":
    print("--- Starting Data Preprocessing ---")

    # --- 2. LOAD METADATA ---
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {csv_file_path}, found {len(df)} rows.")
        # Check if essential columns exist
        essential_columns = [img_id_column, diagnosis_column_name]
        if not all(col in df.columns for col in essential_columns):
            print(f"Error: Missing essential columns. Required: {essential_columns}. Found: {list(df.columns)}")
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
        # Helper function to remove the extension from the img_id column value (e.g., 'image_id.png' -> 'image_id')
        def get_base_id(image_id_with_ext):
            if isinstance(image_id_with_ext, str) and image_id_with_ext.endswith(image_extension):
                return image_id_with_ext[:-len(image_extension)]
            return image_id_with_ext

        df['base_id'] = df[img_id_column].apply(get_base_id)
        df['image_path'] = df['base_id'].apply(lambda base_id: os.path.join(image_dir, images_subdir, f"{base_id}{image_extension}"))
        df['mask_path'] = df['base_id'].apply(lambda base_id: os.path.join(image_dir, masks_subdir, f"{base_id}{mask_suffix}{mask_extension}"))
        print("Example constructed paths (first 2 rows):")
        print(df[['image_path', 'mask_path']].head(2))
    except Exception as e:
        print(f"Error during path construction: {e}")
        exit()

    # --- 5. FILTER DATAFRAME FOR RELEVANT DIAGNOSES ---
    all_relevant_labels = melanoma_labels + non_melanoma_labels
    df_filtered = df[df[diagnosis_column_name].isin(all_relevant_labels)].copy()
    print(f"\nDataFrame filtered for relevant diagnoses: {all_relevant_labels}. Kept {len(df_filtered)} rows.")

    # --- 6. ADD BINARY LABEL ---
    df_filtered['label'] = df_filtered[diagnosis_column_name].apply(lambda x: 'melanoma' if x in melanoma_labels else 'non_melanoma')
    print("\nCounts for binary label:")
    print(df_filtered['label'].value_counts())

    # --- 7. CHECK FILE EXISTENCE ---
    print("\nChecking file existence on disk...")
    df_filtered['image_exists'] = df_filtered['image_path'].apply(os.path.exists)
    df_filtered['mask_exists'] = df_filtered['mask_path'].apply(os.path.exists)
    
    # Report on missing files
    missing_images = df_filtered['image_exists'].value_counts().get(False, 0)
    missing_masks = df_filtered['mask_exists'].value_counts().get(False, 0)
    print(f"Missing images: {missing_images}")
    print(f"Missing masks: {missing_masks}")

    # Keep only rows where both image and mask files exist
    df_filtered = df_filtered[df_filtered['image_exists'] & df_filtered['mask_exists']].copy()
    print(f"Kept {len(df_filtered)} rows where both image and mask exist.")
    df_filtered.reset_index(drop=True, inplace=True) # Reset index after filtering

    # --- 8. LOAD IMAGES AND MASKS INTO DATAFRAME ---
    print("\nLoading images and masks into DataFrame (this might take time for large datasets)...")
    if not df_filtered.empty:
        # Apply the loading function row by row and expand results into two new columns
        df_filtered[['image', 'mask']] = df_filtered.apply(load_image_and_mask, axis=1, result_type='expand')
        
        # Remove rows for which loading failed (those with None in 'image' or 'mask')
        initial_rows_after_loading_attempt = len(df_filtered)
        df_filtered.dropna(subset=['image', 'mask'], inplace=True)
        loaded_rows_final = len(df_filtered)
        print(f"Loaded data for {loaded_rows_final} out of {initial_rows_after_loading_attempt} attempted. Removed {initial_rows_after_loading_attempt - loaded_rows_final} rows with loading errors.")
        
        if loaded_rows_final > 0: 
            df_filtered.reset_index(drop=True, inplace=True) # Reset index again if rows were removed
        else:
            print("No valid data remaining after image/mask loading.")
    else:
        loaded_rows_final = 0
        print("Skipped image loading (no files found or matched initial criteria).")

    # --- 9. DISPLAY FIRST 5 ROWS OF THE FINAL DATAFRAME (before feature extraction) ---
    print("\n--- First 5 rows of the final DataFrame (df_filtered) ---")
    # Exclude 'image' and 'mask' columns for cleaner console display, as they contain NumPy arrays
    cols_to_display = [col for col in df_filtered.columns if col not in ['image', 'mask']]
    print(df_filtered[cols_to_display].head())
    print("\n('image' and 'mask' columns contain NumPy arrays and are not shown here, but are present in the DataFrame.)")

    # --- 10. OPTIONAL: Display Image Examples (call this function when you need it) ---
    # To view examples, uncomment the line below and run the script:
    # show_image_and_mask_examples(df_filtered, num_examples=5) 
    
    print("\n--- Data Preprocessing Completed ---")
    
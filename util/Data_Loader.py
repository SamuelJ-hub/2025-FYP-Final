import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from util.img_util import read_image_bgr, read_mask_grayscale, ImageDataLoader

# --- 1. CONFIGURATION (VERIFY THESE!) ---
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
    Loads an image and its corresponding mask from the paths specified in the DataFrame row
    using functions from img_util.py.
    Returns the image and mask as NumPy arrays, or (None, None) if loading fails.
    """
    try:
        # Use the functions imported from img_util
        img = read_image_bgr(os.path.normpath(row['image_path']))
        mask = read_mask_grayscale(os.path.normpath(row['mask_path']))
        
        if img is None or mask is None: 
            print(f"Warning: Failed to load for ID {row.get('base_id', 'N/A')}. Image valid: {img is not None}, Mask valid: {mask is not None}")
            return None, None
        return img, mask
    except Exception as e:
        print(f"Unexpected error during loading for ID {row.get('base_id', 'N/A')}: {e}")
        return None, None

# --- Function to display images and masks (can be called on demand) ---
def show_image_and_mask_examples(dataframe, num_examples=5):
    """
    Displays a specified number of image and mask examples from the DataFrame.
    """
    required_cols = [img_id_column, 'image', 'mask', 'label'] 
    if dataframe is None or dataframe.empty or not all(col in dataframe.columns for col in required_cols):
        print("Cannot display examples: Invalid DataFrame or missing columns.")
        return

    loaded_rows = len(dataframe)
    if loaded_rows == 0:
        print("No images/masks loaded for display.")
        return

    print(f"\nDisplaying {min(num_examples, loaded_rows)} image and mask examples:")
    for i in range(min(num_examples, loaded_rows)):
        img_id = dataframe.loc[i, img_id_column] 
        img = dataframe.loc[i, 'image']
        mask = dataframe.loc[i, 'mask']
        label = dataframe.loc[i, 'label']

        if img is not None and mask is not None:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"ID: {img_id}\nLabel: {label}")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.title(f"Mask ID: {img_id}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else: 
            print(f"Image/mask data missing for index {i}, ID {img_id}. Skipping display.")

# --- Main function to load and preprocess data ---
def load_and_preprocess_data():
    """
    Loads metadata, constructs paths using ImageDataLoader, filters, adds binary labels,
    checks file existence, and loads images/masks into a DataFrame.
    Returns the preprocessed DataFrame (df_filtered).
    """
    print("--- Starting Data Preprocessing ---")

    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {csv_file_path}, found {len(df)} rows.")
        essential_columns = [img_id_column, diagnosis_column_name]
        if not all(col in df.columns for col in essential_columns):
            print(f"Error: Missing essential columns. Required: {essential_columns}. Found: {list(df.columns)}")
            return None 
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    print(f"\nCounting diagnoses in column: '{diagnosis_column_name}'")
    melanoma_count_initial = df[df[diagnosis_column_name].isin(melanoma_labels)].shape[0]
    non_melanoma_count_initial = df[df[diagnosis_column_name].isin(non_melanoma_labels)].shape[0]
    print(f"Initial count for {melanoma_labels}: {melanoma_count_initial}")
    print(f"Initial count for {non_melanoma_labels}: {non_melanoma_count_initial}")

    # --- Use ImageDataLoader to construct file paths ---
    print("\nConstructing file paths using ImageDataLoader...")
    try:
        # Create ImageDataLoader for images
        # The directory for images is image_dir/images_subdir
        image_loader = ImageDataLoader(
            directory=os.path.join(image_dir, images_subdir),
            file_extension=image_extension,
            suffix="" # Images typically don't have a specific suffix
        )
        # Create ImageDataLoader for masks
        # The directory for masks is image_dir/masks_subdir
        mask_loader = ImageDataLoader(
            directory=os.path.join(image_dir, masks_subdir),
            file_extension=mask_extension,
            suffix=mask_suffix # Masks have the _mask suffix
        )

        # Create temporary DataFrames from the loaders
        image_paths_df = pd.DataFrame({'image_path': list(image_loader)})
        # The base_id for images is just the filename without extension
        image_paths_df['base_id'] = image_paths_df['image_path'].apply(
            lambda p: os.path.splitext(os.path.basename(p))[0]
        )

        mask_paths_df = pd.DataFrame({'mask_path': list(mask_loader)})
        # The base_id for masks is the filename without suffix and extension
        mask_paths_df['base_id'] = mask_paths_df['mask_path'].apply(
            lambda p: os.path.splitext(os.path.basename(p))[0].replace(mask_suffix, '')
        )
        
        # Merge paths with the metadata DataFrame
        # Use a left merge to keep all CSV data and then filter based on found paths
        df = pd.merge(df, image_paths_df, on='base_id', how='left')
        df = pd.merge(df, mask_paths_df, on='base_id', how='left')
        
        print("Example constructed paths (first 2 rows after merge):")
        print(df[['base_id', 'image_path', 'mask_path']].head(2))

    except Exception as e:
        print(f"Error during path construction with ImageDataLoader: {e}")
        return None

    all_relevant_labels = melanoma_labels + non_melanoma_labels
    df_filtered = df[df[diagnosis_column_name].isin(all_relevant_labels)].copy()
    print(f"\nDataFrame filtered for relevant diagnoses: {all_relevant_labels}. Kept {len(df_filtered)} rows.")

    df_filtered['label'] = df_filtered[diagnosis_column_name].apply(lambda x: 'melanoma' if x in melanoma_labels else 'non_melanoma')
    print("\nCounts for binary label:")
    print(df_filtered['label'].value_counts())

    print("\nChecking file existence on disk (via paths generated by ImageDataLoader)...")
    # After the merge, if a path was not found by ImageDataLoader, it will be NaN.
    # We check if the paths exist after the merge, implying they were found by the loader.
    df_filtered['image_exists'] = df_filtered['image_path'].notna()
    df_filtered['mask_exists'] = df_filtered['mask_path'].notna()
    
    missing_images = df_filtered['image_exists'].value_counts().get(False, 0)
    missing_masks = df_filtered['mask_exists'].value_counts().get(False, 0)
    print(f"Missing images in expected folders/format: {missing_images}")
    print(f"Missing masks in expected folders/format: {missing_masks}")

    # Keep only rows where both image and mask paths were successfully identified and merged
    df_filtered = df_filtered[df_filtered['image_exists'] & df_filtered['mask_exists']].copy()
    print(f"Kept {len(df_filtered)} rows where both image and mask paths were successfully identified.")
    df_filtered.reset_index(drop=True, inplace=True)

    print("\nLoading images and masks into DataFrame (this might take time for large datasets)...")
    if not df_filtered.empty:
        # Use the helper function load_image_and_mask, which now uses functions from img_util.
        df_filtered[['image', 'mask']] = df_filtered.apply(load_image_and_mask, axis=1, result_type='expand')
        
        initial_rows_after_loading_attempt = len(df_filtered)
        # Drop rows where the actual image/mask loading (cv2.imread) failed
        df_filtered.dropna(subset=['image', 'mask'], inplace=True)
        loaded_rows_final = len(df_filtered)
        print(f"Loaded data for {loaded_rows_final} out of {initial_rows_after_loading_attempt} attempted. Removed {initial_rows_after_loading_attempt - loaded_rows_final} rows with loading errors.")
        
        if loaded_rows_final > 0: 
            df_filtered.reset_index(drop=True, inplace=True)
        else:
            print("No valid data remaining after image/mask loading.")
            return None
    else:
        loaded_rows_final = 0
        print("Skipped image loading (no files found or matched initial criteria).")
        return None

    print("\n--- Data Preprocessing Completed ---")
    return df_filtered # Return the preprocessed DataFrame

# --- This part ensures the script doesn't run automatically when imported ---
# It only runs if you execute Data_Loader.py directly (e.g., python Data_Loader.py)
if __name__ == "__main__":
    df_loaded = load_and_preprocess_data()
    
    if df_loaded is not None and not df_loaded.empty:
        print("\n--- First 5 rows of the loaded DataFrame (df_loaded) ---")
        # Display all columns except the heavy 'image' and 'mask' NumPy arrays
        cols_to_display = [col for col in df_loaded.columns if col not in ['image', 'mask']]
        print(df_loaded[cols_to_display].head())
        print("\n('image' and 'mask' columns contain NumPy arrays and are not shown here, but are present in the DataFrame.)")
        
        # Uncomment if you want to see image examples when running Data_Loader.py directly
        # show_image_and_mask_examples(df_loaded, num_examples=3)
        
    else:
        print("\nNo data loaded successfully to display.")
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Path to the metadata CSV file
csv_file_path = r'C:\Users\ruben\2025-FYP-Final\data\metadata.csv'

# Path to the main folder holding the 'images' and 'masks' subfolders
image_dir = r'C:\Users\ruben\Desktop\Project_Data_Science\files\final_lesion_data'

# --- Column Names in metadata.csv ---
img_id_column = 'img_id' #(Contains 'filename.png')
diagnosis_column_name = 'diagnostic'

melanoma_labels = ['MEL']

# List containing ALL exact labels for non-melanoma classes
non_melanoma_labels = ['NEV', 'BCC', 'ACK', 'SEK', 'SCC']

# --- File Naming and Extensions ---
images_subdir = 'images'
image_extension = '.png'

<<<<<<< HEAD
masks_subdir = 'masks'
mask_suffix = '_mask'
mask_extension = '.png'

try:
    # Load the CSV file into a pandas DataFrame (table)
    df = pd.read_csv(csv_file_path)
    print(f"Successfully loaded {csv_file_path}, found {len(df)} rows.")

    essential_columns = [img_id_column, diagnosis_column_name]
    missing_cols = [col for col in essential_columns if col not in df.columns]
    if missing_cols:
        # If columns are missing, print error and exit
        print(f"Error: Missing essential columns: {missing_cols}")
        print(f"Available columns are: {list(df.columns)}")
        exit()

except FileNotFoundError:
    # Handle error if the CSV file isn't found
    print(f"Error: File not found at '{csv_file_path}'. Please check the path.")
    exit()
except Exception as e:
    # Handle any other error during CSV loading
    print(f"An error occurred loading the CSV: {e}")
    exit()

print(f"\nCounting diagnoses based on column: '{diagnosis_column_name}'")
# Count rows matching the specified melanoma labels
melanoma_count_initial = df[df[diagnosis_column_name].isin(melanoma_labels)].shape[0]
# Count rows matching the specified non-melanoma labels
non_melanoma_count_initial = df[df[diagnosis_column_name].isin(non_melanoma_labels)].shape[0]
print(f"Initial count for labels {melanoma_labels}: {melanoma_count_initial}")
print(f"Initial count for labels {non_melanoma_labels}: {non_melanoma_count_initial}")
print("\nConstructing image and mask file paths...")

try:
    def get_base_id(image_id_with_ext):
        if isinstance(image_id_with_ext, str) and image_id_with_ext.endswith('.png'):
             # Remove the last 4 characters ('.png')
            return image_id_with_ext[:-4]
        # Return original value if it's not a string or doesn't end with .png
        return image_id_with_ext

    df['base_id'] = df[img_id_column].apply(get_base_id)

    df['image_path'] = df['base_id'].apply(lambda base_id: os.path.join(image_dir, images_subdir, f"{base_id}{image_extension}"))
    df['mask_path'] = df['base_id'].apply(lambda base_id: os.path.join(image_dir, masks_subdir, f"{base_id}{mask_suffix}{mask_extension}"))

    print("Example constructed paths:")
    print(df[['image_path', 'mask_path']].head())

except KeyError as e:
    print(f"Error creating file paths. Missing column: {e}. Check column name settings.")
     exit()
except Exception as e:
    print(f"An unexpected error occurred during path construction: {e}")
    exit()

all_relevant_labels = melanoma_labels + non_melanoma_labels
df_filtered = df[df[diagnosis_column_name].isin(all_relevant_labels)].copy()
print(f"\nFiltered DataFrame to keep only labels: {all_relevant_labels}. Kept {len(df_filtered)} rows.")

df_filtered['label'] = df_filtered[diagnosis_column_name].apply(lambda x: 'melanoma' if x in melanoma_labels else 'non_melanoma')
print("\nCounts after filtering and adding binary label:")
print(df_filtered['label'].value_counts())

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

def load_image_and_mask(row):
    try:
        img_path_norm = os.path.normpath(row['image_path'])
        mask_path_norm = os.path.normpath(row['mask_path'])

        # Read image and mask
        img = cv2.imread(img_path_norm)
        mask = cv2.imread(mask_path_norm, cv2.IMREAD_GRAYSCALE) # Load mask as grayscale

        # Check if loading failed
        if img is None or mask is None:
            return None, None
        return img, mask
    except Exception as e:
        # Handle any unexpected error during loading
        img_id_val = row.get(img_id_column, row.get('base_id', 'N/A')) # Get ID for error message
        print(f"Error loading files for img_id {img_id_val}: {e}")
        return None, None

print("\nLoading images and masks into DataFrame (this may take time and memory)...")
# Check if the DataFrame is empty before trying to load images
if not df_filtered.empty:
    # Apply the loading function row by row, creating 'image' and 'mask' columns
    df_filtered[['image', 'mask']] = df_filtered.apply(load_image_and_mask, axis=1, result_type='expand')

    # Check how many rows are left before removing failed loads
    initial_rows_for_loading = len(df_filtered)
    # Remove rows where either image or mask failed to load (are None)
    df_filtered.dropna(subset=['image', 'mask'], inplace=True)
    # Get the final count of loaded rows
    loaded_rows = len(df_filtered)
    print(f"\nSuccessfully loaded images/masks for {loaded_rows} out of {initial_rows_for_loading} checked rows.")
    initial_rows_for_loading = len(df_filtered)
    # Remove rows where either image or mask failed to load (are None)
    df_filtered.dropna(subset=['image', 'mask'], inplace=True)
    # Get the final count of loaded rows
    loaded_rows = len(df_filtered)
    print(f"\nSuccessfully loaded images/masks for {loaded_rows} out of {initial_rows_for_loading} checked rows.")

    if loaded_rows > 0:
        # Reset index if rows were loaded successfully
        df_filtered.reset_index(drop=True, inplace=True)
    else:
         # Print warning if no rows loaded after filtering
         print("\nWarning: No images/masks loaded successfully after filtering for existence.")
else:
    # Skip loading if no files were found in the previous step
    loaded_rows = 0
    print("\nSkipping image loading as no files were found or matched the criteria.")

def show_image_and_mask(index, dataframe):
    # Define required columns for display
    required_cols = [img_id_column, 'image', 'mask', 'label']
    # Check if DataFrame is valid and has required columns
    if dataframe is None or dataframe.empty:
        print("Error: DataFrame is empty or None, cannot display.")
        return
    if not all(col in dataframe.columns for col in required_cols):
        print(f"Error: Missing required columns in DataFrame for display. Need: {required_cols}")
        return
    # Check if index is within bounds
    if index >= len(dataframe):
        print(f"Error: Index {index} is out of bounds for DataFrame with length {len(dataframe)}.")
        return

    # Get data for the specified index
    img_id = dataframe.loc[index, img_id_column]
    img = dataframe.loc[index, 'image']
    mask = dataframe.loc[index, 'mask']
    label = dataframe.loc[index, 'label']

    # Check if image and mask data are valid
    if img is not None and mask is not None:
        # Create plot
        plt.figure(figsize=(10, 5))

        # Show image on the left
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Image ID: {img_id}\nLabel: {label}")
        plt.axis('off') # Hide axes

        # Show mask on the right
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray') # Show mask in grayscale
        plt.title(f"Mask ID: {img_id}") # Assume mask ID relates to image ID
        plt.axis('off') # Hide axes

        plt.tight_layout() # Adjust layout
        plt.show() # Display the plot
    else:
        # Print warning if data is missing for the index
        print(f"Image or mask is None for index {index}, ID {img_id}. Cannot display.")

# Display examples only if images were successfully loaded
if loaded_rows > 0:
    print("\nDisplaying first few image/mask examples...")
    # Determine how many examples to show (up to 5 or fewer if less were loaded)
    num_examples_to_show = min(5, loaded_rows)
    # Loop to show examples
    for i in range(num_examples_to_show):
        show_image_and_mask(i, df_filtered)
else:
    # Skip display if no images were loaded
    print("\nSkipping example display as no images/masks were loaded.")

print("\nScript finished.")
=======
# image_dir = "/Users/samuel/Desktop/2025-FYP-Final/data/images/"
# mask_dir = "/Users/samuel/Desktop/2025-FYP-Final/data/masks/"

# if 'img_id' in df.columns:
#     df['image_path'] = df['img_id'].map(lambda img_id: os.path.join(image_dir, f"{img_id}"))
#     df['mask_path'] = df['img_id'].map(lambda img_id: os.path.join(mask_dir, f"{img_id}_mask.png"))
# else:
#     raise ValueError("Error: 'img_id' column not found.")

# # Filter and add binary classification label
# valid_labels = melanoma_labels | non_melanoma_labels
# df_filtered = df[df['diagnostic'].isin(valid_labels)].copy()
# df_filtered['label'] = df_filtered['diagnostic'].map(lambda x: 'melanoma' if x in melanoma_labels else 'non_melanoma')

# # Check mask existence
# df_filtered['mask_exists'] = df_filtered['mask_path'].map(os.path.exists)
# print("\nExisting masks:", df_filtered['mask_exists'].value_counts())

# # Load image and mask using the readImageFile function
# def load_data(row):
#     img_rgb, img_gray = readImageFile(row['image_path'])
#     mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE) if row['mask_exists'] else None
#     return pd.Series({'image_rgb': img_rgb, 'image_gray': img_gray, 'mask': mask})

# df_loaded = df_filtered.apply(load_data, axis=1)
# df_filtered = pd.concat([df_filtered, df_loaded], axis=1)

# # Remove rows where the image was not loaded correctly
# df_filtered.dropna(subset=['image_rgb', 'mask'], inplace=True)
# df_filtered.reset_index(drop=True, inplace=True)

# print("\nNumber of images and masks loaded:", len(df_filtered))

# def show_image_and_mask(index):
#     row = df_filtered.iloc[index]
#     img_rgb, mask = row['image_rgb'], row['mask']
#     if img_rgb is not None and mask is not None:
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.imshow(img_rgb)
#         plt.title(f"Image {row['img_id']}")
#         plt.subplot(1, 2, 2)
#         plt.imshow(mask, cmap='gray')
#         plt.title(f"Mask {row['img_id']}")
#         plt.show()

# # Display some examples
# for i in range(min(5, len(df_filtered))):
#     show_image_and_mask(i)

# print("\nDataFrame with loaded images and masks (first few rows):")
# print(df_filtered.head())
>>>>>>> 3547b090fcb9a5cb4eef6819e8bb2b898b7ed231

import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

def readImageFile(file_path):
    try:
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image at {file_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return img_rgb, img_gray
    except Exception as e:
        print(f"Error reading image {file_path}: {e}")
        return None, None

csv_path = "/Users/samuel/Desktop/2025-FYP-Final/data/metadata.csv"
df = pd.read_csv(csv_path)

print("\nUnique values in diagnostic:", df['diagnostic'].unique())

melanoma_labels = {'MEL'}
non_melanoma_labels = {'NEV', 'BCC', 'ACK', 'SEK', 'SCC'}

melanoma_count = df['diagnostic'].isin(melanoma_labels).sum()
print(f'Number of patients with MEL: {melanoma_count}')

non_melanoma_count = df['diagnostic'].isin(non_melanoma_labels).sum()
print("Number of patients without melanoma:", non_melanoma_count)

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
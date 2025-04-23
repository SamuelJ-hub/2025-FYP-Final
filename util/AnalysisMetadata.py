# -*- coding: utf-8 -*-
# Import necessary libraries
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

#print("columns:", df.columns.tolist())

print("\nUnique values in diagnostic:", df['diagnostic'].unique())

melanoma_labels = ['MEL']                                  # melanoma
non_melanoma_labels = ['NEV', 'BCC', 'ACK', 'SEK', 'SCC']  # not melanoma


melanoma_count = df[df['diagnostic'].isin(melanoma_labels)].shape[0]
print(f'Number of patients with MEL: { melanoma_count}')

non_melanoma_count = df[df['diagnostic'].isin(non_melanoma_labels)].shape[0]
print("Number of patients without melanoma:", non_melanoma_count)


image_dir = "/Users/samuel/Desktop/2025-FYP-Final/data/images/"

if 'img_id' in df.columns:
    df['image_path'] = df['img_id'].apply(lambda img_id: os.path.join(image_dir, f"{img_id}.png"))

else:
    print("Error: name not found.")
    exit()


# We use this to ensure we're including only the classes we want (even if I'm pretty sure it's not necessary)
df_filtered = df[df['diagnostic'].isin(melanoma_labels + non_melanoma_labels)].copy()

# Add a 'label' column to simplify binary classification
df_filtered['label'] = df_filtered['diagnostic'].apply(lambda x: 'melanoma' if x in melanoma_labels else 'non_melanoma')


# print("\nLabel count in the filtered DataFrame:")
# print(df_filtered['label'].value_counts())

# Check if the mask file exists
df_filtered['mask_exists'] = df_filtered['mask_path'].apply(os.path.exists)
print("\nExisting masks:", df_filtered['mask_exists'].value_counts())

# Function to load image and mask (example with OpenCV)
def load_image_and_mask(row):
    try:
        img = cv2.imread(row['image_path'])
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE) # Load the mask as grayscale
        return img, mask
    except Exception as e:
        print(f"Error loading {row['img_id']}: {e}")
        return None, None

# Load images and masks into the DataFrame
df_filtered[['image', 'mask']] = df_filtered.apply(load_image_and_mask, axis=1, result_type='expand')

# Remove rows where the image or mask was not loaded correctly
df_filtered.dropna(subset=['image', 'mask'], inplace=True)
df_filtered.reset_index(drop=True, inplace=True)

print("\nNumber of images and masks uploaded:", len(df_filtered))



def show_image_and_mask(index):
    img = df_filtered.loc[index, 'image']
    mask = df_filtered.loc[index, 'mask']
    if img is not None and mask is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Image {df_filtered.loc[index, 'img_id']}")
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask {df_filtered.loc[index, 'img_id']}")
        plt.show()

# Some examples
for i in range(5):
    if i < len(df_filtered):
        show_image_and_mask(i)
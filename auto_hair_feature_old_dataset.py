import pandas as pd
import numpy as np
import cv2
import os
from util.hair_feature_extractor import analyze_hair_amount 

def run_manual_dataset_hair_feature_extraction():
    print("--- Starting Automatic Hair Feature Extraction for Manual Annotation Dataset ---")

    manual_annotations_csv_path = '/Users/samuel/Desktop/ITU/Project in Data Science/2025-FYP-Final/data/result.csv'
    manual_images_base_path = '/Users/samuel/Desktop/ITU/Project in Data Science/2025-FYP-Final/pictures'
    output_csv_path = './data/auto_hair_features_on_manual_dataset.csv'


    extracted_hair_data = []

    try:
        manual_annotations_df = pd.read_csv(manual_annotations_csv_path)

        for index, row in manual_annotations_df.iterrows():
            file_id = row['File_ID']
            image_path = os.path.join(manual_images_base_path, file_id)

            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}. Skipping this entry.")
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}. Skipping.")
                continue

            try:
                hair_features = analyze_hair_amount(image)
                hair_level_auto = hair_features['hair_level']
                hair_coverage_pct_auto = hair_features['hair_coverage_pct']
            except Exception as e:
                print(f"Error extracting features for {file_id}: {e}. Setting to NaN.")
                hair_level_auto = np.nan
                hair_coverage_pct_auto = np.nan

            extracted_hair_data.append({
                'File_ID': file_id,
                'hair_level_auto': hair_level_auto, 
                'hair_coverage_pct_auto': hair_coverage_pct_auto
            })
        
        df_auto_hair_on_manual_dataset = pd.DataFrame(extracted_hair_data)
        df_auto_hair_on_manual_dataset.to_csv(output_csv_path, index=False)
        print(f"Automatic hair features for manual dataset saved to: {output_csv_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file paths for your manual annotations CSV or images.")
    except Exception as e:
        print(f"An unexpected error occurred during feature extraction: {e}")

    print("\n--- Automatic Hair Feature Extraction Complete ---")

if __name__ == '__main__':
    run_manual_dataset_hair_feature_extraction()
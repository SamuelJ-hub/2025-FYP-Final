# This script automates the extraction of hair features for images
# that were part of the manual hair annotation task (preliminary dataset).
# The output is a CSV file containing these automatically extracted features,
# intended for comparison with the manual annotations to evaluate the
# automatic hair feature extraction method.

import pandas as pd
import numpy as np
import cv2
import os
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

# ImportS the custom function for analyzing hair amount from the util package.
from util.hair_feature_extractor import analyze_hair_amount 

# Main function for feature extraction process.
def run_manual_dataset_hair_feature_extraction():
    print("--- Starting Automatic Hair Feature Extraction for Manual Annotation Dataset ---")

    manual_annotations_csv_path = 'HairAnnotationAgreement/manual_annotation.csv'
    manual_images_base_path = 'pictures'
    output_csv_path = 'HairAnnotationAgreement/auto_hair_features_on_manual_dataset.csv'


    extracted_hair_data = []
    # Load the manual annotations CSV into a pandas DataFrame.
    try:
        manual_annotations_df = pd.read_csv(manual_annotations_csv_path)
        # Iterate through each row/entry in the manual annotations DataFrame.
        for index, row in manual_annotations_df.iterrows():
            file_id = row['File_ID']
            image_path = os.path.join(manual_images_base_path, file_id)
            # Verify that the image file exists at the constructed path.
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}. Skipping this entry.")
                continue
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}. Skipping.")
                continue
            # Attempt to extract hair features using the imported analyze_hair_amount function.
            try:
                hair_features = analyze_hair_amount(image)
                hair_level_auto = hair_features['hair_level']
                hair_coverage_pct_auto = hair_features['hair_coverage_pct']
            except Exception as e:
                print(f"Error extracting features for {file_id}: {e}. Setting to NaN.")
                hair_level_auto = np.nan
                hair_coverage_pct_auto = np.nan
            # Append the File_ID and the extracted automatic hair features to the results list.
            extracted_hair_data.append({
                'File_ID': file_id,
                'hair_level_auto': hair_level_auto, 
                'hair_coverage_pct_auto': hair_coverage_pct_auto
            })

        # After processing all images, convert the list of dictionaries to a pandas DataFrame.
        df_auto_hair_on_manual_dataset = pd.DataFrame(extracted_hair_data)
        # Save the DataFrame with automatically extracted features to a CSV file.
        df_auto_hair_on_manual_dataset.to_csv(output_csv_path, index=False)
        print(f"Automatic hair features for manual dataset saved to: {output_csv_path}")

    except FileNotFoundError as e:
        # Handle the case where the main manual annotations CSV file is not found.
        print(f"Error: {e}. Please check the file paths for your manual annotations CSV or images.")
    except Exception as e:
        # Handle errors if expected column names (like 'File_ID') are missing from the CSV.
        print(f"An unexpected error occurred during feature extraction: {e}")

    print("\n--- Automatic Hair Feature Extraction Complete ---")
    
# Runs the main function if script is executed directly.
if __name__ == '__main__':
    run_manual_dataset_hair_feature_extraction()
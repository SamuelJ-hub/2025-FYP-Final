import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_hair_agreement():
   
    print("--- Starting Hair Annotation Agreement Analysis ---")

    manual_annotations_path = 'HairAnnotationAgreement/manual_annotation.csv'
    auto_features_on_manual_dataset_path = 'HairAnnotationAgreement/auto_hair_features_on_manual_dataset.csv'

    manual_rating_column = ['Rating_1', 'Rating_2', 'Rating_3', 'Rating_4']

    try:
        manual_df = pd.read_csv(manual_annotations_path)
        auto_df_on_manual_subset = pd.read_csv(auto_features_on_manual_dataset_path)

        missing_manual_cols = [col for col in manual_rating_column if col not in manual_df.columns]
        if missing_manual_cols:
            print(f"Error: Missing manual rating columns in {manual_annotations_path}: {missing_manual_cols}")
            print(f"Available columns in manual_df: {manual_df.columns.tolist()}")
            return

        for col in manual_rating_column:
            manual_df[col] = pd.to_numeric(manual_df[col], errors='coerce')

        manual_df['manual_hair_level_avg'] = manual_df[manual_rating_column].mean(axis=1)

        manual_df['manual_hair_level_rounded'] = manual_df['manual_hair_level_avg'].round().astype(int)

        
        comparison_df = pd.merge(
            manual_df[['File_ID', 'manual_hair_level_rounded']],
            auto_df_on_manual_subset[['File_ID', 'hair_level_auto']],
            on='File_ID',
            how='inner' 
        )

        if 'manual_hair_level_rounded' not in comparison_df.columns or 'hair_level_auto' not in comparison_df.columns:
            print("Error: Required columns ('manual_hair_level_rounded' or 'hair_level_auto') not found after merge.")
            return
        
        comparison_df.dropna(subset=['manual_hair_level_rounded', 'hair_level_auto'], inplace=True)
        
        comparison_df['manual_hair_level_rounded'] = comparison_df['manual_hair_level_rounded'].astype(int)
        comparison_df['hair_level_auto'] = comparison_df['hair_level_auto'].astype(int)

        manual_labels = comparison_df['manual_hair_level_rounded']
        auto_labels = comparison_df['hair_level_auto']

        # --- Agreement Analysis ---
        print("\n--- Agreement Analysis Results ---")
        print(f"Comparing manual ratings from '{manual_rating_column}' with automatic 'hair_level_auto'.")
        print(f"Number of common images for comparison: {len(comparison_df)}")

        # Agreement Matrix (Confusion Matrix)
        print("\nAgreement Matrix (Rows: Manual Annotation, Cols: Automatic Annotation):")
        labels = sorted(np.union1d(manual_labels.unique(), auto_labels.unique()))
        conf_matrix = confusion_matrix(manual_labels, auto_labels, labels=labels)
        conf_matrix_df = pd.DataFrame(conf_matrix, index=[f'Manual_{i}' for i in labels], columns=[f'Auto_{i}' for i in labels])
        print(conf_matrix_df)

        # Cohen's Kappa Score
        kappa = cohen_kappa_score(manual_labels, auto_labels)
        print(f"\nCohen's Kappa Score: {kappa:.4f}")
        print("Interpretation of Kappa Score:")
        print(" > 0.8: Almost Perfect | 0.6-0.8: Substantial | 0.4-0.6: Moderate")
        print(" 0.2-0.4: Fair         | 0.0-0.2: Slight      | < 0.0: Poor")

        # Visualization of the Agreement Matrix
        plt.figure(figsize=(7, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=[f'Auto_{l}' for l in labels], yticklabels=[f'Manual_{l}' for l in labels])
        plt.title('Agreement Matrix: Manual vs. Automatic Hair Level')
        plt.xlabel('Automatic Annotation')
        plt.ylabel('Manual Annotation')
        plt.show()

        # Visualization of the annotation distributions
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.countplot(x=manual_labels, palette='viridis')
        plt.title(f'Distribution of Manual Hair Annotations ({manual_rating_column})')
        plt.xlabel('Hair Level (Manual)')
        plt.ylabel('Count')

        plt.subplot(1, 2, 2)
        sns.countplot(x=auto_labels, palette='viridis')
        plt.title('Distribution of Automatic Hair Annotations')
        plt.xlabel('Hair Level (Automatic)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"\nError loading data for agreement analysis: {e}")
        print("Please ensure your CSV files exist and paths are correct.")
    except Exception as e:
        print(f"An unexpected error occurred during agreement analysis: {e}")
        
    print("\n--- Hair Annotation Agreement Analysis Complete ---")

if __name__ == '__main__':
    analyze_hair_agreement()
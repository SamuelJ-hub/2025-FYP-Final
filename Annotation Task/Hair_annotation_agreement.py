import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_hair_agreement():
   
    print("--- Starting Hair Annotation Agreement Analysis ---")

    manual_annotations_path = '/Users/samuel/Desktop/ITU/Project in Data Science/2025-FYP-Final/data/result.csv'
    auto_features_on_manual_dataset_path = './data/auto_hair_features_on_manual_dataset.csv'

    manual_rating_column = 'Rating_1' 

    try:
        manual_df = pd.read_csv(manual_annotations_path)
        auto_df_on_manual_subset = pd.read_csv(auto_features_on_manual_dataset_path)

        # --- PREPARAZIONE DELLE ANNOTAZIONI MANUALI ---
        if manual_rating_column not in manual_df.columns:
            print(f"Error: Manual rating column '{manual_rating_column}' not found in {manual_annotations_path}.")
            return
        manual_df['manual_hair_level'] = manual_df[manual_rating_column]
        
        comparison_df = pd.merge(
            manual_df[['File_ID', 'manual_hair_level']],
            auto_df_on_manual_subset[['File_ID', 'hair_level_auto']],
            on='File_ID',
            how='inner' 
        )

        if 'manual_hair_level' not in comparison_df.columns or 'hair_level_auto' not in comparison_df.columns:
            print("Error: Required columns ('manual_hair_level' or 'hair_level_auto') not found after merge.")
            return
        
        comparison_df.dropna(subset=['manual_hair_level', 'hair_level_auto'], inplace=True)
        
        comparison_df['manual_hair_level'] = comparison_df['manual_hair_level'].astype(int)
        comparison_df['hair_level_auto'] = comparison_df['hair_level_auto'].astype(int)

        manual_labels = comparison_df['manual_hair_level']
        auto_labels = comparison_df['hair_level_auto']











        # --- Analisi dell'Accordo ---
        print("\n--- Agreement Analysis Results ---")
        print(f"Comparing manual ratings from '{manual_rating_column}' with automatic 'hair_level_auto'.")
        print(f"Number of common images for comparison: {len(comparison_df)}")

        # A) Matrice di Accordo (Confusion Matrix)
        print("\nAgreement Matrix (Rows: Manual Annotation, Cols: Automatic Annotation):")
        labels = sorted(np.unique(manual_labels.union(auto_labels)))
        conf_matrix = confusion_matrix(manual_labels, auto_labels, labels=labels)
        conf_matrix_df = pd.DataFrame(conf_matrix, index=[f'Manual_{i}' for i in labels], columns=[f'Auto_{i}' for i in labels])
        print(conf_matrix_df)

        # Visualizzazione della Matrice di Accordo
        plt.figure(figsize=(7, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=[f'Auto_{l}' for l in labels], yticklabels=[f'Manual_{l}' for l in labels])
        plt.title('Agreement Matrix: Manual vs. Automatic Hair Level')
        plt.xlabel('Automatic Annotation')
        plt.ylabel('Manual Annotation')
        plt.show()

        # B) Cohen's Kappa Score
        kappa = cohen_kappa_score(manual_labels, auto_labels)
        print(f"\nCohen's Kappa Score: {kappa:.4f}")
        print("Interpretation of Kappa Score:")
        print(" > 0.8: Almost Perfect | 0.6-0.8: Substantial | 0.4-0.6: Moderate")
        print(" 0.2-0.4: Fair         | 0.0-0.2: Slight      | < 0.0: Poor")

        # C) Visualizzazione della distribuzione delle annotazioni
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
# This script trains and evaluates an EXTENDED Logistic Regression model
# for melanoma classification. It uses ABC features PLUS hair features.
# It leverages the MelanomaClassifier class for training (with GridSearch),
# prediction, and model/scaler persistence. It also includes a train/validation/test split.
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve)
from util.classifier import MelanomaClassifier


def main():
    """
    Main function to run the classification pipeline.
    Handles data loading, splitting, model training/loading,
    evaluation, and saving results.
    """
    parser = argparse.ArgumentParser(description="Run the Logistic Regression Classification Pipeline with ABC + Hair Features.")
    parser.add_argument('--data_path', type=str, default='df_with_all_features.csv',
                        help="Path to the input CSV file containing features and labels.")
    parser.add_argument('--output_dir', type=str, default='./results',
                        help="Directory to save all output files (results CSV, plots).")
    parser.add_argument('--save_model_path', type=str, default='./results/best_melanoma_model.joblib',
                        help="Path to save the trained model and scaler.")
    parser.add_argument('--load_model_path', type=str, default=None,
                        help="Path to load a pre-trained model and scaler, instead of training. If provided, training is skipped.")

    args = parser.parse_args()

    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Start of Extended Classification Pipeline (ABC + Hair Features) ---")

    # 1. Load the final DataFrame with features
    try:
        df_final = pd.read_csv(args.data_path)
        print(f"DataFrame '{args.data_path}' loaded successfully. Rows: {len(df_final)}")
    except FileNotFoundError:
        print(f"Error: '{args.data_path}' not found. Make sure you have run your feature extraction and aggregation script.")
        return
    except Exception as e:
        print(f"Error while loading the DataFrame: {e}")
        return

    # 2. Define and prepare features for the extended model
    features_for_extended_model = [
        'rotational_asymmetry_score', 'compactness_score',
        'mean_color_B', 'mean_color_G', 'mean_color_R',
        'std_color_B', 'std_color_G', 'std_color_R',
        'hair_coverage_pct'
    ]

    missing_features = [f for f in features_for_extended_model if f not in df_final.columns]
    if missing_features:
        print(f"Error: The following features for the extended model were not found in the DataFrame: {missing_features}")
        print("Make sure that your feature extraction pipeline is generating them correctly.")
        return

    # Filter the DataFrame by removing rows with NaN in ANY of the features or the label
    df_processed = df_final.dropna(subset=features_for_extended_model + ['label'])
    print(f"Rows after removing NaN in ABC + Hair features and label: {len(df_processed)}")

    if df_processed.empty:
        print("No valid data remains after removing NaN. Unable to proceed with classification.")
        return

    # X: Features (ABC + Hair)
    X = df_processed[features_for_extended_model]
    # y: Target label (melanoma vs non_melanoma)
    y = df_processed['label'].apply(lambda x: 1 if x == 'melanoma' else 0)

    print(f"\nClass distribution in the extended dataset (1=melanoma, 0=non_melanoma):\n{y.value_counts()}")

    # 3. Split the dataset into Training, Validation, and Test sets
    # First split: 80% for Training+Validation, 20% for Test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"\nInitial split: Temp set (for Training+Validation) {X_temp.shape[0]} samples, Test set {X_test.shape[0]} samples.")

    # Second split: From 'Temp', 75% for Training, 25% for Validation
    # This results in an overall split of 60% Training, 20% Validation, 20% Test
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    print(f"Secondary split: Training set {X_train.shape[0]} samples, Validation set {X_val.shape[0]} samples.")
    print(f"Overall data split: Training ({X_train.shape[0]} samples ~60%), Validation ({X_val.shape[0]} samples ~20%), Test ({X_test.shape[0]} samples ~20%).")

    # 4. Initialize and Train / Load the Classifier
    melanoma_clf = MelanomaClassifier(random_state=42)

    if args.load_model_path:
        # Load a pre-trained model if the path is provided
        melanoma_clf = MelanomaClassifier.load_model(args.load_model_path)
        print("Model loaded successfully. Skipping training.")
    else:
        # Train the model if a model to load is not specified
        print("\nTraining Extended Logistic Regression model...")
        # Pass unscaled X_train data, MelanomaClassifier handles scaling
        melanoma_clf.fit(X_train, y_train) 
        # Save the trained model (and scaler)
        melanoma_clf.save_model(args.save_model_path)


    # 5. Evaluation on the Validation Set (for tuning insights)
    print("\n--- Evaluating Extended Model on Validation Set ---")
    # Pass unscaled X_val data, MelanomaClassifier handles scaling
    y_pred_val = melanoma_clf.predict(X_val) 
    y_prob_val = melanoma_clf.predict_proba(X_val)

    print(f"Accuracy (Validation): {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"Precision (Melanoma - Validation): {precision_score(y_val, y_pred_val, pos_label=1, zero_division=0):.4f}")
    print(f"Recall (Melanoma - Validation): {recall_score(y_val, y_pred_val, pos_label=1, zero_division=0):.4f}")
    print(f"F1-Score (Melanoma - Validation): {f1_score(y_val, y_pred_val, pos_label=1, zero_division=0):.4f}")
    print(f"ROC-AUC Score (Validation): {roc_auc_score(y_val, y_prob_val):.4f}")

    cm_val = confusion_matrix(y_val, y_pred_val)
    print("\nConfusion Matrix (Validation Set):\n", cm_val)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Melanoma (0)', 'Melanoma (1)'],
                yticklabels=['Non-Melanoma (0)', 'Melanoma (1)'])
    plt.title('Confusion Matrix - Extended Model (Validation Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


    # 6. FINAL EVALUATION on the Test Set (DO NOT USE FOR TUNING OR MODEL SELECTION)
    print("\n--- FINAL EVALUATION on Test Set ---")
    # Pass unscaled X_test data, MelanomaClassifier handles scaling
    y_pred_test = melanoma_clf.predict(X_test) 
    y_prob_test = melanoma_clf.predict_proba(X_test)

    print("\n--- Extended Model Evaluation Metrics (Test Set) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"Precision (Melanoma): {precision_score(y_test, y_pred_test, pos_label=1, zero_division=0):.4f}")
    print(f"Recall (Melanoma): {recall_score(y_test, y_pred_test, pos_label=1, zero_division=0):.4f}")
    print(f"F1-Score (Melanoma): {f1_score(y_test, y_pred_test, pos_label=1, zero_division=0):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_test):.4f}")

    cm_test = confusion_matrix(y_test, y_pred_test)
    print("\nConfusion Matrix (Test Set):\n", cm_test)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Melanoma (0)', 'Melanoma (1)'],
                yticklabels=['Non-Melanoma (0)', 'Melanoma (1)'])
    plt.title('Confusion Matrix - Extended Model (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    fpr_extended, tpr_extended, thresholds = roc_curve(y_test, y_prob_test)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_extended, tpr_extended, color='darkgreen', lw=2, label=f'Extended ROC Curve (AUC = {roc_auc_score(y_test, y_prob_test):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve - Extended Model (Test Set)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # save prediction results to CSV file
    # Also retrieve 'filename' if present in df_processed
    results_df = df_processed.loc[X_test.index].copy()
    results_df['true_label'] = y_test.values
    results_df['predicted_label'] = y_pred_test
    results_df['predicted_proba_melanoma'] = y_prob_test
    
    # If 'filename' is a column, include it in the saved results
    columns_to_save = ['label', 'true_label', 'predicted_label', 'predicted_proba_melanoma'] + features_for_extended_model
    if 'filename' in df_processed.columns:
        columns_to_save.insert(0, 'filename') # Add 'filename' at the beginning
    
    results_df = results_df[columns_to_save]
    results_df.to_csv(os.path.join(args.output_dir, 'results_extended_model.csv'), index=False)
    print(f"Detailed test results saved to: {os.path.join(args.output_dir, 'results_extended_model.csv')}")

    print("\n--- Extended Classification Pipeline Completed ---")

if __name__ == "__main__":
    main()
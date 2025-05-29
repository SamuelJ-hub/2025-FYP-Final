# This script trains and evaluates a baseline Logistic Regression model
# for melanoma classification using only ABC (Asymmetry, Border, Color) features.
# It includes data loading, preprocessing, model training/loading, evaluation,
# and options for saving the model and detailed results.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

# SKLEARN IMPORTS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import joblib  # For saving/loading the model and scaler

def main_baseline():
    """
    Main function to run the Baseline Classification Pipeline.
    Trains and evaluates a Logistic Regression model using only the ABC features,
    without GridSearch and with a simple train/test split.
    """
    parser = argparse.ArgumentParser(description="Run the Baseline Logistic Regression Classification Pipeline with ABC Features.")
    parser.add_argument('--data_path', type=str, default='df_with_all_features.csv',
                        help="Path to the input CSV file containing features and labels.")
    # MODIFICA QUI: Cambia il default di output_dir a './results'
    parser.add_argument('--output_dir', type=str, default='./results',
                        help="Directory to save all output files (results CSV, plots, model).")
    # Potresti voler aggiornare anche i percorsi di default per il modello,
    # in modo che siano coerenti con la nuova cartella di output.
    parser.add_argument('--save_model_path', type=str, default='./results/baseline_model.joblib',
                        help="Path to save the trained baseline model and scaler.")
    parser.add_argument('--load_model_path', type=str, default=None,
                        help="Path to load a pre-trained baseline model and scaler, instead of training.")

    args = parser.parse_args()

    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Starting Baseline Classification Pipeline ---")

    # 1. LOAD THE FINAL DATAFRAME WITH FEATURES
    try:
        df_final = pd.read_csv(args.data_path)
        print(f"DataFrame '{args.data_path}' loaded successfully. Rows: {len(df_final)}")
    except FileNotFoundError:
        print(f"Error: '{args.data_path}' not found. Make sure your feature extraction and aggregation script has generated it.")
        return
    except Exception as e:
        print(f"Error while loading the DataFrame: {e}")
        return

    # Remove rows with NaN values in the features used for the baseline
    features_for_baseline = [
        'rotational_asymmetry_score', 'compactness_score',
        'mean_color_B', 'mean_color_G', 'mean_color_R',
        'std_color_B', 'std_color_G', 'std_color_R' 
    ]

    # Check if the columns exist in the DataFrame
    missing_features = [f for f in features_for_baseline if f not in df_final.columns]
    if missing_features:
        print(f"Error: The following baseline features were not found in the DataFrame: {missing_features}")
        print("Make sure that your feature extraction pipeline is generating them correctly.")
        return

    # Filter the DataFrame by removing rows with NaN in the relevant features
    df_processed = df_final.dropna(subset=features_for_baseline + ['label'])  # Add 'label' for safety
    print(f"Rows after removing NaNs in ABC features and label: {len(df_processed)}")

    if df_processed.empty:
        print("No valid data remains after removing NaNs. Unable to proceed with classification.")
        return

    # 2. PREPARE DATA FOR THE MODEL
    # X: Features (ABC)
    X = df_processed[features_for_baseline]

    # y: Target label (melanoma vs non_melanoma)
    # Convert 'melanoma' to 1 and 'non_melanoma' to 0
    y = df_processed['label'].apply(lambda x: 1 if x == 'melanoma' else 0)

    print(f"\nClass distribution in the baseline dataset (1=melanoma, 0=non_melanoma):\n{y.value_counts()}")

    # 3. SPLIT THE DATASET INTO TRAINING AND TEST SETS
    # Use 20% of the data for testing, stratifying to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining set sizes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set sizes: X={X_test.shape}, y={y_test.shape}")

    # Initialize scaler and model
    scaler = StandardScaler()
    model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')

    # 4. STANDARDIZATION AND TRAINING/LOADING OF THE BASELINE MODEL
    if args.load_model_path:
        # Load the pre-trained model and scaler
        try:
            loaded_data = joblib.load(args.load_model_path)
            model = loaded_data['model']
            scaler = loaded_data['scaler']
            print(f"Baseline model and scaler loaded from {args.load_model_path}")
            # If loading, apply the scaler to the test data for evaluation
            X_test_scaled = scaler.transform(X_test)
        except Exception as e:
            print(f"Error loading model from {args.load_model_path}: {e}. Training new model instead.")
            # If loading fails, train a new model
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            print("\nNew Baseline model (Logistic Regression) trained.")
            joblib.dump({'model': model, 'scaler': scaler}, args.save_model_path)
            print(f"New Baseline model and scaler saved to {args.save_model_path}")
    else:
        # If no load path is specified, train the model
        print("\nStandardizing features...")
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\nTraining Baseline model (Logistic Regression)...")
        model.fit(X_train_scaled, y_train)
        print("Baseline model (Logistic Regression) trained.")
        
        # Save the trained model and scaler
        joblib.dump({'model': model, 'scaler': scaler}, args.save_model_path)
        print(f"Baseline model and scaler saved to {args.save_model_path}")

    # 5. MODEL EVALUATION
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability of the positive class (melanoma)

    print("\n--- Baseline Model Evaluation Metrics ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (Melanoma): {precision_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"Recall (Melanoma): {recall_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"F1-Score (Melanoma): {f1_score(y_test, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # 6. VISUALIZATION OF RESULTS - PLOTS
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Melanoma (0)', 'Melanoma (1)'],
                yticklabels=['Non-Melanoma (0)', 'Melanoma (1)'])
    plt.title('Confusion Matrix - Baseline Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Salviamo l'immagine nella cartella specificata
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve - Baseline Model')
    plt.legend(loc='lower right')
    plt.grid(True)
    # Salviamo l'immagine nella cartella specificata
    plt.show()

    # Optionally: save the prediction results to a CSV file
    results_df = df_processed.loc[X_test.index].copy()  # Use the original index from the processed DataFrame
    results_df['true_label'] = y_test.values
    results_df['predicted_label'] = y_pred
    results_df['predicted_proba_melanoma'] = y_prob
    
    columns_to_save = ['label', 'true_label', 'predicted_label', 'predicted_proba_melanoma'] + features_for_baseline
    if 'filename' in df_processed.columns:
        columns_to_save.insert(0, 'filename')
    
    results_df = results_df[columns_to_save]
    # Salviamo il CSV nella cartella specificata
    results_df.to_csv(os.path.join(args.output_dir, 'results_baseline_model.csv'), index=False)
    print(f"Detailed test results saved to: {os.path.join(args.output_dir, 'results_baseline_model.csv')}")

    print("\n--- Baseline Classification Pipeline Completed ---")


if __name__ == "__main__":
    main_baseline()
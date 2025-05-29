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
    Funzione principale per eseguire la pipeline di classificazione.
    Gestisce il caricamento dei dati, la suddivisione, l'addestramento/caricamento del modello,
    la valutazione e il salvataggio dei risultati.
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

    # Creare la directory di output se non esiste
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Start of Extended Classification Pipeline (ABC + Hair Features) ---")

    # 1. Caricamento del DataFrame finale con le feature
    try:
        df_final = pd.read_csv(args.data_path)
        print(f"DataFrame '{args.data_path}' loaded successfully. Rows: {len(df_final)}")
    except FileNotFoundError:
        print(f"Error: '{args.data_path}' not found. Make sure you have run your feature extraction and aggregation script.")
        return
    except Exception as e:
        print(f"Error while loading the DataFrame: {e}")
        return

    # 2. Definizione e preparazione delle feature per il modello esteso
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

    # Filtrare il DataFrame rimuovendo righe con NaN in QUALSIASI delle feature o dell'etichetta
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

    # 3. Suddivisione del dataset in Training, Validation e Test set
    # Primo split: 80% per Training+Validation, 20% per Test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"\nInitial split: Temp set (for Training+Validation) {X_temp.shape[0]} samples, Test set {X_test.shape[0]} samples.")

    # Secondo split: Dal set 'Temp', 75% per Training, 25% per Validation
    # Questo porta a una suddivisione complessiva di 60% Training, 20% Validation, 20% Test
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    print(f"Secondary split: Training set {X_train.shape[0]} samples, Validation set {X_val.shape[0]} samples.")
    print(f"Overall data split: Training ({X_train.shape[0]} samples ~60%), Validation ({X_val.shape[0]} samples ~20%), Test ({X_test.shape[0]} samples ~20%).")

    # 4. Inizializzazione e Addestramento / Caricamento del Classificatore
    melanoma_clf = MelanomaClassifier(random_state=42)

    if args.load_model_path:
        # Carica un modello pre-addestrato se il percorso è fornito
        melanoma_clf = MelanomaClassifier.load_model(args.load_model_path)
        print("Model loaded successfully. Skipping training.")
    else:
        # Addestra il modello se non è stato specificato un modello da caricare
        print("\nTraining Extended Logistic Regression model...")
        # Passa i dati X_train non scalati, la classe MelanomaClassifier si occupa dello scaling
        melanoma_clf.fit(X_train, y_train) 
        # Salva il modello addestrato (e lo scaler)
        melanoma_clf.save_model(args.save_model_path)


    # 5. Valutazione sul Set di Validazione (per approfondimenti sul tuning)
    print("\n--- Evaluating Extended Model on Validation Set ---")
    # Passa i dati X_val non scalati, la classe MelanomaClassifier si occupa dello scaling
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
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix_validation.png'))
    plt.show()


    # 6. VALUTAZIONE FINALE sul Test Set (NON USARE PER TUNING O SELEZIONE DEL MODELLO)
    print("\n--- FINAL EVALUATION on Test Set ---")
    # Passa i dati X_test non scalati, la classe MelanomaClassifier si occupa dello scaling
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
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix_test.png'))
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
    plt.savefig(os.path.join(args.output_dir, 'roc_curve_extended_model.png'))
    plt.show()

    # Opzionale: salvare i risultati delle predizioni su file CSV
    # Reperiamo anche i 'filename' se sono presenti nel df_processed
    results_df = df_processed.loc[X_test.index].copy()
    results_df['true_label'] = y_test.values
    results_df['predicted_label'] = y_pred_test
    results_df['predicted_proba_melanoma'] = y_prob_test
    
    # Se 'filename' è una colonna, la includiamo nei risultati salvati
    columns_to_save = ['label', 'true_label', 'predicted_label', 'predicted_proba_melanoma'] + features_for_extended_model
    if 'filename' in df_processed.columns:
        columns_to_save.insert(0, 'filename') # Aggiungiamo 'filename' all'inizio
    
    results_df = results_df[columns_to_save]
    results_df.to_csv(os.path.join(args.output_dir, 'results_extended_model.csv'), index=False)
    print(f"Detailed test results saved to: {os.path.join(args.output_dir, 'results_extended_model.csv')}")

    print("\n--- Extended Classification Pipeline Completed ---")

if __name__ == "__main__":
    main()
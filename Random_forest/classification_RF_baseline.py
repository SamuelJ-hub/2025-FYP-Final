# Random_forest/classification_RF_baseline.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# SKLEARN IMPORTS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

print("--- Starting Baseline Random Forest Classification Pipeline ---")

# --- 1. LOAD THE FINAL DATAFRAME WITH ALL FEATURES ---
# Path to the CSV file, assuming it's in the project root folder
# (where you run the script from)
csv_file_path = 'df_with_all_features.csv' 

try:
    df_all_features = pd.read_csv(csv_file_path)
    print(f"DataFrame '{csv_file_path}' loaded successfully. Rows: {len(df_all_features)}")
except FileNotFoundError:
    print(f"Error: '{csv_file_path}' not found in the project root directory.")
    print("Make sure 'main_analysis.py' (or your data processing script) has run and saved the file correctly in the project root.")
    exit()
except Exception as e:
    print(f"Error while loading the DataFrame: {e}")
    exit()

# --- 2. PREPARE DATA FOR THE BASELINE MODEL (ONLY ABC FEATURES) ---
print("\nPreparing data for baseline model (using only ABC features)...")

# <<--- VERIFY/EDIT THESE: List your exact ABC feature column names --- >>
# Ensure these names EXACTLY match the columns in your CSV for ABC features
abc_feature_columns = [
    'rotational_asymmetry_score', 
    'compactness_score',
    'mean_color_R', 'mean_color_G', 'mean_color_B',
    'std_color_R', 'std_color_G', 'std_color_R' 
]
label_column = 'label' # This should contain 'melanoma' or 'non_melanoma'

# Check if all required columns exist
required_cols = abc_feature_columns + [label_column]
missing_cols = [col for col in required_cols if col not in df_all_features.columns]
if missing_cols:
    print(f"Error: Missing required columns in the CSV: {missing_cols}")
    print(f"Available columns: {list(df_all_features.columns)}")
    print("Make sure your data processing script is generating and saving them correctly.")
    exit()

# Select only the necessary columns for the baseline model
df_baseline = df_all_features[required_cols].copy()

# Handle NaN values
if df_baseline[label_column].isnull().any():
    print(f"Warning: Found {df_baseline[label_column].isnull().sum()} NaN values in label column. Dropping these rows.")
    df_baseline.dropna(subset=[label_column], inplace=True)

if df_baseline[abc_feature_columns].isnull().values.any():
    print("Warning: NaN values found in ABC features. Filling with column mean.")
    for col in abc_feature_columns: 
        if df_baseline[col].isnull().any():
            df_baseline[col].fillna(df_baseline[col].mean(), inplace=True)

if df_baseline.empty:
    print("No data remains after handling NaNs. Exiting.")
    exit()
print(f"Rows after removing NaNs in ABC features & label: {len(df_baseline)}")

# X: Features (ABC)
X = df_baseline[abc_feature_columns]

# y: Target label (melanoma vs non_melanoma)
le = LabelEncoder()
y = le.fit_transform(df_baseline[label_column])
try:
    positive_class_numeric = le.transform(['melanoma'])[0]
    print(f"\n'{le.classes_[positive_class_numeric]}' (melanoma) has been encoded as: {positive_class_numeric}")
    # Determine the other class based on what's left in le.classes_
    other_class_numeric = [val for val in [0,1] if val != positive_class_numeric][0]
    print(f"Other class '{le.classes_[other_class_numeric]}' encoded as: {other_class_numeric}")
except ValueError:
    print("Error: 'melanoma' not found in label column by LabelEncoder after NaN drop. Check your 'label' column data.")
    if 'melanoma' in df_baseline[label_column].unique():
         print("Manually setting positive_class_numeric for 'melanoma' to 1 (default), but LabelEncoder failed to find it. VERIFY.")
         positive_class_numeric = 1 
    elif len(le.classes_) > 0 : 
        print(f"Using '{le.classes_[0]}' as the positive class (encoded as 0) due to 'melanoma' not being found clearly.")
        positive_class_numeric = 0 
    else:
        print("Error: No classes found by LabelEncoder. Check your 'label' column.")
        exit()
print(f"\nClass distribution in the baseline dataset (after encoding):\n{pd.Series(y).value_counts(normalize=True)}")

# --- 3. SPLIT THE DATASET INTO TRAINING AND TEST SETS ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"\nTraining set sizes: X={X_train.shape}, y={y_train.shape}")
print(f"Test set sizes: X={X_test.shape}, y={y_test.shape}")

# --- 4. FEATURE SCALING (OPTIONAL FOR RANDOM FOREST) ---
# (Skipping for simplicity in baseline Random Forest)

# --- 5. TRAIN THE BASELINE MODEL (RANDOM FOREST) ---
print("\nTraining baseline model (Random Forest)...")
baseline_rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
baseline_rf_model.fit(X_train, y_train)
print("\nBaseline model (Random Forest) trained.")

# --- 6. MODEL EVALUATION ---
y_pred = baseline_rf_model.predict(X_test)
y_prob = baseline_rf_model.predict_proba(X_test)[:, positive_class_numeric] 

print("\n--- Baseline Model Evaluation Metrics (Random Forest) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (for '{le.inverse_transform([positive_class_numeric])[0]}'): {precision_score(y_test, y_pred, pos_label=positive_class_numeric, zero_division=0):.4f}")
print(f"Recall (for '{le.inverse_transform([positive_class_numeric])[0]}'): {recall_score(y_test, y_pred, pos_label=positive_class_numeric, zero_division=0):.4f}")
print(f"F1-Score (for '{le.inverse_transform([positive_class_numeric])[0]}'): {f1_score(y_test, y_pred, pos_label=positive_class_numeric, zero_division=0):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# --- 7. VISUALIZE THE RESULTS - PLOTS ---
cm = confusion_matrix(y_test, y_pred, labels=le.transform(le.classes_))
print("\nConfusion Matrix:")
print(cm) 
readable_labels = [str(cls) for cls in le.classes_]
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=readable_labels,
            yticklabels=readable_labels)
plt.title('Confusion Matrix - Baseline Model (Random Forest, ABC Features)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=positive_class_numeric)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve - Baseline Model (Random Forest, ABC Features)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print("\n--- Baseline Classification Pipeline (Random Forest) Completed ---")
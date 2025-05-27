

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("--- Starting Baseline Random Forest Model Pipeline ---")

# --- 1. LOAD THE DATAFRAME WITH ALL FEATURES ---
csv_path = os.path.join('results', 'df_with_all_features.csv')

try:
    df_all_features = pd.read_csv(csv_path)
    print(f"DataFrame '{csv_path}' loaded successfully. Rows: {len(df_all_features)}")
except FileNotFoundError:
    print(f"Error: '{csv_path}' not found. Make sure 'main_analysis.py' has run and saved the file.")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- 2. PREPARE DATA FOR BASELINE MODEL (ONLY ABC FEATURES) ---
print("\nPreparing data for baseline model (using only ABC features)...")

# <<--- VERIFY/EDIT THESE: List your exact ABC feature column names --- >>
abc_feature_columns = [
    'rotational_asymmetry_score', 
    'compactness_score',
    'mean_color_R', 'mean_color_G', 'mean_color_B',
    'std_color_R', 'std_color_G', 'std_color_R'
    # Add/remove feature names to match exactly what's in your CSV for ABC
]

# Define the label column
label_column = 'label' # This should be 'melanoma' or 'non_melanoma'

# Check if all required columns exist
required_data_cols = abc_feature_columns + [label_column]
missing_cols = [col for col in required_data_cols if col not in df_all_features.columns]
if missing_cols:
    print(f"Error: Missing required columns in the CSV: {missing_cols}")
    print(f"Available columns: {list(df_all_features.columns)}")
    exit()

# Select only the ABC features and the label
df_baseline = df_all_features[required_data_cols].copy()

# Handle potential NaN values in features or labels before splitting
# For labels, rows with NaN labels are often dropped.
if df_baseline[label_column].isnull().any():
    print(f"Warning: Found {df_baseline[label_column].isnull().sum()} NaN values in label column. Dropping these rows.")
    df_baseline.dropna(subset=[label_column], inplace=True)

# For features, fill with mean (or median, or drop)
if df_baseline[abc_feature_columns].isnull().values.any():
    print("Warning: NaN values found in ABC features. Filling with column mean.")
    for col in abc_feature_columns: # Fill NaNs per column
        if df_baseline[col].isnull().any():
            df_baseline[col].fillna(df_baseline[col].mean(), inplace=True)

if df_baseline.empty:
    print("No data remains after handling NaNs. Exiting.")
    exit()

X = df_baseline[abc_feature_columns]
y_text = df_baseline[label_column]

# Convert text labels to numerical labels (e.g., melanoma=1, non_melanoma=0)
le = LabelEncoder()
y = le.fit_transform(y_text)
# print(f"Labels '{le.classes_[0]}' and '{le.classes_[1]}' encoded to 0 and 1 respectively.")
# It's good to know which class is 0 and which is 1 for interpreting confusion matrix
# Typically, LabelEncoder sorts alphabetically, so if 'melanoma' < 'non_melanoma', melanoma might be 0.
# Or, if you want to be sure 'melanoma' is the positive class (1):
# y = np.where(y_text == 'melanoma', 1, 0)
# positive_class_label = 1 # if 'melanoma' is 1

# --- 3. SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"\nData split: Training set {X_train.shape[0]} samples, Testing set {X_test.shape[0]} samples.")

# --- 4. FEATURE SCALING (Optional but often good for Random Forest, though less critical than for SVM/Logistic Regression) ---
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# print("\nFeatures scaled (using X_train_scaled, X_test_scaled for model).")
# For simplicity and because Random Forest is less sensitive, we might skip scaling for this first baseline.
# If you use scaling, use X_train_scaled and X_test_scaled below.
X_train_to_use = X_train
X_test_to_use = X_test


# --- 5. TRAIN BASELINE RANDOM FOREST MODEL ---
print("\nTraining baseline Random Forest model...")
# class_weight='balanced' can help with imbalanced datasets
baseline_rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
baseline_rf_model.fit(X_train_to_use, y_train)
print("Baseline Random Forest model trained.")

# --- 6. EVALUATE BASELINE MODEL ---
print("\nEvaluating baseline Random Forest model...")
y_pred_baseline = baseline_rf_model.predict(X_test_to_use)
y_prob_baseline = baseline_rf_model.predict_proba(X_test_to_use)[:, 1] # Probabilities for the positive class

print("\n--- Baseline Random Forest Model Evaluation Metrics ---")
accuracy = accuracy_score(y_test, y_pred_baseline)
print(f"Accuracy: {accuracy:.4f}")

# Determine positive label for precision/recall/F1 (e.g., if melanoma is encoded as 1)
# Find out what 'melanoma' was encoded to:
positive_label_numeric = le.transform(['melanoma'])[0] if 'melanoma' in le.classes_ else 1 
# (Fallback to 1 if 'melanoma' somehow not in original labels, though it should be)

print(f"Precision (for class '{le.inverse_transform([positive_label_numeric])[0]}'): {precision_score(y_test, y_pred_baseline, pos_label=positive_label_numeric, zero_division=0):.4f}")
print(f"Recall (for class '{le.inverse_transform([positive_label_numeric])[0]}'): {recall_score(y_test, y_pred_baseline, pos_label=positive_label_numeric, zero_division=0):.4f}")
print(f"F1-Score (for class '{le.inverse_transform([positive_label_numeric])[0]}'): {f1_score(y_test, y_pred_baseline, pos_label=positive_label_numeric, zero_division=0):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_baseline):.4f}")

print("\nConfusion Matrix (Baseline Random Forest):")
cm = confusion_matrix(y_test, y_pred_baseline, labels=le.transform(le.classes_))
print(cm)

# For readable labels in confusion matrix plot and classification report
class_names_report = [str(cls) for cls in le.classes_]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names_report, yticklabels=class_names_report)
plt.title('Confusion Matrix - Baseline Random Forest (ABC Features)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\nClassification Report (Baseline Random Forest):")
print(classification_report(y_test, y_pred_baseline, target_names=class_names_report, zero_division=0))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_baseline, pos_label=positive_label_numeric)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='purple', lw=2, label=f'Random Forest ROC Curve (AUC = {roc_auc_score(y_test, y_prob_baseline):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity/Recall)')
plt.title('ROC Curve - Baseline Random Forest (ABC Features)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print("\n--- Baseline Random Forest Model Pipeline Completed ---")
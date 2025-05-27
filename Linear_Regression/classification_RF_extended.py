# train_extended_random_forest.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("--- Starting Extended Random Forest Model Pipeline (ABC + Hair Features) ---")

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

# --- 2. PREPARE DATA FOR EXTENDED MODEL (ABC + HAIR FEATURES) ---
print("\nPreparing data for extended model (using ABC + Hair features)...")

# <<--- VERIFY/EDIT THESE: List your exact ABC feature column names AND YOUR HAIR FEATURE COLUMN NAMES --- >>
features_for_extended_model = [
    # ABC Features (same as baseline)
    'rotational_asymmetry_score', 
    'compactness_score',
    'mean_color_R', 'mean_color_G', 'mean_color_B',
    'std_color_R', 'std_color_G', 'std_color_R',
    # Hair Features (ADD THE EXACT NAMES OF YOUR HAIR FEATURE COLUMNS HERE)
    # For example, if you created 'hair_level' and 'hair_coverage_pct':
    'hair_level',             # <<--- EXAMPLE: VERIFY OR CHANGE
    'hair_coverage_pct'       # <<--- EXAMPLE: VERIFY OR CHANGE
    # Add/remove feature names to match exactly what's in your CSV
]

# Define the label column
label_column = 'label' # This should be 'melanoma' or 'non_melanoma'

# Check if all required columns exist
required_data_cols_extended = features_for_extended_model + [label_column]
missing_cols_extended = [col for col in required_data_cols_extended if col not in df_all_features.columns]
if missing_cols_extended:
    print(f"Error: Missing required columns in the CSV for the extended model: {missing_cols_extended}")
    print(f"Available columns: {list(df_all_features.columns)}")
    exit()

# Select only the specified features and the label
df_extended = df_all_features[required_data_cols_extended].copy()

# Handle potential NaN values
if df_extended[label_column].isnull().any():
    print(f"Warning: Found {df_extended[label_column].isnull().sum()} NaN values in label column. Dropping these rows.")
    df_extended.dropna(subset=[label_column], inplace=True)

if df_extended[features_for_extended_model].isnull().values.any():
    print("Warning: NaN values found in features for extended model. Filling with column mean.")
    for col in features_for_extended_model:
        if df_extended[col].isnull().any():
            df_extended[col].fillna(df_extended[col].mean(), inplace=True)

if df_extended.empty:
    print("No data remains after handling NaNs for extended model. Exiting.")
    exit()

X_extended = df_extended[features_for_extended_model]
y_text_extended = df_extended[label_column]

# Convert text labels to numerical labels
le_extended = LabelEncoder() # Use a new encoder instance or ensure it's fit on the same full set of labels
y_extended = le_extended.fit_transform(y_text_extended)
# print(f"Labels for extended model '{le_extended.classes_[0]}' and '{le_extended.classes_[1]}' encoded to 0 and 1.")

# --- 3. SPLIT DATA (EXTENDED) ---
X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(X_extended, y_extended, test_size=0.20, random_state=42, stratify=y_extended)
print(f"\nData split for extended model: Training set {X_train_ext.shape[0]} samples, Testing set {X_test_ext.shape[0]} samples.")

# --- 4. FEATURE SCALING (EXTENDED - Optional for Random Forest) ---
# scaler_ext = StandardScaler()
# X_train_ext_scaled = scaler_ext.fit_transform(X_train_ext)
# X_test_ext_scaled = scaler_ext.transform(X_test_ext)
# print("\nExtended features scaled.")
# X_train_to_use_ext = X_train_ext_scaled
# X_test_to_use_ext = X_test_ext_scaled
X_train_to_use_ext = X_train_ext # Using non-scaled for Random Forest simplicity
X_test_to_use_ext = X_test_ext


# --- 5. TRAIN EXTENDED RANDOM FOREST MODEL ---
print("\nTraining extended Random Forest model (ABC + Hair)...")
extended_rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
extended_rf_model.fit(X_train_to_use_ext, y_train_ext)
print("Extended Random Forest model trained.")

# --- 6. EVALUATE EXTENDED MODEL ---
print("\nEvaluating extended Random Forest model...")
y_pred_extended = extended_rf_model.predict(X_test_to_use_ext)
y_prob_extended = extended_rf_model.predict_proba(X_test_to_use_ext)[:, 1]

print("\n--- Extended Random Forest Model Evaluation Metrics ---")
accuracy_ext = accuracy_score(y_test_ext, y_pred_extended)
print(f"Accuracy (Extended): {accuracy_ext:.4f}")

positive_label_numeric_ext = le_extended.transform(['melanoma'])[0] if 'melanoma' in le_extended.classes_ else 1

print(f"Precision (for class '{le_extended.inverse_transform([positive_label_numeric_ext])[0]}'): {precision_score(y_test_ext, y_pred_extended, pos_label=positive_label_numeric_ext, zero_division=0):.4f}")
print(f"Recall (for class '{le_extended.inverse_transform([positive_label_numeric_ext])[0]}'): {recall_score(y_test_ext, y_pred_extended, pos_label=positive_label_numeric_ext, zero_division=0):.4f}")
print(f"F1-Score (for class '{le_extended.inverse_transform([positive_label_numeric_ext])[0]}'): {f1_score(y_test_ext, y_pred_extended, pos_label=positive_label_numeric_ext, zero_division=0):.4f}")
print(f"ROC-AUC Score (Extended): {roc_auc_score(y_test_ext, y_prob_extended):.4f}")

print("\nConfusion Matrix (Extended Random Forest):")
cm_ext = confusion_matrix(y_test_ext, y_pred_extended, labels=le_extended.transform(le_extended.classes_))
print(cm_ext)

class_names_report_ext = [str(cls) for cls in le_extended.classes_]

plt.figure(figsize=(6, 5))
sns.heatmap(cm_ext, annot=True, fmt='d', cmap='Greens', cbar=False, # Changed cmap for distinction
            xticklabels=class_names_report_ext, yticklabels=class_names_report_ext)
plt.title('Confusion Matrix - Extended RF (ABC + Hair)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\nClassification Report (Extended Random Forest):")
print(classification_report(y_test_ext, y_pred_extended, target_names=class_names_report_ext, zero_division=0))

# ROC Curve for Extended Model
fpr_ext, tpr_ext, thresholds_ext = roc_curve(y_test_ext, y_prob_extended, pos_label=positive_label_numeric_ext)
plt.figure(figsize=(7, 6))
plt.plot(fpr_ext, tpr_ext, color='green', lw=2, label=f'Extended RF ROC Curve (AUC = {roc_auc_score(y_test_ext, y_prob_extended):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity/Recall)')
plt.title('ROC Curve - Extended Random Forest (ABC + Hair)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print("\n--- Extended Random Forest Model Pipeline Completed ---")
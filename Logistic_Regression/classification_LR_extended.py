import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Start of Extended Classification Pipeline (ABC + Hair Features) ---")

# 1. Load the final DataFrame with features
# Make sure that 'df_with_all_features.csv' is correctly generated and located at the project root
try:
    # Direct path if the file is in the same directory as the script or in the project root
    df_final = pd.read_csv('df_with_all_features.csv')
    print(f"DataFrame 'df_with_all_features.csv' loaded successfully. Rows: {len(df_final)}")
except FileNotFoundError:
    print("Error: 'df_with_all_features.csv' not found. Make sure you have run your feature extraction and aggregation script.")
    exit()
except Exception as e:
    print(f"Error while loading the DataFrame: {e}")
    exit()

# 2. Definition and Preparation of Features for the Extended Model
# These are the ABC features + hair features
features_for_extended_model = [
    'rotational_asymmetry_score',
    'compactness_score',
    'mean_color_B', 'mean_color_G', 'mean_color_R',
    'std_color_B', 'std_color_G', 'std_color_R',
    'hair_coverage_pct' 
]

# Check if all required columns exist in the DataFrame
missing_features = [f for f in features_for_extended_model if f not in df_final.columns]
if missing_features:
    print(f"Error: The following features for the extended model were not found in the DataFrame: {missing_features}")
    print("Make sure that your feature extraction pipeline is generating them correctly.")
    exit()

# Filter the DataFrame by removing rows with NaN in ANY of the features or the label
# It is crucial to remove NaNs before splitting the data
df_processed = df_final.dropna(subset=features_for_extended_model + ['label'])
print(f"Rows after removing NaN in ABC + Hair features and label: {len(df_processed)}")

if df_processed.empty:
    print("No valid data remains after removing NaN. Unable to proceed with classification.")
    exit()

# X: Features (ABC + Hair)
X = df_processed[features_for_extended_model]

# y: Target label (melanoma vs non_melanoma)
y = df_processed['label'].apply(lambda x: 1 if x == 'melanoma' else 0)

print(f"\nClass distribution in the extended dataset (1=melanoma, 0=non_melanoma):\n{y.value_counts()}")


# 3. Split the dataset into Training, Validation, and Test sets
# First split: 80% for Training+Validation, 20% for Test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"\nInitial split: Temp set (for Training+Validation) {X_temp.shape[0]} samples, Test set {X_test.shape[0]} samples.")

# Second split: From the 'Temp' set, 75% for Training, 25% for Validation
# This results in an overall split of 60% Training, 20% Validation, 20% Test
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
print(f"Secondary split: Training set {X_train.shape[0]} samples, Validation set {X_val.shape[0]} samples.")
print(f"Overall data split: Training ({X_train.shape[0]} samples ~60%), Validation ({X_val.shape[0]} samples ~20%), Test ({X_test.shape[0]} samples ~20%).")



# 4. Standardize the features
# The scaler is fitted only on the training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# The same transformation is applied to the validation and test sets
X_val_scaled = scaler.transform(X_val) 
X_test_scaled = scaler.transform(X_test)

print("\nFeatures standardized using StandardScaler (fitted on training data).")

# 5. Train the Extended Model (Logistic Regression) with Hyperparameter Tuning
print("\nTraining Extended Logistic Regression model with hyperparameter tuning...")

# Define the base Logistic Regression model
# class_weight='balanced' is useful for imbalanced datasets
# max_iter to ensure convergence for some solvers
lr_model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced', max_iter=1000)

# Define the hyperparameter grid to explore
# 'C' is the inverse of regularization strength. Smaller values = more regularization
# 'penalty' defines the type of regularization (L1 or L2)
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

# Initialize GridSearchCV
# cv=5 means 5-fold cross-validation on the training set
# scoring='recall' means the tuning goal is to maximize recall for the positive class (Melanoma)
# n_jobs=-1 uses all available cores to speed up the search
grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='recall', verbose=1, n_jobs=-1)

# Run the search on the training set
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest hyperparameters found: {grid_search.best_params_}")
print(f"Best score (Mean Recall on cross-validation of training set): {grid_search.best_score_:.4f}")

# The best model trained with the optimal hyperparameters
best_lr_model = grid_search.best_estimator_
print("Extended Logistic Regression model trained with optimal hyperparameters.")


# 6. Evaluate the Extended Model on the Validation Set (for tuning insights)
print("\n--- Evaluating Extended Model on Validation Set ---")
y_pred_val = best_lr_model.predict(X_val_scaled)
y_prob_val = best_lr_model.predict_proba(X_val_scaled)[:, 1] # Probability for the positive class (Melanoma)

print(f"Accuracy (Validation): {accuracy_score(y_val, y_pred_val):.4f}")
print(f"Precision (Melanoma - Validation): {precision_score(y_val, y_pred_val, pos_label=1, zero_division=0):.4f}")
print(f"Recall (Melanoma - Validation): {recall_score(y_val, y_pred_val, pos_label=1, zero_division=0):.4f}")
print(f"F1-Score (Melanoma - Validation): {f1_score(y_val, y_pred_val, pos_label=1, zero_division=0):.4f}")
print(f"ROC-AUC Score (Validation): {roc_auc_score(y_val, y_prob_val):.4f}")

# Confusion Matrix for Validation Set
cm_val = confusion_matrix(y_val, y_pred_val)
print("\nConfusion Matrix (Validation Set):")
print(cm_val)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Melanoma (0)', 'Melanoma (1)'],
            yticklabels=['Non-Melanoma (0)', 'Melanoma (1)'])
plt.title('Confusion Matrix - Extended Model (Validation Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# 7. FINAL EVALUATION ON TEST SET (DO NOT USE FOR TUNING OR MODEL SELECTION)
# This is the final set, results here are the most unbiased estimate of model performance
print("\n--- FINAL EVALUATION on Test Set ---")
y_pred_test = best_lr_model.predict(X_test_scaled)
y_prob_test = best_lr_model.predict_proba(X_test_scaled)[:, 1] # Probability for the positive class (Melanoma)

print("\n--- Extended Model Evaluation Metrics (Test Set) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision (Melanoma): {precision_score(y_test, y_pred_test, pos_label=1, zero_division=0):.4f}")
print(f"Recall (Melanoma): {recall_score(y_test, y_pred_test, pos_label=1, zero_division=0):.4f}")
print(f"F1-Score (Melanoma): {f1_score(y_test, y_pred_test, pos_label=1, zero_division=0):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_test):.4f}")

# Confusion Matrix for Test Set
cm_test = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix (Test Set):")
print(cm_test)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Melanoma (0)', 'Melanoma (1)'],
            yticklabels=['Non-Melanoma (0)', 'Melanoma (1)'])
plt.title('Confusion Matrix - Extended Model (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC Curve for Test Set
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

print("\n--- Extended Classification Pipeline Completed ---")
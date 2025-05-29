# This module defines the MelanomaClassifier class, which handles
# training, prediction, saving, and loading of a Logistic Regression model
# for melanoma classification, including feature scaling and hyperparameter tuning.

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

class MelanomaClassifier:
    """
    Trains, predicts, and manages a melanoma classifier using Logistic Regression
    with standardization and GridSearch for hyperparameter tuning.
    """
    def __init__(self, random_state=42):
        # Constructor: Initializes the scaler and model (to be trained).
        self.scaler = StandardScaler()
        self.model = None  # The model will be initialized after training
        self.random_state = random_state

    def fit(self, X_train, y_train):
        # Scales training data and trains the Logistic Regression model using GridSearchCV.
        # Optimizes for 'recall' to better identify melanoma cases.
        print("Scaling training data...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        # Fit the scaler on training data and transform it (normalize).

        print("Starting GridSearchCV for Logistic Regression...")
        lr_base = LogisticRegression(random_state=self.random_state, solver='liblinear', class_weight='balanced', max_iter=1000)
        
        # Define parameter grid for hyperparameter tuning.
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']
        }
        
        # Perform GridSearch with 5-fold cross-validation.
        grid_search = GridSearchCV(lr_base, param_grid, cv=5, scoring='recall', verbose=1, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        self.model = grid_search.best_estimator_
        print(f"Best hyperparameters found: {grid_search.best_params_}")
        print(f"Best score (Mean Recall on cross-validation): {grid_search.best_score_:.4f}")
        print("Classifier training complete.")
        return self.model
         # Store the best model found by GridSearch.


    def predict(self, X):
        """
        Makes binary predictions on the data, scaling the features first.
        
        Args:
            X (pd.DataFrame or np.array): Features to predict, unscaled.
        
        Returns:
            np.array: Binary predictions (0 or 1).
        """
        # This function makes predictions (0 for non-melanoma, 1 for melanoma) on new data (X).
        # It first scales the new data using the same scaler that was set up during training.
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call .fit() first.")
        X_scaled = self.scaler.transform(X)
        # Scale the input data using the already fitted scaler.
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """
        Returns the prediction probabilities for the positive class (Melanoma),
        scaling the features first.
        
        Args:
            X (pd.DataFrame or np.array): Features to predict, unscaled.
        
        Returns:
            np.array: Probability of the positive class.
        """
        # This function predicts the probability that each sample belongs to the positive class (melanoma).
        # It's useful to see how confident the model is.
        # Like .predict(), it scales the new data first.


        if self.model is None:# Check if the model has been trained.
            raise RuntimeError("Model has not been trained yet. Call .fit() first.")
        X_scaled = self.scaler.transform(X) # Scale the input data.
        return self.model.predict_proba(X_scaled)[:, 1] 
        # Predict probabilities. `predict_proba` returns probabilities for all classes.
        # We take [:, 1] to get the probability for the positive class (melanoma) only.

    def save_model(self, path):
        """
        Saves the trained model and scaler in a single file.
        
        Args:
            path (str): File path where to save the model.
        """
        # This function saves the trained model and the scaler to a file.
        # This allows you to load and reuse them later without retraining.
        if self.model is None:  # Check if there's a model to save.
            raise RuntimeError("No model to save. Train the model first.")
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        # Save both the model and the scaler (which is important for new data) together.
        print(f"Model and scaler saved to {path}")

    @classmethod
    def load_model(cls, path):
        """
        Loads a trained model and scaler from a file.
        
        Args:
            path (str): File path from which to load the model.
        
        Returns:
            MelanomaClassifier: An instance of MelanomaClassifier with the loaded model.
        """
        # This function loads a previously saved model and scaler from a file.
        # It's a class method, meaning you can call it on the class itself (MelanomaClassifier.load_model(...))
        # without needing to create an object first.


        data = joblib.load(path) # Load the dictionary containing the model and scaler.
        classifier = cls() # Create a new MelanomaClassifier object.
        # Put the loaded model and scaler into this new object.
        classifier.model = data['model']
        classifier.scaler = data['scaler']
        print(f"Model and scaler loaded from {path}")
        return classifier

# This block of code runs only if you execute this script (classifier.py) directly.
# It won't run if you import MelanomaClassifier into another script.
# It's often used for simple tests or to show how to use the module.
if __name__ == '__main__':
    print("This is the classifier module. It defines the model's logic.")
    print("To run the full pipeline (training, evaluation, saving), execute the main script.")

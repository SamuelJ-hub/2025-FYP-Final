import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

class MelanomaClassifier:
    """
    Class to train, predict, and manage a melanoma classifier based on
    Logistic Regression, including standardization and GridSearch.
    """
    def __init__(self, random_state=42):
        self.scaler = StandardScaler()
        self.model = None  # The model will be initialized after training
        self.random_state = random_state

    def fit(self, X_train, y_train):
        """
        Fits the scaler on the training data and then the classifier with GridSearch.
        
        Args:
            X_train (pd.DataFrame or np.array): Unscaled training features.
            y_train (pd.Series or np.array): Training labels.
        
        Returns:
            sklearn.linear_model.LogisticRegression: The trained and optimized model.
        """
        print("Scaling training data...")
        X_train_scaled = self.scaler.fit_transform(X_train)

        print("Starting GridSearchCV for Logistic Regression...")
        # Definition of the base model and hyperparameter grid
        lr_base = LogisticRegression(random_state=self.random_state, solver='liblinear', class_weight='balanced', max_iter=1000)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']
        }
        
        # Initialization and execution of GridSearch
        grid_search = GridSearchCV(lr_base, param_grid, cv=5, scoring='recall', verbose=1, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        self.model = grid_search.best_estimator_
        print(f"Best hyperparameters found: {grid_search.best_params_}")
        print(f"Best score (Mean Recall on cross-validation): {grid_search.best_score_:.4f}")
        print("Classifier training complete.")
        return self.model

    def predict(self, X):
        """
        Makes binary predictions on the data, scaling the features first.
        
        Args:
            X (pd.DataFrame or np.array): Features to predict, unscaled.
        
        Returns:
            np.array: Binary predictions (0 or 1).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call .fit() first.")
        X_scaled = self.scaler.transform(X)
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
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call .fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1] # Probability for class 1 (Melanoma)

    def save_model(self, path):
        """
        Saves the trained model and scaler in a single file.
        
        Args:
            path (str): File path where to save the model.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
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
        data = joblib.load(path)
        classifier = cls() # Create a new instance of the class
        classifier.model = data['model']
        classifier.scaler = data['scaler']
        print(f"Model and scaler loaded from {path}")
        return classifier

# This block is executed only if classifier.py is run directly,
# not when it is imported. Useful for small module tests.
if __name__ == '__main__':
    print("This is the classifier module. It defines the model's logic.")
    print("To run the full pipeline (training, evaluation, saving), execute the main script.")
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score


class SVMClassifier:
    def __init__(self, use_pca=True, pca_components=100):
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.scaler = StandardScaler()
        self.svm = None
        self.pca = None
        
    def train(self, X_train, y_train):
        print("Scaling training data...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.use_pca:
            print(f"Applying PCA with {self.pca_components} components...")
            self.pca = PCA(n_components=self.pca_components, random_state=42)
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
        
        print("Training SVM classifier...")
        self.svm = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            random_state=42,
            probability=True
        )
        self.svm.fit(X_train_scaled, y_train)
        print("Training completed!")
        
    def predict(self, X):
        if self.svm is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.use_pca and self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        return self.svm.predict(X_scaled)
    
    def predict_proba(self, X):
        if self.svm is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.use_pca and self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        return self.svm.predict_proba(X_scaled)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Cat', 'Dog'])
        
        return accuracy, report
    
    def save_model(self, filepath):
        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'pca': self.pca,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.svm = model_data['svm']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.use_pca = model_data['use_pca']
        self.pca_components = model_data['pca_components']
        print(f"Model loaded from {filepath}")

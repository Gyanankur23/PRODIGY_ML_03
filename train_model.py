import os
import sys
from pathlib import Path

from preprocessing import DataPreprocessor
from classifier import SVMClassifier


def main():
    data_dir = Path(r"c:\Users\Gyanankur Baruah\Downloads\Asirra_ cat vs dogs")
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    
    print("Initializing data preprocessor...")
    preprocessor = DataPreprocessor(data_dir, img_size=(64, 64))
    
    print("Loading images...")
    images, labels = preprocessor.load_images(max_samples_per_class=500)
    
    print(f"Dataset shape: {images.shape}")
    print(f"Labels distribution: Cats={sum(labels==0)}, Dogs={sum(labels==1)}")
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = preprocessor.get_train_test_split(
        test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    print("Initializing SVM classifier...")
    classifier = SVMClassifier(use_pca=True, pca_components=100)
    
    print("Training model...")
    classifier.train(X_train, y_train)
    
    print("Evaluating model...")
    accuracy, report = classifier.evaluate(X_test, y_test)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    
    model_path = model_dir / "svm_cat_dog_classifier.pkl"
    classifier.save_model(model_path)
    print(f"\nModel saved successfully to {model_path}")


if __name__ == "__main__":
    main()

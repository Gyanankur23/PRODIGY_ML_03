import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


class DataPreprocessor:
    def __init__(self, data_dir, img_size=(64, 64)):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.images = []
        self.labels = []
        
    def load_images(self, max_samples_per_class=None):
        cat_files = sorted(self.data_dir.glob('cat.*.jpg'))
        dog_files = sorted(self.data_dir.glob('dog.*.jpg'))
        
        if max_samples_per_class:
            cat_files = cat_files[:max_samples_per_class]
            dog_files = dog_files[:max_samples_per_class]
        
        print(f"Loading {len(cat_files)} cat images and {len(dog_files)} dog images...")
        
        for img_path in cat_files:
            img = self._load_and_preprocess_image(img_path)
            if img is not None:
                self.images.append(img)
                self.labels.append(0)  # 0 for cat
        
        for img_path in dog_files:
            img = self._load_and_preprocess_image(img_path)
            if img is not None:
                self.images.append(img)
                self.labels.append(1)  # 1 for dog
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f"Successfully loaded {len(self.images)} images")
        return self.images, self.labels
    
    def _load_and_preprocess_image(self, img_path):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            img = cv2.resize(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            return img.flatten()
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        if len(self.images) == 0:
            raise ValueError("No images loaded. Call load_images() first.")
        
        return train_test_split(
            self.images, self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels
        )

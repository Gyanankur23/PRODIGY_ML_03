# Cat vs Dog Image Classifier

A Support Vector Machine (SVM) based image classifier that distinguishes between cat and dog images. This project uses traditional machine learning techniques with PCA dimensionality reduction for efficient classification.

## Features

- SVM classifier with RBF kernel
- PCA dimensionality reduction for feature optimization
- StandardScaler for feature normalization
- Streamlit web interface for easy image classification
- Professional UI with confidence scores

## Project Structure

```
PRODIGY_ML_03/
├── preprocessing.py      # Data loading and preprocessing
├── classifier.py         # SVM classifier implementation
├── train_model.py        # Model training script
├── app.py               # Streamlit web application
├── requirements.txt      # Python dependencies
├── README.md           # Project documentation
└── models/             # Trained model directory
    └── svm_cat_dog_classifier.pkl
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Gyanankur23/PRODIGY_ML_03.git
cd PRODIGY_ML_03
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

Before running the application, you need to train the model:

1. Ensure your dataset is in the correct location (update the path in `train_model.py`)
2. Run the training script:
```bash
python train_model.py
```

This will:
- Load and preprocess images from the dataset
- Split data into training and test sets
- Train the SVM classifier with PCA
- Evaluate the model performance
- Save the trained model to `models/svm_cat_dog_classifier.pkl`

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

1. Upload an image (JPG, JPEG, or PNG format)
2. Click on the uploaded image to see the classification result
3. View the prediction (Cat or Dog) with confidence score
4. See probability breakdown for both classes

## Model Details

- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: RBF (Radial Basis Function)
- **Feature Extraction**: Raw pixel values with PCA (100 components)
- **Preprocessing**: Image resizing to 64x64, RGB normalization
- **Training**: Balanced dataset with 500 samples per class

## Performance

The model achieves competitive accuracy on the test set. Detailed performance metrics are displayed during training.

## Dependencies

- streamlit==1.28.0
- opencv-python==4.8.1.78
- numpy==1.24.3
- scikit-learn==1.3.0
- pillow==10.0.0

## Dataset

The model is trained on the Asirra cat vs dog dataset. Update the dataset path in `train_model.py` to point to your local dataset location.

## Deployment

This application is designed for easy deployment on Streamlit Cloud:

1. Push the code to GitHub
2. Connect your repository to Streamlit Cloud
3. Deploy with the provided requirements.txt

## License

This project is created for educational purposes.

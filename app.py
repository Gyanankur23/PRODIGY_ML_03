import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import pickle

from preprocessing import DataPreprocessor
from classifier import SVMClassifier


st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

st.title("Cat vs Dog Image Classifier")
st.markdown("Upload an image to classify it as a cat or dog using SVM")

MODEL_PATH = Path(__file__).parent / "models" / "svm_cat_dog_classifier.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please train the model first using train_model.py")
        return None
    
    classifier = SVMClassifier()
    classifier.load_model(MODEL_PATH)
    return classifier

def preprocess_image(uploaded_file):
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("Failed to decode image. Please upload a valid image file.")
            return None
        
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        return img.flatten().reshape(1, -1)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def main():
    classifier = load_model()
    
    if classifier is None:
        st.warning("Model not available. Please run train_model.py first.")
        return
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.write("Classification Result:")
            
            with st.spinner("Processing..."):
                processed_img = preprocess_image(uploaded_file)
                
                if processed_img is not None:
                    prediction = classifier.predict(processed_img)[0]
                    probabilities = classifier.predict_proba(processed_img)[0]
                    
                    label = "Cat" if prediction == 0 else "Dog"
                    confidence = probabilities[prediction] * 100
                    
                    st.success(f"Prediction: {label}")
                    st.info(f"Confidence: {confidence:.2f}%")
                    
                    st.progress(confidence / 100)
                    
                    st.write("\nProbabilities:")
                    st.write(f"Cat: {probabilities[0] * 100:.2f}%")
                    st.write(f"Dog: {probabilities[1] * 100:.2f}%")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application uses a Support Vector Machine (SVM) classifier to distinguish between cat and dog images.
    
    **Model Details:**
    - Algorithm: SVM with RBF kernel
    - Features: Raw pixel values with PCA dimensionality reduction
    - Training: Balanced dataset of cat and dog images
    """)

if __name__ == "__main__":
    main()

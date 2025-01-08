import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import datetime

# Load the trained model
model = load_model(r'C:\project\models.keras')

# Define class labels
class_labels = ['Beginning Level', 'Early Level', 'Pre Level', 'Pro Level']

# Function to preprocess the image
def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = preprocess_input(resized_image.astype(np.float32))
    return normalized_image

# Streamlit app setup
st.title("Image Classification with MobileNetV2")
st.write("Upload an image for classification")

# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and preprocess the uploaded image
    img = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img, -1)
    
    # Preprocess the image for prediction
    processed_image = preprocess_image(img)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

    # Measure prediction time
    start = datetime.datetime.now()

    # Predict the class probabilities
    classes = model.predict(processed_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(classes)
    predicted_class_label = class_labels[predicted_class_index]

    # Calculate prediction time
    finish = datetime.datetime.now()
    elapsed = finish - start

    # Display the results
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write(f"Prediction: {predicted_class_label}")
    st.write(f"Prediction time: {elapsed}")


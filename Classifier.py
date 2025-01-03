import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(image_path):
    # Read and resize image
    original_image = cv2.imread(image_path)
    resized_image = cv2.resize(original_image, (224, 224))
    
    # Normalize the image using MobileNetV2's preprocess_input function
    segmented_image = preprocess_input(resized_image.astype(np.float32))  # Normalize for MobileNetV2
    return segmented_image

# Load the trained model
Classifier = load_model(r'C:\project\models.keras')

# Path to the test image
test_image_path = r'C:\project\dataset\Original\Pre\WBC-Malignant-Pre-020.jpg'

# Preprocess the image
segmented_image = preprocess_image(test_image_path)

# Save the segmented image (optional)
segmented_image_path = os.path.join('path\\Test-images\\Segmented', 'segmented_' + os.path.basename(test_image_path))
os.makedirs(os.path.dirname(segmented_image_path), exist_ok=True)
cv2.imwrite(segmented_image_path, cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB) * 255)  # Convert to RGB before saving

# Prepare the image for prediction by adding an extra dimension (batch size of 1)
Test_seg = np.expand_dims(segmented_image, axis=0)

# Measure prediction time
start = datetime.datetime.now()

# Predict the class probabilities
classes = Classifier.predict(Test_seg)

# Get the predicted class index
predicted_class_index = np.argmax(classes)

# Define class labels
class_labels = ['Beginning Level', 'Early Level', 'Pre Level', 'Pro Level']
predicted_class_label = class_labels[predicted_class_index]

# Print the predicted class index and label
print('Predicted index:', predicted_class_index)
print('Predicted Level:', predicted_class_label)

# Calculate and print the time taken for the prediction
finish = datetime.datetime.now()
elapsed = finish - start
print('________________')
print('Total time elapsed: ', elapsed)

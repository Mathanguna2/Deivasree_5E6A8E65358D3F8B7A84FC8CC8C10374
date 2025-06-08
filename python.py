import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained waste classification model
waste_model = load_model('waste_classification_model.h5')

# Define classes based on training
classes = ['plastic', 'glass', 'metal', 'paper']

# Load Haar cascade for object detection (using face detection cascade as a placeholder)
haar_file = 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)

# Initialize webcam
webcam = cv2.VideoCapture(0)

while True:
    # Capture frame
    ret, frame = webcam.read()
    if not ret:
        break

    # Convert to grayscale for object detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_objects = cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected_objects:
        # Draw rectangle around detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Crop and preprocess image for CNN model
        roi = frame[y:y + h, x:x + w]
        roi = cv2.resize(roi, (224, 224))
        roi = img_to_array(roi) / 255.0  # Normalize
        roi = np.expand_dims(roi, axis=0)

        # Predict waste type
        predictions = waste_model.predict(roi)
        class_index = np.argmax(predictions)
        label = classes[class_index]

        # Display label on screen
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show output
    cv2.imshow('Waste Classification', frame)

    # Exit on 'Esc' key
    if cv2.waitKey(10) == 27:
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()

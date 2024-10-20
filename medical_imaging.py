# -*- coding: utf-8 -*-
"""Medical_Imaging.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14aYOaGvRg0jzhsDUTLszTsrg0mjasT-u
"""

pip install tensorflow keras opencv-python matplotlib numpy

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the dataset
train_dir = 'C:/Users/Anushree S R/Downloads/Ai task 2/train'
test_dir = 'C:/Users/Anushree S R/Downloads/Ai task 2/test'

# Image size expected by the model
IMG_SIZE = 150

def preprocess_image(img_path):
    # Load the image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not load image at path: {img_path}")
        return None  # Or raise an exception

    # Resize to the required dimensions
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Normalize the image
    img = img / 255.0
    return img

# Example of loading and displaying an image
# Use os.path.join to create the correct file path
sample_image = preprocess_image(os.path.join(train_dir, 'NORMAL2-IM-1023-0001.jpeg'))

# Check if the image was preprocessed successfully
if sample_image is not None:
    plt.imshow(sample_image, cmap='gray')
    plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model
model = Sequential()

# First convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output of the convolutions
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer (Binary classification: Pneumonia or Not)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use a portion of training data for validation
)

# Load the images from the directories and apply augmentation
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='training'  # Training data
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Validation data
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary'
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot accuracy and loss curves
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()

# Load and preprocess a new image
new_image = preprocess_image('path_to_new_image.jpg')

# Add batch dimension and predict
new_image = np.expand_dims(new_image, axis=0)
prediction = model.predict(new_image)

# Interpret the result
if prediction[0][0] > 0.5:
    print("Pneumonia detected")
else:
    print("No Pneumonia detected")
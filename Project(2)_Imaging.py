# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 12:32:28 2025

@author: filot
"""

from tensorflow.keras.preprocessing import image
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (500, 500)
TEST_IMAGES = [
    ("test/crack/test_crack.jpg", "crack"),
    ("test/missing-head/test_missinghead.jpg", "missing-head"),
    ("test/paint-off/test_paintoff.jpg", "paint-off"),]

model_1 = keras.models.load_model("model_tf.keras")
model_2 = keras.models.load_model("model_tf2.keras")

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE, color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# By building the clas index maping

datagen = ImageDataGenerator(rescale=1./255)
temp_gen = datagen.flow_from_directory(
    "test", target_size=IMAGE_SIZE, color_mode="grayscale",
    class_mode="categorical", shuffle=False, batch_size=1
)
idx_to_name = {v: k for k, v in temp_gen.class_indices.items()}
del temp_gen

def show_predictions(model, model_name):
    for i, (img_path, true_label) in enumerate(TEST_IMAGES):
        x, img_display = preprocess_img(img_path)
        preds = model.predict(x, verbose=0)[0]
        pred_label = idx_to_name[np.argmax(preds)]

        plt.figure(figsize=(6, 6))
        plt.imshow(img_display, cmap="gray")
        plt.axis("off")
        
# Predicted & true label titles

        color = "green" if pred_label == true_label else "red"
        plt.title(f"{model_name}\nP: {pred_label} | T: {true_label}", color=color, fontsize=20, pad=20)
    
# Adding class probabilities

        y_text = 500  
        for j, prob in enumerate(preds):
            class_name = idx_to_name[j]
            conf = prob * 100
            plt.text(
                5, y_text, f"{class_name}: {conf:.1f}%", 
                color="lime" if j == np.argmax(preds) else "white",
                fontsize=20, ha="left", va="bottom",
                bbox=dict(facecolor="black", alpha=0.5, pad=1)
            )
            y_text -= 40

        plt.tight_layout()
        plt.show()
        
# Displaying the images as their own figure
show_predictions(model_1, "First CNN")
show_predictions(model_2, "Second CNN")
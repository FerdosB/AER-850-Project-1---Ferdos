# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 18:15:46 2025

@author: filot
"""

import numpy as np
import tensorflow as tf



from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


np.random.seed(42)
keras.utils.set_random_seed(42)
tf.random.set_seed(42)

# Data set

TRAIN_DIR = "train"
VAL_DIR = "valid"
TEST_DIR = "test"

# Step 1 Data processing

IMAGE_SIZE = (500, 500)
BATCH_SIZE = 6
EPOCHS = 14

# Train data sets

train_datagen = ImageDataGenerator(
    rescale=1./255.0,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=False
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=False
)

# Step 2 Neural Network Architecture Design

# Building a simple NN (1)
from tensorflow.keras import layers
#%% NN model 1
model_tf = keras.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(500,500,1)),
    layers.MaxPooling2D(2,2),


    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax")
])

#%% NN model 2
model_tf2 = keras.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(500,500,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    
    layers.Dense(3, activation="softmax")
])

#%%
# Step 3: Hyperparameter Analysis
# model 1
model_tf.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
    )


early_stop_1st = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
    )

historytf = model_tf.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop_1st],
    verbose=1
)

# model 2
model_tf2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
    )


early_stop_2nd = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
    )

historytf2= model_tf2.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop_2nd],
    verbose=1
)

# %%
# Step 4: Model Evaluation
# model_1
import matplotlib.pyplot as plt


test_loss, test_acc = model_tf.evaluate(test_generator, verbose = 0)
print(f"[Model 1] Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

model_tf.save("model_tf.keras")

hist= historytf.history
train_acc = hist.get("accuracy", hist.get("categorical_acccuracy"))
val_acc   = hist.get("val_accuracy", hist.get("val_categorical_accuracy"))

plt.figure(figsize=(6,4))
plt.plot(train_acc, label="Train Acc (Model 1)")
plt.plot(val_acc,   label="Val Acc (Model 1)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Model 1 Accuracy vs Epoch")
plt.legend();
plt.grid(True); 
plt.show()

plt.figure(figsize=(6,4))
plt.plot(hist["loss"],    label="Train Loss (Model 1)")
plt.plot(hist["val_loss"], label="Val Loss (Model 1)")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch of Model 1")
plt.legend(); 
plt.grid(True); 
plt.show()

# %%
# madel 2
Second_test_loss, Second_test_acc = model_tf2.evaluate(test_generator, verbose=0)
print(f"[Model 2] Test accuracy: {Second_test_acc:.3f} | Test loss: {Second_test_loss:.3f}")

model_tf2.save("model_tf2.keras")

# Plot Accuracy and Loss (model_tf2)
hist2 = historytf2.history
Second_train_acc = hist2.get("accuracy", hist2.get("categorical_accuracy"))
Second_val_acc   = hist2.get("val_accuracy", hist2.get("val_categorical_accuracy"))

plt.figure(figsize=(6,4))
plt.plot(Second_train_acc, label="Train Acc (Model 2)")
plt.plot(Second_val_acc,   label="Val Acc (Model 2)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Model 2 Accuracy vs Epoch")
plt.legend();
plt.grid(True); 
plt.show()

plt.figure(figsize=(6,4))
plt.plot(hist2["loss"],     label="Train Loss (Model 2)")
plt.plot(hist2["val_loss"], label="Val Loss (Model 2)")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch of Model 2")
plt.legend(); 
plt.grid(True); 
plt.show()
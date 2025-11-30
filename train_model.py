import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19

# ==========================
# Dataset Paths
# ==========================
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

# ==========================
# Model Architecture (same as app.py)
# ==========================
base_model = VGG19(include_top=False, input_shape=(128, 128, 3), weights="imagenet")

x = base_model.output
flat = Flatten()(x)
dense1 = Dense(4608, activation='relu')(flat)
dropout = Dropout(0.2)(dense1)
dense2 = Dense(1152, activation='relu')(dropout)
output = Dense(2, activation='softmax')(dense2)

model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================
# Image Data Generators
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=8,
    zoom_range=0.10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# ==========================
# Training
# ==========================
history = model.fit(
    train_data,
    epochs=15,
    validation_data=val_data
)

# ==========================
# Save Weights (MATCHES app.py)
# ==========================
os.makedirs("model_weights", exist_ok=True)
model.save_weights("model_weights/vgg19_model_01.weights.h5")

print("\nWeights saved to model_weights/vgg19_model_01.weights.h5")

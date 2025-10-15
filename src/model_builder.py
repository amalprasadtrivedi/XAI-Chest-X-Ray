# src/model_builder.py
"""
Model Builder Module for Chest X-Ray Classification (NORMAL vs PNEUMONIA)

This script provides:
1. A simple custom CNN model.
2. A Transfer Learning model (default: VGG16, can be swapped with ResNet50, MobileNet, etc.).
3. Utility to compile models with common settings.

Author: Amal Prasad Trivedi
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.applications import VGG16


# ------------------------------
# Model Compilation Utility
# ------------------------------
def compile_model(model, learning_rate=0.0001):
    """
    Compiles a given model with Adam optimizer, binary crossentropy loss.

    Args:
        model (tf.keras.Model): The model to compile.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        tf.keras.Model: Compiled model.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ------------------------------
# Simple CNN Model
# ------------------------------
def build_cnn_model(input_shape=(224, 224, 3)):
    """
    Builds a custom CNN model from scratch.

    Args:
        input_shape (tuple): Shape of input images.

    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    model = Sequential([
        # Convolution + Pooling Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Convolution + Pooling Block 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Convolution + Pooling Block 3
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        # Fully Connected Layers
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (Normal vs Pneumonia)
    ])

    return compile_model(model)


# ------------------------------
# Transfer Learning Model (VGG16)
# ------------------------------
def build_transfer_model(input_shape=(224, 224, 3)):
    """
    Builds a Transfer Learning model using VGG16 as base.

    Args:
        input_shape (tuple): Shape of input images.

    Returns:
        tf.keras.Model: Compiled Transfer Learning model.
    """

    # Load VGG16 without top layers (pre-trained on ImageNet)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze base model layers so they are not retrained
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)   # Converts feature maps into a single vector
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Binary classification

    model = Model(inputs=base_model.input, outputs=predictions)

    return compile_model(model)


# ------------------------------
# Debug Run (Optional)
# ------------------------------
if __name__ == "__main__":
    # Test CNN
    cnn_model = build_cnn_model()
    cnn_model.summary()

    # Test Transfer Learning
    tl_model = build_transfer_model()
    tl_model.summary()

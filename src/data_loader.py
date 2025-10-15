# src/data_loader.py
"""
Data Loader Module for Chest X-Ray Classification (NORMAL vs PNEUMONIA)

This script handles:
1. Data augmentation for training.
2. Data generators for train, validation, and test sets.
3. A helper function to load a single image for prediction.

Author: Amal Prasad Trivedi
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# ------------------------------
# CONFIGURATION
# ------------------------------
IMG_SIZE = (224, 224)   # Default size (can be changed depending on model)
BATCH_SIZE = 16


def get_data_generators(dataset_dir: str, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Prepares train, validation, and test data generators.

    Args:
        dataset_dir (str): Path to dataset directory containing train/, val/, test/.
        img_size (tuple): Target image size for resizing (default 224x224).
        batch_size (int): Number of images per batch.

    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """

    # ------------------------------
    # Data Augmentation (only for training set)
    # ------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,      # Normalize pixel values
        rotation_range=20,    # Random rotations
        width_shift_range=0.2, # Horizontal shifts
        height_shift_range=0.2, # Vertical shifts
        shear_range=0.2,      # Shear transformation
        zoom_range=0.2,       # Random zoom
        horizontal_flip=True, # Random flips
        fill_mode='nearest'   # Fill empty pixels
    )

    # Validation/Test should not use augmentation, only rescaling
    test_val_datagen = ImageDataGenerator(rescale=1.0/255)

    # ------------------------------
    # Data Generators
    # ------------------------------
    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    val_generator = test_val_datagen.flow_from_directory(
        os.path.join(dataset_dir, "val"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    test_generator = test_val_datagen.flow_from_directory(
        os.path.join(dataset_dir, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def load_single_image(img_path: str, img_size=IMG_SIZE):
    """
    Loads and preprocesses a single image for prediction.

    Args:
        img_path (str): Path to the image file.
        img_size (tuple): Target image size.

    Returns:
        np.ndarray: Preprocessed image ready for model prediction.
    """

    # Load image and resize
    img = load_img(img_path, target_size=img_size)

    # Convert to numpy array
    img_array = img_to_array(img)

    # Normalize pixel values (0-1 range)
    img_array = img_array / 255.0

    # Add batch dimension (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ------------------------------
# Debug Run (Optional)
# ------------------------------
if __name__ == "__main__":
    # Example usage (only when running this file directly)
    dataset_path = "../dataset"  # adjust if needed
    train_gen, val_gen, test_gen = get_data_generators(dataset_path)

    print("Train classes:", train_gen.class_indices)
    print("Validation samples:", val_gen.samples)
    print("Test samples:", test_gen.samples)

# src/train.py
"""
Training Script for Chest X-Ray Classification (NORMAL vs PNEUMONIA)

This script:
1. Loads training, validation, and test datasets.
2. Builds a CNN or Transfer Learning model.
3. Trains the model and evaluates it.
4. Saves the trained model in ../models/.
5. Saves training history plots (accuracy/loss).

Author: Amal Prasad Trivedi
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from src.data_loader import get_data_generators
from src.model_builder import build_cnn_model, build_transfer_model


# ------------------------------
# CONFIGURATION
# ------------------------------
DATASET_PATH = "../dataset"          # Dataset directory
MODEL_SAVE_PATH = "../models"        # Folder to save models
EPOCHS = 10                          # Change based on need
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
MODEL_TYPE = "transfer"              # Options: "cnn" or "transfer"


# ------------------------------
# Plot Training History
# ------------------------------
def plot_history(history, model_name="model"):
    """
    Plots training & validation accuracy and loss.

    Args:
        history (History): Keras training history object.
        model_name (str): Name for saving plots.
    """
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save plots
    os.makedirs("../results", exist_ok=True)
    plt.savefig(f"../results/{model_name}_training_curves.png")
    plt.show()


# ------------------------------
# Main Training Function
# ------------------------------
def train_model():
    """
    Loads data, builds model, trains and evaluates.
    Saves trained model in ../models/
    """

    print("[INFO] Loading data...")
    train_gen, val_gen, test_gen = get_data_generators(DATASET_PATH, IMG_SIZE, BATCH_SIZE)

    # ------------------------------
    # Build Model
    # ------------------------------
    if MODEL_TYPE == "cnn":
        print("[INFO] Building CNN model...")
        model = build_cnn_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        model_name = "cnn_model"
    else:
        print("[INFO] Building Transfer Learning model (VGG16)...")
        model = build_transfer_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        model_name = "transfer_model"

    model.summary()

    # ------------------------------
    # Training
    # ------------------------------
    print("[INFO] Training model...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    # ------------------------------
    # Save Model
    # ------------------------------
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model_save_file = os.path.join(MODEL_SAVE_PATH, f"{model_name}.h5")
    model.save(model_save_file)
    print(f"[INFO] Model saved at {model_save_file}")

    # ------------------------------
    # Plot History
    # ------------------------------
    plot_history(history, model_name)

    # ------------------------------
    # Evaluation
    # ------------------------------
    print("[INFO] Evaluating on test set...")
    loss, acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {acc*100:.2f}% | Test Loss: {loss:.4f}")


# ------------------------------
# Run Script
# ------------------------------
if __name__ == "__main__":
    train_model()

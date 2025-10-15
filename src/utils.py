# src/utils.py
"""
Utility functions for Chest X-Ray Classification Project

This script provides:
1. File and directory helpers
2. Model saving/loading utilities
3. Plotting functions for metrics
4. Logging utility
5. Reproducibility (seed setting)
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging


# ------------------------------
# 1. Reproducibility
# ------------------------------
def set_seed(seed=42):
    """
    Sets random seed for reproducibility.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ------------------------------
# 2. Directory Management
# ------------------------------
def create_dir(path):
    """
    Creates a directory if it does not exist.

    Args:
        path (str): Path to directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)


# ------------------------------
# 3. Model Saving / Loading
# ------------------------------
def save_model(model, path):
    """
    Saves a trained model to disk.

    Args:
        model (tf.keras.Model): Trained model.
        path (str): Destination file path (e.g., 'models/cnn_model.h5').
    """
    create_dir(os.path.dirname(path))
    model.save(path)
    print(f"[INFO] Model saved at {path}")


def load_trained_model(path):
    """
    Loads a trained Keras model from disk.

    Args:
        path (str): Path to model file (.h5).

    Returns:
        tf.keras.Model: Loaded model.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Model file not found: {path}")
    model = load_model(path)
    print(f"[INFO] Loaded model from {path}")
    return model


# ------------------------------
# 4. Plotting Utilities
# ------------------------------
def plot_training_history(history, save_path=None):
    """
    Plots training & validation accuracy and loss.

    Args:
        history: Keras History object from model.fit().
        save_path (str): Path to save the plot (optional).
    """
    acc = history.history['accuracy']
    val_acc = history.history.get('val_accuracy')
    loss = history.history['loss']
    val_loss = history.history.get('val_loss')
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Acc')
    if val_acc:
        plt.plot(epochs, val_acc, 'r-', label='Validation Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    if save_path:
        create_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
        print(f"[INFO] Training history plot saved at {save_path}")

    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plots confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix.
        class_names (list): List of class names.
        save_path (str): Path to save the plot (optional).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Tick marks
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        create_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
        print(f"[INFO] Confusion matrix plot saved at {save_path}")

    plt.show()


# ------------------------------
# 5. Logging Utility
# ------------------------------
def get_logger(log_file="logs/training.log"):
    """
    Returns a logger object for logging.

    Args:
        log_file (str): File to store logs.

    Returns:
        logging.Logger: Configured logger.
    """
    create_dir(os.path.dirname(log_file))

    logger = logging.getLogger("ChestXRayLogger")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ------------------------------
# Debug Test
# ------------------------------
if __name__ == "__main__":
    set_seed(42)

    # Create and log
    logger = get_logger()
    logger.info("Testing utils.py functionality...")

    # Dummy plot
    history = type('History', (), {})()
    history.history = {
        "accuracy": [0.7, 0.8, 0.9],
        "val_accuracy": [0.65, 0.78, 0.85],
        "loss": [0.6, 0.4, 0.2],
        "val_loss": [0.7, 0.5, 0.3],
    }
    plot_training_history(history)

    # Dummy confusion matrix
    cm = np.array([[50, 10], [8, 60]])
    plot_confusion_matrix(cm, ["NORMAL", "PNEUMONIA"])

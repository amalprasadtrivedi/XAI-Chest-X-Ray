# src/evaluate.py
"""
Evaluation Script for Chest X-Ray Classification (NORMAL vs PNEUMONIA)

This script:
1. Loads a trained model from ../models/.
2. Evaluates on the test set.
3. Generates confusion matrix & classification report.
4. Saves evaluation results in ../results/.

Author: Amal Prasad Trivedi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from src.data_loader import get_data_generators


# ------------------------------
# CONFIGURATION
# ------------------------------
DATASET_PATH = "../dataset"
MODEL_PATH = "../models/transfer_model.h5"   # change to cnn_model.h5 if needed
RESULTS_DIR = "../results"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16


# ------------------------------
# Plot Confusion Matrix
# ------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, model_name="model"):
    """
    Plots and saves confusion matrix.

    Args:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
        class_names (list): Class names (e.g., ["NORMAL", "PNEUMONIA"]).
        model_name (str): For saving the plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png"))
    plt.show()


# ------------------------------
# Evaluate Model
# ------------------------------
def evaluate_model():
    """
    Loads trained model and evaluates on test set.
    Saves results in ../results/
    """

    # Load test data
    print("[INFO] Loading data...")
    _, _, test_gen = get_data_generators(DATASET_PATH, IMG_SIZE, BATCH_SIZE)

    # Load model
    print(f"[INFO] Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)

    # Evaluate on test set
    print("[INFO] Evaluating model on test set...")
    loss, acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {acc*100:.2f}% | Test Loss: {loss:.4f}")

    # Predictions
    print("[INFO] Generating predictions...")
    y_pred_probs = model.predict(test_gen)
    y_pred = (y_pred_probs > 0.5).astype("int32").flatten()
    y_true = test_gen.classes

    # Class names
    class_names = list(test_gen.class_indices.keys())

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report)

    # Save report to file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=====================\n")
        f.write(report)
    print(f"[INFO] Report saved at {report_path}")

    # Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, class_names,
                          model_name=os.path.basename(MODEL_PATH).replace(".h5", ""))


# ------------------------------
# Run Script
# ------------------------------
if __name__ == "__main__":
    evaluate_model()

# src/explainability.py
"""
Explainability Module for Chest X-Ray Classification (NORMAL vs PNEUMONIA)

This script provides:
1. Grad-CAM visualization for CNN and Transfer Learning models.
2. SHAP analysis for feature importance.
3. Utilities to integrate with Streamlit frontend.

Author: Amal Prasad Trivedi
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ------------------------------
# Grad-CAM Implementation
# ------------------------------
def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Computes Grad-CAM heatmap for a given image and model.

    Args:
        model (tf.keras.Model): Trained model.
        img_array (np.ndarray): Preprocessed image (1, H, W, C).
        last_conv_layer_name (str): Name of last convolutional layer.
        pred_index (int): Class index for which Grad-CAM is computed.

    Returns:
        np.ndarray: Heatmap (0-1 scale).
    """
    # Get the last conv layer
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Gradient calculation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Get gradients
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling (importance of each filter)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply channels by importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap.numpy()


def overlay_gradcam(img_path, heatmap, alpha=0.4):
    """
    Overlays Grad-CAM heatmap on original image.

    Args:
        img_path (str): Path to original image.
        heatmap (np.ndarray): Grad-CAM heatmap.
        alpha (float): Transparency factor.

    Returns:
        np.ndarray: Image with heatmap overlay.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img


# ------------------------------
# SHAP Analysis
# ------------------------------
def shap_explain(model, img_array, background=None, nsamples=5):
    """
    Performs SHAP explanation for a given image.

    Args:
        model (tf.keras.Model): Trained model.
        img_array (np.ndarray): Preprocessed image (1, H, W, C).
        background (np.ndarray): Background data for SHAP (default: repeat image).
        nsamples (int): Number of background samples.

    Returns:
        shap_values: SHAP values for image.
    """
    if background is None:
        background = np.repeat(img_array, nsamples, axis=0)

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(img_array)

    return shap_values


def plot_shap_explanation(shap_values, img_array):
    """
    Plots SHAP explanation.

    Args:
        shap_values: SHAP values from shap_explain().
        img_array (np.ndarray): Original preprocessed image.
    """
    plt.figure(figsize=(6, 6))
    shap.image_plot(shap_values, img_array, show=False)
    plt.title("SHAP Explanation")
    plt.show()


# ------------------------------
# Utility: Load and Preprocess Single Image
# ------------------------------
def preprocess_image(img_path, target_size=(224, 224)):
    """
    Loads and preprocesses single image.

    Args:
        img_path (str): Path to image file.
        target_size (tuple): Resize dimensions.

    Returns:
        np.ndarray: Preprocessed image array.
    """
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ------------------------------
# Debug Run (Optional)
# ------------------------------
if __name__ == "__main__":
    # Example for testing
    from tensorflow.keras.models import load_model

    model = load_model("../models/transfer_model.h5")
    img_path = "../dataset/test/NORMAL/NORMAL2.jpg"  # adjust path

    # Grad-CAM
    img_array = preprocess_image(img_path)
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name="block5_conv3")  # for VGG16
    overlay_img = overlay_gradcam(img_path, heatmap)
    cv2.imwrite("../results/gradcam_example.jpg", overlay_img)

    # SHAP
    shap_values = shap_explain(model, img_array)
    plot_shap_explanation(shap_values, img_array)

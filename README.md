# 🩻 Explainable AI for Chest X-Ray Diagnosis

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-green)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

---

## 📌 Project Overview

This project is a **B.Tech Final Year Major Project** developed by **Amal Prasad Trivedi**.  
It leverages **Deep Learning (CNN & Transfer Learning)** and **Explainable AI (XAI)** to classify **Chest X-Rays** into two categories:

- ✅ **Normal**
- ❌ **Pneumonia**

The system not only predicts the class but also provides **visual explanations** using **Grad-CAM** and **SHAP** to highlight important regions in the X-ray images.  

This helps **doctors and healthcare professionals** understand the model's decision, enhancing trust and interpretability.

---

## 🎯 Objectives

- Build a **robust and accurate CNN model** to detect pneumonia.
- Apply **Transfer Learning** to improve performance with limited datasets.
- Integrate **Explainable AI (XAI)** methods (Grad-CAM & SHAP) to visualize feature importance.
- Provide a **user-friendly interface** using Streamlit:
  - Upload and predict X-ray images.
  - Visualize model explainability.
  - Explore performance metrics and ROC curves.
- Create a project suitable for **healthcare decision support systems**.

---

## 🗂 Dataset Description

```

xai_chest_xray/
│
├── dataset/
│ ├── train/
│ │ ├── NORMAL/
│ │ ├── PNEUMONIA/
│ ├── val/
│ │ ├── NORMAL/
│ │ ├── PNEUMONIA/
│ └── test/
│ ├── NORMAL/
│ ├── PNEUMONIA/

```



- **Training Data:** 160 images (80 Normal, 80 Pneumonia)  
- **Validation Data:** 16 images (8 Normal, 8 Pneumonia)  
- **Test Data:** 40 images (20 Normal, 20 Pneumonia)  

> Images are preprocessed (resized and normalized) before training. Augmentation is applied to avoid overfitting.

---

## ⚙️ Tech Stack

| Category            | Tools / Libraries                          |
|--------------------|-------------------------------------------|
| Programming Language| Python 3.10+                               |
| Deep Learning       | TensorFlow, Keras                          |
| Data Handling       | NumPy, Pandas                              |
| Visualization       | Matplotlib, Seaborn                         |
| Explainable AI      | Grad-CAM, SHAP                              |
| Frontend            | Streamlit                                   |
| Model Saving        | Pickle, Joblib                              |

---

## 🧪 Methodology

1. **Data Preprocessing**
   - Resize X-ray images.
   - Normalize pixel values to [0,1].
   - Augment dataset to reduce overfitting.

2. **Model Training**
   - Train CNN from scratch.
   - Use Transfer Learning (VGG16, ResNet50) for better performance.

3. **Explainability**
   - Generate **Grad-CAM heatmaps** to highlight important regions.
   - Use **SHAP** to visualize pixel-level contributions.

4. **Deployment**
   - Streamlit application with sections:
     - **Home** – Project introduction
     - **Predict** – Upload & classify X-ray images
     - **Explainability** – Grad-CAM & SHAP visualization
     - **Performance** – Metrics, confusion matrix, ROC curves
     - **About** – Project description & credits

---

## 📊 Model Performance

- **Accuracy:** 90%  
- **Precision:** 92%  
- **Recall:** 88%  
- **F1 Score:** 0.90  
- **AUC:** 0.956  

- Confusion matrix demonstrates high True Positive & True Negative detection.
- ROC Curve indicates excellent model discrimination between Normal and Pneumonia.

---

## 🧠 Explainability Results

- **Grad-CAM:** Highlights regions of the X-ray image that influenced the model’s decision.
- **SHAP:** Shows pixel-level feature importance and contribution to the prediction.
- Enhances **trust, transparency, and interpretability** for medical professionals.

---

## 🚀 Future Scope

- Expand dataset with more diverse chest X-ray images.
- Extend to **multi-class classification** (e.g., COVID-19, Tuberculosis).
- Integrate with hospital **PACS systems** for real-time usage.
- Deploy cloud-based application for remote healthcare access.
- Implement **federated learning** to preserve data privacy.

---

## 🙋 Author & Contribution

**👨‍💻 Amal Prasad Trivedi**  
🎓 B.Tech 4th Year, Computer Science (AI & ML)  

**Contributions:**
- Dataset preprocessing & augmentation
- CNN & Transfer Learning model development
- Explainable AI integration (Grad-CAM & SHAP)
- Model evaluation, visualization & performance dashboard
- Streamlit frontend development

---

## 🔗 Links & Portfolio

- GitHub: [https://github.com/amalprasadtrivedi](https://github.com/amalprasadtrivedi)  
- Portfolio: [https://amal-prasad-trivedi-portfolio.vercel.app/](https://amal-prasad-trivedi-portfolio.vercel.app/)  
- LinkedIn: [https://www.linkedin.com/in/amal-prasad-trivedi-b47718271/](https://www.linkedin.com/in/amalprasadtrivedi-aiml-engineer/)

---

## ⚠️ Disclaimer

💡 This system is **a prototype built for academic purposes** and **should not be used as a substitute for professional medical diagnosis**. Always consult healthcare professionals for medical decisions.

---



The dataset used in this project contains chest X-ray images divided into **Normal** and **Pneumonia** categories.


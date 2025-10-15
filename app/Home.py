# app/Home.py
"""
🏥 Home Page - Explainable AI for Chest X-Ray Classification

This page introduces the Explainable AI (XAI) Chest X-Ray Analysis System.
It presents the motivation, disease overview, dataset, workflow, and usage guide.
A professional sidebar and icons enhance navigation and developer branding.
"""

# ------------------------------
# Imports
# ------------------------------
import streamlit as st
from pathlib import Path

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="🩻 Explainable AI - Chest X-Ray Classification",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("ℹ️ About the System")

st.sidebar.markdown("""
**🩻 Explainable AI for Chest X-Ray Analysis**  
A deep learning-based diagnostic support system that classifies X-rays into **Normal** or **Pneumonia** and explains its decisions using **XAI**.
""")

# External Links
st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Creator Info")
st.sidebar.markdown("""
**Amal Prasad Trivedi**  
🎓 B.Tech CSE (AI & ML)  
💡 Researcher | ML Developer  
""")
st.sidebar.markdown("### 🔍📬  Connect with Developer")
st.sidebar.link_button("🌐 Portfolio", "https://amalprasadtrivediportfolio.vercel.app/")
st.sidebar.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/amalprasadtrivedi-aiml-engineer")
st.sidebar.link_button("💻 GitHub", "https://github.com/amalprasadtrivedi")
st.sidebar.markdown("---")

st.sidebar.info("👨‍💻 Developed by **Amal Prasad Trivedi**\nB.Tech Final Year • Computer Science (AI & ML)")
st.sidebar.markdown("---")
st.sidebar.caption("⚙️ Developed as part of the AI-Based Radiology Assistant Project.")
st.sidebar.caption("⚙️ Built as part of Final Year Major Project - Explainable AI for Radiology")

# ------------------------------
# Title & Introduction
# ------------------------------
st.title("🧠 Explainable AI for Chest X-Ray Classification")
st.subheader("Harnessing Deep Learning and XAI for Medical Transparency")

st.markdown("""
Welcome to the **Explainable AI (XAI)** system for **Chest X-Ray Classification**.  
This system aids radiologists and healthcare professionals by identifying **Pneumonia** from **Normal** chest X-rays — with explainability and trust at its core.

It integrates:
- 🩺 **AI-based Diagnosis**
- 🔍 **Visual Explanations (Grad-CAM & SHAP)**
- 📊 **Transparent Predictions**
- ⚙️ **End-to-End Medical Workflow**
---
""")

# ------------------------------
# Page 1: Medical Background
# ------------------------------
st.header("🫁 Understanding Pneumonia")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("""
**Pneumonia** is an inflammatory condition of the lungs affecting the alveoli.  
It can be caused by bacteria, viruses, or fungi.

### 🧬 Symptoms:
- Persistent cough  
- Chest pain  
- Shortness of breath  
- Fever & chills  

### 📉 Global Impact:
- Affects over **400 million people annually**  
- Major cause of child mortality worldwide  

### 🩻 Medical Imaging Role:
Chest X-rays are critical in pneumonia detection, helping clinicians visualize lung opacity and fluid buildup.
""")

with col2:
    img_path = Path("assets/pneumonia_example.jpg")
    if img_path.exists():
        st.image(str(img_path), caption="Pneumonia X-Ray Example", use_container_width=True)
    else:
        st.info("🖼️ Pneumonia example image not found in assets/")

st.markdown("---")

# ------------------------------
# Page 2: System Overview
# ------------------------------
st.header("⚙️ System Overview")

st.markdown("""
Our **Explainable AI (XAI)** system combines **Convolutional Neural Networks (CNNs)** with **XAI techniques** to provide **both predictions and reasons** behind them.

### 🚀 Key Features:
- **Binary Classification** → Normal / Pneumonia  
- **Explainable Outputs** → Grad-CAM & SHAP  
- **User-Friendly Interface** → Streamlit Dashboard  
- **Model Transparency** → Feature importance & heatmaps  

This enhances **clinical trust** and **interpretability** of deep learning models in healthcare.
""")

system_img = Path("assets/system_overview.jpg")
if system_img.exists():
    st.image(str(system_img), caption="End-to-End System Overview", use_container_width=True)
else:
    st.info("🖼️ Add system_overview.png to assets/ for better visualization.")

st.markdown("---")

# ------------------------------
# Page 3: Dataset Description
# ------------------------------
st.header("📂 Dataset Information")

st.markdown("""
The model was trained on a publicly available **Chest X-Ray Dataset** containing two classes:

| Class | Description | Example Count |
|:------|:-------------|:--------------:|
| 🟢 **Normal** | Healthy lungs without infection | 80 Train / 8 Val / 20 Test |
| 🔴 **Pneumonia** | Infected lungs with opacity patterns | 80 Train / 8 Val / 20 Test |

The dataset was preprocessed by:
- Resizing all images to **224×224 pixels**
- Normalizing pixel values to **[0, 1]**
- Data Augmentation (rotation, zoom, flipping)

These steps ensure the model generalizes well to new patient scans.
""")

dataset_img = Path("assets/dataset_example.png")
if dataset_img.exists():
    st.image(str(dataset_img), caption="Sample Dataset Visualization", use_container_width=True)
else:
    st.info("🖼️ Add dataset_example.png to assets/ for better visualization.")

st.markdown("---")

# ------------------------------
# Page 4: Workflow & Architecture
# ------------------------------
st.header("🔄 System Workflow & Model Architecture")

st.markdown("""
The complete pipeline consists of the following **5 stages**:

1. 🧹 **Preprocessing** → Resize & Normalize input images.  
2. 🧠 **Model Training** → Use CNNs or Transfer Learning models (e.g., VGG16, ResNet).  
3. ⚖️ **Prediction** → Classify X-rays as *Normal* or *Pneumonia*.  
4. 🔥 **Explainability** → Grad-CAM & SHAP visualizations for transparency.  
5. 📊 **Evaluation** → Metrics such as accuracy, precision, recall, F1-score.

Below is a simplified diagram representing this process:
""")

workflow_img = Path("assets/workflow.png")
if workflow_img.exists():
    st.image(str(workflow_img), caption="System Workflow Diagram", use_container_width=True)
else:
    st.info("🖼️ Add workflow.png to assets/ to show pipeline steps visually.")

st.markdown("""
The **Explainable AI Component** plays a crucial role by visually demonstrating which lung regions influenced the decision — helping radiologists validate AI predictions.
""")

st.markdown("---")

# ------------------------------
# Page 5: Using the Application
# ------------------------------
st.header("🧭 How to Use the App")

st.markdown("""
Follow these simple steps to interact with the system:

1. **🏠 Home** → Understand the project and background (this page).  
2. **📤 Upload & Predict** → Upload a Chest X-Ray image to get the classification result.  
3. **🔍 Explainability** → View model reasoning via **Grad-CAM** & **SHAP** explanations.  
4. **📈 Results** → Analyze training metrics and confusion matrix visualization.  

💡 *Tip:* For best results, use high-quality grayscale X-rays with clear contrast.
""")

st.success("✅ Ready to begin? Use the sidebar to explore the app sections!")

st.markdown("---")

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
<div style="text-align: center; padding: 20px; font-size: 16px;">
    <b>🩻 Explainable AI Chest X-Ray Classification System</b><br>
    Developed by <b>Amal Prasad Trivedi</b><br>
    B.Tech – Computer Science (AI & ML)
</div>
""", unsafe_allow_html=True)

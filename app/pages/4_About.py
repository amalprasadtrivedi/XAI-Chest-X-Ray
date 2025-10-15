# ==============================================================
# 📘 app/pages/4_About.py
# Enhanced About Page for Explainable AI - Chest X-Ray Diagnosis
# ==============================================================

import streamlit as st
from PIL import Image

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="ℹ️ About Project",
    page_icon="📘",
    layout="wide"
)

# ==============================================================
# 🧭 SIDEBAR SECTION
# ==============================================================
with st.sidebar:
    st.image("assets/xray_logo.png", use_container_width=True)
    st.title("🧠 Explainable AI System")

    st.markdown("""
    ### 🩺 About
    This system leverages **Deep Learning** and **Explainable AI** to classify **Chest X-Rays** as  
    **Normal** or **Pneumonia**, while explaining how predictions are made.

    The project aims to bring **trust** and **interpretability** to medical AI systems.
    """)

    st.markdown("---")

    st.markdown("### 👨‍💻 Creator Info")
    st.markdown("""
        **Amal Prasad Trivedi**  
        🎓 B.Tech CSE (AI & ML)  
        💡 Researcher | ML Developer  
        """)

    # Buttons with external links
    st.markdown("#### 🌐 Connect with me:")
    st.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/amal-prasad-trivedi-b47718271/")
    st.link_button("💻 GitHub", "https://github.com/amalprasadtrivedi")
    st.link_button("🌍 Portfolio", "https://amal-prasad-trivedi-portfolio.vercel.app/")

    st.markdown("---")
    st.caption("⚙️ Developed as part of the AI-Based Radiology Assistant Project.")
    st.caption("⚙️ Built as part of Final Year Major Project - Explainable AI for Radiology")
    st.info("💡 Passionate about AI, Explainability, and Healthcare Innovation!")

# ==============================================================
# 🧩 PAGE 1: INTRODUCTION
# ==============================================================

st.title("📘 About the Project")
st.markdown("""
Welcome to the **Explainable AI for Chest X-Ray Diagnosis** system — an AI-driven medical imaging solution 
that helps radiologists identify **Pneumonia** from chest X-ray images using **Deep Learning** and **XAI (Explainable AI)** techniques.

---

### 🩻 Why Pneumonia Detection?
- **Pneumonia** is a severe lung infection that causes inflammation and fluid accumulation in the lungs.  
- It can be caused by **bacteria, viruses, or fungi**, and can be **life-threatening**, especially for infants and the elderly.  
- Early and accurate diagnosis significantly improves treatment outcomes.  

However, manual diagnosis through X-rays is **subjective** and **time-consuming**.  
Our system automates this process using a **Convolutional Neural Network (CNN)**, improving both **speed** and **accuracy**.
""")

st.markdown("---")

# ==============================================================
# 🎯 PAGE 2: PROJECT OBJECTIVES
# ==============================================================

st.header("🎯 Project Objectives")
st.markdown("""
The main aim of this project is to combine **AI accuracy** with **human interpretability**.

**Key Objectives:**
- 🧠 Build a robust **CNN model** for binary classification: *Normal vs Pneumonia*  
- 🔄 Utilize **Transfer Learning** (VGG16, ResNet50) for better generalization  
- 💬 Implement **Explainable AI (Grad-CAM, SHAP)** to interpret model decisions  
- 💻 Develop a **user-friendly Streamlit interface** for predictions and visualization  
- ⚕️ Deliver a **trustworthy AI system** to assist healthcare professionals
""")

st.markdown("---")

# ==============================================================
# 🧪 PAGE 3: METHODOLOGY & PIPELINE
# ==============================================================

st.header("🧪 Methodology & System Workflow")
st.markdown("""
The complete workflow follows a **4-stage pipeline**:

1. **Data Preprocessing**  
   - Resize and normalize X-ray images  
   - Apply augmentation (rotation, zoom, flip) to prevent overfitting  

2. **Model Training**  
   - Train CNN models on labeled datasets  
   - Evaluate using validation data to tune hyperparameters  

3. **Explainability Layer (XAI)**  
   - Apply **Grad-CAM** to visualize “what the model sees”  
   - Use **SHAP** to analyze which features influenced predictions  

4. **Deployment**  
   - Build a modular, multi-page **Streamlit app**  
   - Integrate prediction, explainability, and performance dashboards
""")

st.image("assets/system_overview.jpg", caption="System Workflow - Data to Explainability", use_container_width=True)

st.markdown("---")

# ==============================================================
# 🗂 PAGE 4: DATASET DETAILS
# ==============================================================

st.header("🗂 Dataset Description")

st.markdown("""
The dataset used for model training and testing contains **chest X-ray images** divided into:
- **NORMAL** (Healthy lungs)
- **PNEUMONIA** (Infected lungs)

**Dataset Structure:**
```plaintext
dataset/
├── train/
│   ├── NORMAL/
│   ├── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   ├── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   ├── PNEUMONIA/
```


### 📊 Dataset Summary
| Dataset Split | Normal Images | Pneumonia Images | Total |
|----------------|----------------|------------------|--------|
| Training       | 80             | 80               | 160    |
| Validation     | 8              | 8                | 16     |
| Testing        | 20             | 20               | 40     |

The dataset provides **balanced classes**, ensuring the model learns equally from both healthy and diseased cases.
""")

# -----------------------------
# 🫁 DISEASE DESCRIPTION
# -----------------------------
st.header("🫁 Understanding Pneumonia")
st.markdown("""
**Pneumonia** is a serious lung infection that inflames the air sacs (alveoli), often causing them to fill with fluid or pus.  
It can result from **bacterial, viral, or fungal** infections and may be life-threatening if untreated — especially for children, the elderly, and people with weakened immune systems.

### ⚠️ Symptoms:
- High fever  
- Cough with phlegm or pus  
- Shortness of breath  
- Chest pain during breathing or coughing  
- Fatigue and weakness  

### 🧬 Diagnosis:
Doctors typically use **chest X-rays** to detect patterns in lung tissue that indicate infection.  
This is where **AI-assisted systems** like ours play a crucial role — by analyzing X-ray patterns and aiding radiologists with **interpretable predictions**.

""")

# -----------------------------
# 🧠 SYSTEM WORKFLOW
# -----------------------------
st.header("🧠 System Workflow")
st.markdown("""
The entire pipeline of the **Explainable AI for Chest X-Ray Diagnosis** system can be summarized in five key stages:

1. **🩻 Image Input:**  
   User uploads a chest X-ray image.

2. **⚙️ Preprocessing:**  
   Image is resized, normalized, and converted into a suitable tensor format.

3. **🤖 Model Prediction:**  
   The trained CNN or Transfer Learning model classifies the image as **Normal** or **Pneumonia**.

4. **🔍 Explainability:**  
   - **Grad-CAM** highlights important regions influencing model decision.  
   - **SHAP** explains how individual pixels contribute to classification.

5. **📈 Visualization & Reporting:**  
   Users can view Grad-CAM overlays, SHAP plots, and performance metrics such as accuracy, ROC, and confusion matrix.
""")

st.image("assets/workflow.png", caption="System Workflow Overview", use_container_width=True)

# -----------------------------
# ⚙️ TECHNOLOGIES USED
# -----------------------------
st.header("⚙️ Technologies & Tools")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    - 🐍 **Python 3.10**  
    - 🧠 **TensorFlow / Keras** — Deep Learning Framework  
    - 🔢 **NumPy, Pandas** — Data Processing  
    - 📊 **Matplotlib, Seaborn** — Visualization  
    - 🧩 **scikit-learn** — Evaluation Metrics  
    """)

with col2:
    st.markdown("""
    - 🧮 **SHAP, Grad-CAM** — Explainability  
    - 🖥️ **Streamlit** — Web Frontend  
    - 💾 **Joblib / Pickle** — Model Serialization  
    - ☁️ **Streamlit Cloud / Local Deployment**  
    - 🧬 **OpenCV, PIL** — Image Handling  
    """)

# -----------------------------
# 🧪 METHODOLOGY
# -----------------------------
st.header("🧪 Methodology")
st.markdown("""
### Step 1: Data Preparation  
- X-ray images are resized to (224×224).  
- Pixel values are normalized (0–1).  
- Data augmentation (rotation, zoom, flip) enhances generalization.

### Step 2: Model Training  
- Base model: **Custom CNN / ResNet50 / VGG16**  
- Optimizer: **Adam**  
- Loss Function: **Binary Cross-Entropy**  
- Early Stopping used to prevent overfitting.

### Step 3: Explainability  
- **Grad-CAM** → Identifies key regions influencing the prediction.  
- **SHAP** → Explains how each pixel contributes to the model output.

### Step 4: Evaluation  
- Metrics: Accuracy, Precision, Recall, F1-Score  
- Visualizations: ROC Curve, Confusion Matrix

### Step 5: Deployment  
- Final model integrated into a **Streamlit interface**  
- Accessible to doctors, researchers, and students for demonstration.
""")

# -----------------------------
# 🧑‍💻 AUTHOR INFORMATION
# -----------------------------
st.header("🧑‍💻 Author & Contribution")
st.markdown("""
**👨‍🎓 Amal Prasad Trivedi**  
🎓 *B.Tech 4th Year, Computer Science (AI & Machine Learning)*  

### 🏗️ Contributions:
- Model design and training (CNN + Transfer Learning)  
- Grad-CAM and SHAP explainability modules  
- Frontend development using Streamlit  
- Performance analytics and visualization  
- Documentation and system deployment  
""")

# -----------------------------
# 🚀 FUTURE SCOPE
# -----------------------------
st.header("🚀 Future Enhancements")
st.markdown("""
- 🔍 Include more diseases (COVID-19, Tuberculosis, Lung Cancer).  
- ☁️ Deploy on cloud for hospital-scale integration.  
- 🏥 Integrate with **hospital PACS systems** for real-time scans.  
- 🧑‍⚕️ Include multi-doctor feedback for clinical validation.  
- 🔐 Add **federated learning** for privacy-preserving updates.  
""")

# -----------------------------
# 🙏 ACKNOWLEDGEMENTS
# -----------------------------
st.header("🙏 Acknowledgements")
st.markdown("""
- Grateful to **faculty mentors** for their guidance.  
- Thanks to **open-source communities** and **Kaggle** for dataset availability.  
- Appreciation to **Streamlit**, **TensorFlow**, and **SHAP** developers for empowering open innovation.  
""")

# -----------------------------
# ⚠️ FOOTER DISCLAIMER
# -----------------------------
st.markdown("---")
st.warning("""
**Disclaimer:**  
This Explainable AI system is designed **for educational and research purposes only**.  
It is **not a medical diagnostic tool** and should not replace professional medical advice.
""")

#  st.markdown("<center>🫁 Built with ❤️ and AI by Amal Prasad Trivedi</center>", unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; font-size: 16px;">
    <b>🩻 Explainable AI Chest X-Ray Classification System</b><br>
    Developed by <b>Amal Prasad Trivedi</b><br>
    B.Tech – Computer Science (AI & ML)
</div>
""", unsafe_allow_html=True)
# NeuroTab — Neurological Headache Classification System

> A web-based Clinical Decision Support System (CDSS) that uses a custom-trained TabNet deep learning model to classify neurological headache disorders, with explainable AI (XAI) insights for clinical interpretability.

## Live Demo
The application is hosted on HuggingFace Spaces:
**[https://huggingface.co/spaces/dewanjee/NeuroTab](https://huggingface.co/spaces/dewanjee/NeuroTab)**

## Overview

NeuroTab is a proof-of-concept clinical decision support tool developed as part of a research project on neurological disorder classification. It accepts patient symptom data as input and predicts the most likely headache/migraine diagnosis, along with confidence scores and XAI visualizations to explain the model's reasoning.

The system supports two diagnostic pathways:
- **Migraine Classification** — classifies into 7 migraine subtypes based on the International Headache Society (IHS) criteria.
- **Headache Classification** — differentiates between Migraine, Cluster headache, and Tension-type headache.

---

## Features

- **Dual Diagnostic Modules** — separate models and input forms for migraine subtype classification and general headache classification.
- **Confidence Scoring** — displays the model's prediction confidence as a percentage.
- **SHAP Explanations** — waterfall plots showing how each input feature contributed to the prediction.
- **LIME Explanations** — local surrogate model plots highlighting the most influential features.
- **Smart Cross-referral** — if a migraine prediction returns "Other", the system recommends switching to the headache classifier, and vice versa.
- **Non-technical UI** — plain-language explanations of XAI plots designed for clinical users.
- **Fully Responsive Web Interface** — works on desktop and mobile browsers.

---

## How It Works

### 1. Input Collection
The user fills in a clinical symptom form — including pain location, character, intensity, associated symptoms, and patient history. Each field maps directly to features used during model training.

### 2. Preprocessing
Input values are encoded using the same label mappings and `StandardScaler` used during training, ensuring consistent feature representation.

### 3. Prediction
The preprocessed input is passed through a custom-built **TabNet** classifier implemented in PyTorch. The model outputs class probabilities via softmax, and the highest probability class is returned as the prediction.

### 4. Explainability
Two XAI methods run in parallel on each prediction:
- **SHAP** (SHapley Additive exPlanations): uses a background dataset to compute each feature's marginal contribution.
- **LIME** (Local Interpretable Model-agnostic Explanations): fits a local linear model around the input to identify the most influential features.

The model was trained on clinical datasets of migraine and headache patients. Two separate models were trained:
- **Migraine model** — 17 input features, 7 output classes.
- **Headache model** — 19 input features, 3 output classes.

---

## Tech Stack

| Layer                | Technology                           |
|----------------------|---------------------------------------|
| **Backend**          | Python, Flask                         |
| **Deep Learning**    | PyTorch                               |
| **Explainability**   | SHAP, LIME                            |
| **Data Processing**  | NumPy, Pandas, scikit-learn           |
| **Visualization**    | Matplotlib                            |
| **Server**           | Gunicorn (WSGI)                       |
| **Frontend**         | HTML, CSS, JavaScript                 |
| **Deployment**       | HuggingFace Spaces (Docker)           |
| **Model Storage**    | HuggingFace Model Hub                 |

---

## User Flow

To visualize how the system works, here’s a detailed **User Flow Diagram** that describes the entire process:

![User Flow Diagram](Snapshots/User%20Flow%20Diagram.pdf)

---

## Image Gallery

### **Landing Page**

The landing page is designed for intuitive use, allowing users to easily navigate through the tool. Here’s a screenshot of the landing page interface:

![Landing Page](Snapshots/Landing%20Page/Landing%20Page.png)

---

### **Headache Type Classification**

This diagnostic module is responsible for classifying general headaches into **Migraine**, **Cluster headache**, and **Tension-type headache**. Here's a glimpse into the process:

1. **Input Form** – The user inputs the clinical symptoms for diagnosis.

   ![Input Form](Snapshots/Headache%20Type%20Classification/Input%20Form.png)

2. **Prediction on Sample Inputs** – Based on the input data, the model predicts the most likely headache type.

   ![Prediction on Sample Inputs](Snapshots/Headache%20Type%20Classification/Prediction%20on%20Sample%20Inputs.png)

3. **Recommendation** – If the model predicts a migraine, it recommends checking for migraine subtypes.

   ![Recommendation](Snapshots/Headache%20Type%20Classification/Recommendation.png)

4. **Explainability** – The model’s decision-making process is explained using LIME and SHAP plots.

   - **LIME plot**:
     ![XAI (LIME Plot)](Snapshots/Headache%20Type%20Classification/XAI%20%28LIME%20Plot%29.png)

   - **SHAP plot**:
     ![XAI (SHAP plot)](Snapshots/Headache%20Type%20Classification/XAI%20%28SHAP%20plot%29.png)

---

### **Migraine Variants Classification**

This diagnostic module classifies **migraine subtypes** based on input symptoms, helping distinguish between various types of migraines.

1. **Input Form** – The user provides migraine-specific symptom data.

   ![Input Form](Snapshots/Migraine%20Variants%20Classification/Input%20Form.png)

2. **Prediction on Sample Input** – The model predicts the most likely migraine subtype based on the inputs.

   ![Prediction on Sample Input](Snapshots/Migraine%20Variants%20Classification/Prediction%20on%20Sample%20Input.png)

3. **Explainability** – The model’s decision-making process is visualized using LIME and SHAP plots.

   - **LIME plot**:
     ![XAI (LIME Plot)](Snapshots/Migraine%20Variants%20Classification/XAI%20%28LIME%20Plot%29.png)

   - **SHAP plot**:
     ![XAI (SHAP plot)](Snapshots/Migraine%20Variants%20Classification/XAI%20%28SHAP%20plot%29.png)

---

## Disclaimer
NeuroTab is a **research prototype** and proof of concept. It is **not intended for clinical use** and should not be used as a substitute for professional medical diagnosis. All predictions are for research and demonstration purposes only.

---

## License
MIT License.

#  NeuroTab — Neurological Headache Classification System

> A web-based Clinical Decision Support System (CDSS) that uses a custom-trained TabNet deep learning model to classify neurological headache disorders, with explainable AI (XAI) insights for clinical interpretability.

## Live Demo
The application is hosted on HuggingFace Spaces:
**[https://huggingface.co/spaces/dewanjee/NeuroTab](https://huggingface.co/spaces/dewanjee/NeuroTab)**

##  Overview

NeuroTab is a proof-of-concept clinical decision support tool developed as part of a research project on neurological disorder classification. It accepts patient symptom data as input and predicts the most likely headache/migraine diagnosis, along with confidence scores and XAI visualizations to explain the model's reasoning.

The system supports two diagnostic pathways:
- **Migraine Classification** — classifies into 7 migraine subtypes based on the International Headache Society (IHS) criteria
- **Headache Classification** — differentiates between Migraine, Cluster headache, and Tension-type headache

----

## Features

- **Dual Diagnostic Modules** — separate models and input forms for migraine subtype classification and general headache classification
- **Confidence Scoring** — displays the model's prediction confidence as a percentage
- **SHAP Explanations** — waterfall plots showing how each input feature contributed to the prediction
- **LIME Explanations** — local surrogate model plots highlighting the most influential features
- **Smart Cross-referral** — if a migraine prediction returns "Other", the system recommends switching to the headache classifier, and vice versa
- **Non-technical UI** — plain-language explanations of XAI plots designed for clinical users
- **Fully Responsive Web Interface** — works on desktop and mobile browsers

----

##  How It Works

### 1. Input Collection
The user fills in a clinical symptom form — including pain location, character, intensity, associated symptoms, and patient history. Each field maps directly to features used during model training.

### 2. Preprocessing
Input values are encoded using the same label mappings and `StandardScaler` used during training, ensuring consistent feature representation.

### 3. Prediction
The preprocessed input is passed through a custom-built **TabNet** classifier implemented in PyTorch. The model outputs class probabilities via softmax, and the highest probability class is returned as the prediction.

### 4. Explainability
Two XAI methods run in parallel on each prediction:
- **SHAP** (SHapley Additive exPlanations): uses a background dataset to compute each feature's marginal contribution
- **LIME** (Local Interpretable Model-agnostic Explanations): fits a local linear model around the input to identify the most influential features

The model was trained on clinical datasets of migraine and headache patients. Two separate models were trained:
- **Migraine model** — 17 input features, 7 output classes
- **Headache model** — 19 input features, 3 output classes

## Tech Stack
| Layer | Technology |
|---|---|
| **Backend** | Python, Flask |
| **Deep Learning** | PyTorch |
| **Explainability** | SHAP, LIME |
| **Data Processing** | NumPy, Pandas, scikit-learn |
| **Visualization** | Matplotlib |
| **Server** | Gunicorn (WSGI) |
| **Frontend** | HTML, CSS, JavaScript |
| **Deployment** | HuggingFace Spaces (Docker) |
| **Model Storage** | HuggingFace Model Hub |

## Disclaimer
NeuroTab is a **research prototype** and proof of concept. It is **not intended for clinical use** and should not be used as a substitute for professional medical diagnosis. All predictions are for research and demonstration purposes only.

## License
MIT License.

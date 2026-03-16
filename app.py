from flask import Flask, render_template, request, jsonify, redirect, url_for
from datetime import datetime
import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
import warnings
from huggingface_hub import hf_hub_download

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def ensure_models():
    model_files = [
        "headache_model.pth",
        "headache_scaler.pkl",
        "migraine_model.pth",
        "migraine_scaler.pkl"
    ]
    for filename in model_files:
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            hf_hub_download(
                repo_id="dewanjee/NeuroTab",
                filename=filename,
                repo_type="space",
                local_dir="."
            )
            print(f"Downloaded {filename}")

ensure_models()

app = Flask(__name__)

class GLULayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(GLULayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.bn = nn.BatchNorm1d(output_dim * 2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x_proj = self.fc(x)
        x_proj = self.bn(x_proj)
        x1, x2 = x_proj.chunk(2, dim=-1)
        x_glu = x1 * torch.sigmoid(x2)
        x_glu = self.dropout(x_glu)
        return x_glu

class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, n_glu_layers, shared_layers=None, dropout=0.1):
        super().__init__()
        self.shared = shared_layers
        self.blocks = nn.ModuleList()
        if input_dim != output_dim:
            self.initial_layer = GLULayer(input_dim, output_dim, dropout)
        else:
            self.initial_layer = None
        for i in range(n_glu_layers):
            self.blocks.append(GLULayer(output_dim, output_dim, dropout))
    def forward(self, x):
        if self.initial_layer is not None:
            x = self.initial_layer(x)
        if self.shared is not None and x.size(1) == self.shared[0].fc.out_features // 2:
            for layer in self.shared:
                residual = x
                x = layer(x)
                if residual.shape == x.shape:
                    x = (x + residual) * math.sqrt(0.5)
        for layer in self.blocks:
            residual = x
            x = layer(x)
            if residual.shape == x.shape:
                x = (x + residual) * math.sqrt(0.5)
        return x

class Sparsemax(nn.Module):
    def forward(self, input):
        input = input - input.max(dim=1, keepdim=True)[0]
        z_sorted, _ = torch.sort(input, dim=1, descending=True)
        k = torch.arange(1, input.size(1)+1, device=input.device).float()
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        k_mask = 1 + k * z_sorted > z_cumsum
        k_max = torch.clamp(k_mask.sum(dim=1, keepdim=True), min=1)
        tau_sum = torch.gather(z_cumsum, 1, k_max.long() - 1)
        tau = (tau_sum - 1) / k_max
        output = torch.clamp(input - tau, min=0)
        return output

class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.sparsemax = Sparsemax()
    def forward(self, x, prior):
        x = self.fc(x)
        x = self.bn(x)
        x = x * prior
        x = self.sparsemax(x)
        return x

class DecisionStep(nn.Module):
    def __init__(self, input_dim, feature_dim, output_dim, shared_layers=None, n_glu=2, gamma=1.3, dropout=0.1):
        super().__init__()
        self.gamma = gamma
        self.attentive_transformer = AttentiveTransformer(
            input_dim=feature_dim // 2,
            output_dim=input_dim
        )
        self.feature_transformer = FeatureTransformer(
            input_dim=input_dim,
            output_dim=feature_dim,
            n_glu_layers=n_glu,
            shared_layers=shared_layers,
            dropout=dropout
        )
        self.n_d = feature_dim // 2
        self.n_a = feature_dim - self.n_d
    def forward(self, a_prev, prior, x_o):
        mask = self.attentive_transformer(a_prev, prior)
        masked_x = x_o * mask
        ft_output = self.feature_transformer(masked_x)
        d_i = ft_output[:, :self.n_d]
        a_i = ft_output[:, self.n_d:]
        d_i = torch.nn.functional.relu(d_i)
        prior_next = prior * (self.gamma - mask)
        return d_i, a_i, mask, prior_next

class CustomTabNetEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim=96, n_steps=3, gamma=1.3,
                 n_glu=2, n_shared=2, n_independent=2, dropout=0.1):
        super().__init__()
        self.n_steps = n_steps
        self.gamma = gamma
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.shared_layers = nn.ModuleList()
        for i in range(n_shared):
            self.shared_layers.append(GLULayer(feature_dim, feature_dim, dropout))
        self.initial_transform = FeatureTransformer(
            input_dim=input_dim,
            output_dim=feature_dim,
            n_glu_layers=n_shared + n_independent,
            shared_layers=None,
            dropout=dropout
        )
        self.decision_steps = nn.ModuleList([
            DecisionStep(
                input_dim=input_dim,
                feature_dim=feature_dim,
                output_dim=feature_dim,
                shared_layers=self.shared_layers,
                n_glu=n_glu,
                dropout=dropout
            )
            for _ in range(n_steps)
        ])
        for step in self.decision_steps:
            step.gamma = gamma
        self.bn = nn.BatchNorm1d(input_dim)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x_o = self.bn(x)
        prior = torch.ones_like(x_o)
        initial_output = self.initial_transform(x_o)
        n_d = self.feature_dim // 2
        d0 = initial_output[:, :n_d]
        a0 = initial_output[:, n_d:]
        decision_outputs = []
        a_prev = a0
        for step in self.decision_steps:
            d_i, a_i, mask, prior = step(a_prev, prior, x_o)
            decision_outputs.append(d_i)
            a_prev = a_i
        aggregated = torch.stack(decision_outputs, dim=0).sum(dim=0)
        return aggregated

class CustomTabNetClassifier(nn.Module):
    def __init__(self, input_dim, feature_dim=96, n_steps=3, gamma=1.3,
                 n_glu=2, n_shared=2, n_independent=2, num_classes=7, dropout=0.1):
        super(CustomTabNetClassifier, self).__init__()
        self.encoder = CustomTabNetEncoder(
            input_dim=input_dim,
            feature_dim=feature_dim,
            n_steps=n_steps,
            gamma=gamma,
            n_glu=n_glu,
            n_shared=n_shared,
            n_independent=n_independent,
            dropout=dropout
        )
        self.head = nn.Linear(feature_dim // 2, num_classes)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        encoded = self.encoder(x)
        return self.head(encoded)

def load_model(model_path, scaler_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"Warning: Could not load with weights_only=True. Error: {e}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']
    model_config['num_classes'] = num_classes
    model = CustomTabNetClassifier(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    migraine_model, migraine_scaler = load_model('migraine_model.pth', 'migraine_scaler.pkl', num_classes=7)
    headache_model, headache_scaler = load_model('headache_model.pth', 'headache_scaler.pkl', num_classes=3)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    migraine_model = None
    migraine_scaler = None
    headache_model = None
    headache_scaler = None

migraine_mappings = {
    'Location': {'None': 0, 'Unilateral': 1, 'Bilateral': 2},
    'Character': {'None': 0, 'Throbbing': 1, 'Constant': 2},
    'Intensity': {'None': 0, 'Mild': 1, 'Medium': 2, 'Severe': 3},
    'Vomit': {'No': 0, 'Yes': 1},
    'Phonophobia': {'No': 0, 'Yes': 1},
    'Photophobia': {'No': 0, 'Yes': 1},
    'Dysphasia': {'No': 0, 'Yes': 1},
    'Vertigo': {'No': 0, 'Yes': 1},
    'Tinnitus': {'No': 0, 'Yes': 1},
    'Hypoacusis': {'No': 0, 'Yes': 1},
    'Defect': {'No': 0, 'Yes': 1},
    'Conscience': {'No': 0, 'Yes': 1},
    'DPF': {'No': 0, 'Yes': 1}
}

migraine_descriptions = {
    'Age': 'Age of the patient',
    'Frequency': 'How often do you experience headaches?',
    'Location': 'Where is the pain located?',
    'Character': 'What is the character of the pain?',
    'Intensity': 'How severe is the pain?',
    'Vomit': 'Do you experience vomiting?',
    'Phonophobia': 'Are you sensitive to sound?',
    'Photophobia': 'Are you sensitive to light?',
    'Visual': 'Do you experience visual disturbances?',
    'Sensory': 'Do you experience sensory disturbances?',
    'Dysphasia': 'Do you have difficulty speaking?',
    'Vertigo': 'Do you experience dizziness?',
    'Tinnitus': 'Do you experience ringing in the ears?',
    'Hypoacusis': 'Do you experience hearing loss?',
    'Defect': 'Any visual field defects?',
    'Conscience': 'Any changes in consciousness?',
    'DPF': 'Do you have a family history of migraines?'
}

headache_mappings = {
    'headache_days': {
        'Hours': 0.5, 'Weeks': 7.5, 'A Year': 186,
        'Months': 15, 'More than a Year': 366
    },
    'durationGroup': {
        'Less than 1 minute': 0, '3-5 minutes': 1, '2-4 minutes': 2,
        '4-15 minutes': 3, '15-30 minutes': 4, '30 minutes - 3 hours': 5,
        '3-4 hours': 6, '4 hours - 3 days': 7, '3-7 days': 8, 'More than 1 week': 9
    },
    'location': {'Unilateral': 0, 'Orbital': 1, 'Bilateral': 2},
    'severity': {'Mild': 0, 'Moderate': 1, 'Severe': 2},
    'characterisation': {'Pressing': 0, 'Pulsating': 1, 'Stabbing': 2}
}

headache_descriptions = {
    'durationGroup': 'How long do your headaches typically last?',
    'photophobia': 'Are you sensitive to light?',
    'nausea': 'Do you experience nausea?',
    'aggravation': 'Does physical activity make the pain worse?',
    'characterisation': 'What is the character of the pain?',
    'phonophobia': 'Are you sensitive to sound?',
    'severity': 'How severe is the pain?',
    'rhinorrhoea': 'Do you experience a runny nose?',
    'headache_days': 'First headache experience since:',
    'location': 'Where is the pain located?',
    'lacrimation': 'Do you experience tearing?',
    'conjunctival_injection': 'Are your eyes red?',
    'pericranial': 'Do you experience tenderness in the scalp?',
    'eyelid_oedema': 'Do you have swelling in the eyelids?',
    'sweating': 'Do you experience sweating during headaches?',
    'nasal_congestion': 'Do you experience a stuffy nose?',
    'agitation': 'Do you feel restless during headaches?'
}

headache_binary_features = [
    'photophobia', 'nausea', 'aggravation', 'phonophobia', 'rhinorrhoea',
    'lacrimation', 'conjunctival_injection', 'pericranial', 'eyelid_oedema',
    'sweating', 'nasal_congestion', 'agitation'
]

migraine_classes = [
    'Typical aura with migraine', 'Migraine without aura',
    'Typical aura without migraine', 'Familial hemiplegic migraine',
    'Sporadic hemiplegic migraine', 'Basilar-type aura', 'Other'
]

headache_classes = ['Migraine', 'Cluster', 'Tension-type']

migraine_feature_names = [
    'Age', 'Frequency', 'Location', 'Character', 'Intensity',
    'Vomit', 'Phonophobia', 'Photophobia', 'Visual', 'Sensory',
    'Dysphasia', 'Vertigo', 'Tinnitus', 'Hypoacusis', 'Defect',
    'Conscience', 'DPF'
]

headache_feature_names = [
    'durationGroup', 'photophobia', 'nausea', 'aggravation',
    'characterisation_pulsating', 'phonophobia', 'severity',
    'rhinorrhoea', 'headache_days', 'location_unilateral',
    'lacrimation', 'conjunctival_injection', 'characterisation_stabbing',
    'pericranial', 'eyelid_oedema', 'location_orbital',
    'sweating', 'nasal_congestion', 'agitation'
]

plot_explanations = {
    'shap': "SHAP shows how each feature contributed to the prediction. Red features increased the probability, blue features decreased it.",
    'lime': "LIME highlights the most important features for this prediction. Green bars support the prediction, red bars contradict it."
}

@app.route('/')
def index():
    return render_template('index.html', now=datetime.now())

@app.route('/migraine')
def migraine():
    return render_template('migraine.html',
                           mappings=migraine_mappings,
                           descriptions=migraine_descriptions)

@app.route('/headache')
def headache():
    return render_template('headache.html',
                           mappings=headache_mappings,
                           descriptions=headache_descriptions,
                           binary_features=headache_binary_features)

@app.route('/predict_migraine', methods=['POST'])
def predict_migraine():
    print("Received migraine prediction request")
    try:
        if migraine_model is None or migraine_scaler is None:
            return jsonify({'error': 'Model not loaded properly. Please check server logs.'}), 500
        if not all(request.form.values()):
            return jsonify({'error': 'Please fill in all fields'}), 400
        input_data = []
        input_data.append(float(request.form['age']))
        input_data.append(float(request.form['frequency']))
        input_data.append(migraine_mappings['Location'][request.form['location']])
        input_data.append(migraine_mappings['Character'][request.form['character']])
        input_data.append(migraine_mappings['Intensity'][request.form['intensity']])
        input_data.append(migraine_mappings['Vomit'][request.form['vomit']])
        input_data.append(migraine_mappings['Phonophobia'][request.form['phonophobia']])
        input_data.append(migraine_mappings['Photophobia'][request.form['photophobia']])
        input_data.append(float(request.form['visual']))
        input_data.append(float(request.form['sensory']))
        input_data.append(migraine_mappings['Dysphasia'][request.form['dysphasia']])
        input_data.append(migraine_mappings['Vertigo'][request.form['vertigo']])
        input_data.append(migraine_mappings['Tinnitus'][request.form['tinnitus']])
        input_data.append(migraine_mappings['Hypoacusis'][request.form['hypoacusis']])
        input_data.append(migraine_mappings['Defect'][request.form['defect']])
        input_data.append(migraine_mappings['Conscience'][request.form['conscience']])
        input_data.append(migraine_mappings['DPF'][request.form['dpf']])
        input_data = np.array(input_data).reshape(1, -1)
        prediction, confidence = make_prediction(migraine_model, migraine_scaler, input_data)
        print(f"Migraine prediction: {prediction}, confidence: {confidence}")
        plots = generate_plots(migraine_model, migraine_scaler, input_data, migraine_classes, prediction, migraine_feature_names)
        recommendation = None
        if migraine_classes[prediction] == 'Other':
            recommendation = 'headache'
        return jsonify({
            'prediction': migraine_classes[prediction],
            'confidence': confidence,
            'plots': plots,
            'explanations': plot_explanations,
            'recommendation': recommendation
        })
    except Exception as e:
        print(f"Error in migraine prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_headache', methods=['POST'])
def predict_headache():
    print("Received headache prediction request")
    try:
        if headache_model is None or headache_scaler is None:
            return jsonify({'error': 'Model not loaded properly. Please check server logs.'}), 500
        if not all(request.form.values()):
            return jsonify({'error': 'Please fill in all fields'}), 400
        input_data = []
        input_data.append(headache_mappings['durationGroup'][request.form['durationGroup']])
        input_data.append(1 if request.form['photophobia'] == 'Yes' else 0)
        input_data.append(1 if request.form['nausea'] == 'Yes' else 0)
        input_data.append(1 if request.form['aggravation'] == 'Yes' else 0)
        characterisation = request.form['characterisation']
        input_data.append(1 if characterisation == 'Pulsating' else 0)
        input_data.append(1 if request.form['phonophobia'] == 'Yes' else 0)
        input_data.append(headache_mappings['severity'][request.form['severity']])
        input_data.append(1 if request.form['rhinorrhoea'] == 'Yes' else 0)
        input_data.append(headache_mappings['headache_days'][request.form['headache_days']])
        location = request.form['location']
        input_data.append(1 if location == 'Unilateral' else 0)
        input_data.append(1 if request.form['lacrimation'] == 'Yes' else 0)
        input_data.append(1 if request.form['conjunctival_injection'] == 'Yes' else 0)
        input_data.append(1 if characterisation == 'Stabbing' else 0)
        input_data.append(1 if request.form['pericranial'] == 'Yes' else 0)
        input_data.append(1 if request.form['eyelid_oedema'] == 'Yes' else 0)
        input_data.append(1 if location == 'Orbital' else 0)
        input_data.append(1 if request.form['sweating'] == 'Yes' else 0)
        input_data.append(1 if request.form['nasal_congestion'] == 'Yes' else 0)
        input_data.append(1 if request.form['agitation'] == 'Yes' else 0)
        input_data = np.array(input_data).reshape(1, -1)
        prediction, confidence = make_prediction(headache_model, headache_scaler, input_data)
        print(f"Headache prediction: {prediction}, confidence: {confidence}")
        plots = generate_plots(headache_model, headache_scaler, input_data, headache_classes, prediction, headache_feature_names)
        recommendation = None
        if headache_classes[prediction] == 'Migraine':
            recommendation = 'migraine'
        return jsonify({
            'prediction': headache_classes[prediction],
            'confidence': confidence,
            'plots': plots,
            'explanations': plot_explanations,
            'recommendation': recommendation
        })
    except Exception as e:
        print(f"Error in headache prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

def make_prediction(model, scaler, input_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        prediction = np.argmax(probabilities)
        confidence = probabilities[prediction]
    return prediction, float(confidence)

def generate_plots(model, scaler, input_data, class_names, prediction, feature_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_fn(data):
        data_scaled = scaler.transform(data)
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(data_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        return probs

    def model_fn_scaled(x):
        x_scaled = scaler.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
        return logits.cpu().numpy()

    X_one_df = pd.DataFrame(input_data, columns=feature_names)
    background_unscaled = pd.DataFrame(
        np.random.randn(100, input_data.shape[1]) * 0.1 + input_data,
        columns=feature_names
    )
    plots = {}

    try:
        explainer_shap = shap.Explainer(model_fn_scaled, background_unscaled)
        shap_values = explainer_shap(X_one_df)
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0][:, prediction], show=False)
        plt.figtext(0.5, 0.01, plot_explanations['shap'], ha="center", fontsize=10,
                    bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots['shap'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    except Exception as e:
        print(f"Error generating SHAP plot: {str(e)}")
        try:
            background_scaled = scaler.transform(background_unscaled)
            explainer_fallback = shap.KernelExplainer(predict_fn, background_scaled)
            shap_values_fallback = explainer_fallback.shap_values(input_data)
            plt.figure(figsize=(10, 6))
            if isinstance(shap_values_fallback, list):
                class_shap_values = shap_values_fallback[prediction][0]
            else:
                class_shap_values = shap_values_fallback[0]
            shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': class_shap_values})
            shap_df = shap_df.sort_values('SHAP Value', ascending=False)
            plt.barh(shap_df['Feature'], shap_df['SHAP Value'])
            plt.xlabel('SHAP Value')
            plt.title(f'SHAP Feature Importance for {class_names[prediction]}')
            plt.figtext(0.5, 0.01, plot_explanations['shap'], ha="center", fontsize=10,
                        bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plots['shap'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        except Exception as e2:
            print(f"Error generating SHAP bar plot: {str(e2)}")
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, f'SHAP Plot Error: {str(e)}', ha='center', va='center')
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plots['shap'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

    try:
        background_scaled = scaler.transform(background_unscaled)
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=background_scaled,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )
        exp_lime = explainer_lime.explain_instance(
            scaler.transform(input_data)[0],
            predict_fn=lambda x: predict_fn(scaler.inverse_transform(x)),
            num_features=min(10, input_data.shape[1]),
            top_labels=1
        )
        lime_exp = exp_lime.as_list(label=prediction)
        fig, ax = plt.subplots(figsize=(10, 6))
        features = [x[0] for x in lime_exp]
        values = [x[1] for x in lime_exp]
        colors = ['green' if x > 0 else 'red' for x in values]
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Weight')
        ax.set_title(f'LIME Explanation for {class_names[prediction]}')
        plt.figtext(0.5, 0.01, plot_explanations['lime'], ha="center", fontsize=10,
                    bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots['lime'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    except Exception as e:
        print(f"Error generating LIME plot: {str(e)}")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'LIME Plot Error: {str(e)}', ha='center', va='center')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots['lime'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

    return plots

if __name__ == '__main__':
    app.run(debug=True)
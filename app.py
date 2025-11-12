"""
NeuroVoice AI - Flask Backend
Parkinson's Disease Detection from Voice Input
"""

from flask import Flask, render_template, request, jsonify, send_file
import librosa
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from fpdf import FPDF
from werkzeug.utils import secure_filename
import soundfile as sf

# --- CORS Fix ---
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # <-- This enables secure cross-origin requests!

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = os.environ.get('SESSION_SECRET', 'neurovoice-ai-secret-key-2024')

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

try:
    with open('parkinsons_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        feature_columns = model_data.get('feature_columns', ['jitter', 'shimmer', 'hnr', 'mfcc_mean', 'mfcc_std', 'pitch_mean', 'pitch_std', 'energy_mean', 'spectral_centroid', 'zero_crossing_rate'])
        feature_stats = model_data.get('feature_stats', {})
    print("Model loaded successfully!")
    print(f"Feature columns: {feature_columns}")
except FileNotFoundError:
    print("Warning: parkinsons_model.pkl not found. Run train_model.py first.")
    model = None
    scaler = None
    feature_columns = []
    feature_stats = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) == 0:
            raise ValueError("Audio file is empty")
        y = librosa.util.normalize(y)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) < sr * 0.5:
            y_trimmed = y
        frame_length = min(2048, len(y_trimmed))
        hop_length = frame_length // 4
        pitches, magnitudes = librosa.piptrack(y=y_trimmed, sr=sr, fmin=50, fmax=400, hop_length=hop_length)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        if len(pitch_values) > 0:
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            pitch_periods = 1.0 / (np.array(pitch_values) + 1e-10)
            if len(pitch_periods) > 1:
                period_diffs = np.abs(np.diff(pitch_periods))
                jitter = np.mean(period_diffs) / np.mean(pitch_periods) if np.mean(pitch_periods) > 0 else 0.005
            else:
                jitter = 0.005
        else:
            pitch_mean = 150
            pitch_std = 40
            jitter = 0.005
        rms = librosa.feature.rms(y=y_trimmed, frame_length=frame_length, hop_length=hop_length)[0]
        if len(rms) > 1:
            rms_diffs = np.abs(np.diff(rms))
            shimmer = np.mean(rms_diffs) / (np.mean(rms) + 1e-10)
        else:
            shimmer = 0.02
        harmonic, percussive = librosa.effects.hpss(y_trimmed)
        harmonic_energy = np.sum(harmonic ** 2)
        noise_energy = np.sum(percussive ** 2)
        hnr = 10 * np.log10((harmonic_energy + 1e-10) / (noise_energy + 1e-10))
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc_mean = np.mean(mfccs)
        mfcc_std = np.std(mfccs)
        energy_mean = np.mean(rms)
        spectral_centroids = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr, hop_length=hop_length)[0]
        spectral_centroid = np.mean(spectral_centroids)
        zcr = librosa.feature.zero_crossing_rate(y_trimmed, frame_length=frame_length, hop_length=hop_length)[0]
        zero_crossing_rate = np.mean(zcr)
        features = {
            'jitter': float(jitter),
            'shimmer': float(shimmer),
            'hnr': float(hnr),
            'mfcc_mean': float(mfcc_mean),
            'mfcc_std': float(mfcc_std),
            'pitch_mean': float(pitch_mean),
            'pitch_std': float(pitch_std),
            'energy_mean': float(energy_mean),
            'spectral_centroid': float(spectral_centroid),
            'zero_crossing_rate': float(zero_crossing_rate)
        }
        return features
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        raise

def validate_features(features, feature_stats):
    validated = features.copy()
    if feature_stats and 'mins' in feature_stats and 'maxs' in feature_stats:
        for feat_name in features.keys():
            if feat_name in feature_stats['mins']:
                min_val = feature_stats['mins'][feat_name] * 0.5
                max_val = feature_stats['maxs'][feat_name] * 1.5
                if features[feat_name] < min_val or features[feat_name] > max_val:
                    print(f"Warning: {feat_name} = {features[feat_name]:.5f} outside training range")
                    validated[feat_name] = np.clip(features[feat_name], min_val, max_val)
    return validated

def predict_parkinsons(features):
    if model is None or scaler is None:
        raise ValueError("Model not loaded. Please run train_model.py first.")
    validated_features = validate_features(features, feature_stats)
    features_df = pd.DataFrame([validated_features])[feature_columns]
    feature_scaled = scaler.transform(features_df)
    prediction = model.predict(feature_scaled)[0]
    probability = model.predict_proba(feature_scaled)[0]
    confidence = float(probability[prediction])
    if confidence < 0.6:
        print(f"Warning: Low confidence prediction ({confidence:.2%}). Results may be uncertain.")
    feature_importance = model.feature_importances_
    feature_values = features_df.values[0]
    top_features = sorted(zip(feature_columns, feature_importance, feature_values), key=lambda x: x[1], reverse=True)[:5]
    return {
        'prediction': int(prediction),
        'confidence': confidence,
        'probability_healthy': float(probability[0]),
        'probability_parkinsons': float(probability[1]),
        'top_features': [
            {
                'name': name,
                'importance': float(importance),
                'value': float(value)
            }
            for name, importance, value in top_features
        ],
        'needs_review': confidence < 0.6
    }

def generate_pdf_report(user_data, features, prediction_result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(63, 81, 181)
    pdf.cell(0, 15, "NeuroVoice AI", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Parkinson's Disease Voice Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", '', 11)
    pdf.cell(50, 8, "Name:", 0)
    pdf.cell(0, 8, user_data.get('name', 'N/A'), ln=True)
    pdf.cell(50, 8, "Age:", 0)
    pdf.cell(0, 8, str(user_data.get('age', 'N/A')), ln=True)
    pdf.cell(50, 8, "Gender:", 0)
    pdf.cell(0, 8, user_data.get('gender', 'N/A'), ln=True)
    pdf.cell(50, 8, "Date:", 0)
    pdf.cell(0, 8, datetime.now().strftime("%B %d, %Y %H:%M"), ln=True)
    pdf.ln(8)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Analysis Results", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    result_text = "Parkinson's Disease Detected" if prediction_result['prediction'] == 1 else "Healthy Voice Pattern"
    confidence_pct = prediction_result['confidence'] * 100
    pdf.set_font("Arial", 'B', 12)
    if prediction_result['prediction'] == 1:
        pdf.set_text_color(220, 53, 69)
    else:
        pdf.set_text_color(40, 167, 69)
    pdf.cell(0, 10, f"Result: {result_text}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Confidence: {confidence_pct:.1f}%", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Voice Features Analysis", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", '', 10)
    feature_names = {
        'jitter': 'Jitter (voice instability)',
        'shimmer': 'Shimmer (amplitude variation)',
        'hnr': 'Harmonic-to-Noise Ratio',
        'pitch_mean': 'Average Pitch',
        'pitch_std': 'Pitch Variation'
    }
    for feature_data in prediction_result['top_features'][:5]:
        name = feature_data['name']
        value = feature_data['value']
        display_name = feature_names.get(name, name.replace('_', ' ').title())
        pdf.cell(0, 7, f"{display_name}: {value:.4f}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Medical Disclaimer", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", '', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, "This analysis is for informational purposes only and should not be considered as medical advice. "
                         "If Parkinson's Disease is detected or suspected, please consult a qualified neurologist or healthcare "
                         "professional for proper diagnosis and treatment. Early consultation with medical experts is recommended.")
    pdf.ln(5)
    if prediction_result['prediction'] == 1:
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(220, 53, 69)
        pdf.cell(0, 8, "Recommendation: Consult a neurologist for further evaluation", ln=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"neurovoice_report_{timestamp}.pdf"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf.output(filepath)
    return filepath

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please contact administrator.'}), 500
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        user_data = {
            'name': request.form.get('name', 'Anonymous'),
            'age': request.form.get('age', 'N/A'),
            'gender': request.form.get('gender', 'N/A')
        }
        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        features = extract_features(filepath)
        prediction_result = predict_parkinsons(features)
        pdf_path = generate_pdf_report(user_data, features, prediction_result)
        response = {
            'success': True,
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'probability_healthy': prediction_result['probability_healthy'],
            'probability_parkinsons': prediction_result['probability_parkinsons'],
            'features': features,
            'top_features': prediction_result['top_features'],
            'pdf_filename': os.path.basename(pdf_path),
            'user_data': user_data
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_report(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 404

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
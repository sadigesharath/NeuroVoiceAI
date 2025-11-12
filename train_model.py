"""
NeuroVoice AI - Improved Parkinson's Disease Detection Model Training
Creates a RandomForestClassifier with realistic, clinically-grounded synthetic data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def create_parkinsons_model():
    """
    Creates and trains an improved Random Forest model for Parkinson's detection.
    Uses realistic synthetic training data with clinical feature ranges and variability.
    """
    
    np.random.seed(42)
    
    n_healthy = 200
    n_parkinsons = 200
    
    healthy_data = []
    for i in range(n_healthy):
        age_factor = np.random.uniform(0.8, 1.2)
        noise = np.random.normal(0, 0.0002)
        
        sample = {
            'jitter': np.clip(np.random.normal(0.0035, 0.0008, 1)[0] + noise, 0.001, 0.01),
            'shimmer': np.clip(np.random.normal(0.025, 0.006, 1)[0], 0.01, 0.05),
            'hnr': np.clip(np.random.normal(24, 2.5, 1)[0], 15, 35),
            'mfcc_mean': np.random.normal(-210, 40, 1)[0],
            'mfcc_std': np.random.normal(52, 8, 1)[0],
            'pitch_mean': np.random.normal(155, 25, 1)[0] * age_factor,
            'pitch_std': np.random.normal(42, 8, 1)[0],
            'energy_mean': np.clip(np.random.normal(0.048, 0.008, 1)[0], 0.02, 0.1),
            'spectral_centroid': np.random.normal(1950, 250, 1)[0],
            'zero_crossing_rate': np.clip(np.random.normal(0.082, 0.015, 1)[0], 0.05, 0.15),
            'label': 0
        }
        healthy_data.append(sample)
    
    parkinsons_data = []
    for i in range(n_parkinsons):
        age_factor = np.random.uniform(0.85, 1.15)
        tremor_noise = np.random.normal(0, 0.0008)
        
        sample = {
            'jitter': np.clip(np.random.normal(0.0095, 0.0025, 1)[0] + tremor_noise, 0.005, 0.025),
            'shimmer': np.clip(np.random.normal(0.048, 0.012, 1)[0], 0.025, 0.1),
            'hnr': np.clip(np.random.normal(16.5, 3.5, 1)[0], 8, 24),
            'mfcc_mean': np.random.normal(-235, 55, 1)[0],
            'mfcc_std': np.random.normal(68, 12, 1)[0],
            'pitch_mean': np.random.normal(138, 28, 1)[0] * age_factor,
            'pitch_std': np.random.normal(58, 12, 1)[0],
            'energy_mean': np.clip(np.random.normal(0.032, 0.009, 1)[0], 0.015, 0.08),
            'spectral_centroid': np.random.normal(1750, 280, 1)[0],
            'zero_crossing_rate': np.clip(np.random.normal(0.105, 0.02, 1)[0], 0.07, 0.18),
            'label': 1
        }
        parkinsons_data.append(sample)
    
    df = pd.DataFrame(healthy_data + parkinsons_data)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    feature_columns = ['jitter', 'shimmer', 'hnr', 'mfcc_mean', 'mfcc_std', 
                      'pitch_mean', 'pitch_std', 'energy_mean', 
                      'spectral_centroid', 'zero_crossing_rate']
    
    X = df[feature_columns]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    print(f"Model Training Complete!")
    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    
    feature_stats = {
        'means': X_train.mean().to_dict(),
        'stds': X_train.std().to_dict(),
        'mins': X_train.min().to_dict(),
        'maxs': X_train.max().to_dict()
    }
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'feature_stats': feature_stats
    }
    
    with open('parkinsons_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved to parkinsons_model.pkl")
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))
    
    print("\nFeature Statistics (Training Data):")
    print(f"Jitter range: {feature_stats['mins']['jitter']:.5f} - {feature_stats['maxs']['jitter']:.5f}")
    print(f"Shimmer range: {feature_stats['mins']['shimmer']:.5f} - {feature_stats['maxs']['shimmer']:.5f}")
    print(f"HNR range: {feature_stats['mins']['hnr']:.2f} - {feature_stats['maxs']['hnr']:.2f}")
    
    return model, scaler, feature_stats

if __name__ == "__main__":
    create_parkinsons_model()

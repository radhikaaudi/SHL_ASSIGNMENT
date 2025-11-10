#!/usr/bin/env python
"""
Grammar Scoring Engine - Standalone Script
This script runs the complete pipeline to generate submission.csv
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import librosa
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

print("="*70)
print("GRAMMAR SCORING ENGINE - RUNNING PIPELINE")
print("="*70)

# ============================================================================
# 1. Data Loading
# ============================================================================
print("\n[1/8] Loading data...")
train_df = pd.read_csv('datasets/csvs/train.csv')
test_df = pd.read_csv('datasets/csvs/test.csv')
print(f"Training: {len(train_df)} samples, Test: {len(test_df)} samples")

# ============================================================================
# 2. Audio Feature Extraction Functions
# ============================================================================
print("\n[2/8] Defining feature extraction functions...")

def extract_audio_features(audio_path, sr=16000):
    """Extract comprehensive audio features from a WAV file."""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=60)
        features = {}
        
        # Basic audio properties
        features['duration'] = len(y) / sr
        features['rms_energy'] = np.mean(librosa.feature.rms(y=y)[0])
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # MFCC features (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        # Prosody features (pitch)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # Pause detection
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        silence_threshold = np.percentile(rms, 10)
        silence_ratio = np.sum(rms < silence_threshold) / len(rms)
        features['silence_ratio'] = silence_ratio
        
        features['energy_variation'] = np.std(librosa.feature.rms(y=y)[0])

        # Additional robust features
        try:
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            log_mel = librosa.power_to_db(mel + 1e-9)
            features['log_mel_mean'] = float(np.mean(log_mel))
            features['log_mel_std'] = float(np.std(log_mel))
            features['log_mel_p10'] = float(np.percentile(log_mel, 10))
            features['log_mel_p50'] = float(np.percentile(log_mel, 50))
            features['log_mel_p90'] = float(np.percentile(log_mel, 90))
        except Exception:
            features['log_mel_mean'] = 0.0
            features['log_mel_std'] = 0.0
            features['log_mel_p10'] = 0.0
            features['log_mel_p50'] = 0.0
            features['log_mel_p90'] = 0.0

        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['contrast_mean'] = float(np.mean(contrast))
            features['contrast_std'] = float(np.std(contrast))
        except Exception:
            features['contrast_mean'] = 0.0
            features['contrast_std'] = 0.0

        try:
            y_harm = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
            features['tonnetz_mean'] = float(np.mean(tonnetz))
            features['tonnetz_std'] = float(np.std(tonnetz))
        except Exception:
            features['tonnetz_mean'] = 0.0
            features['tonnetz_std'] = 0.0

        # Delta MFCCs
        try:
            delta_mfcc = librosa.feature.delta(mfccs)
            for i in range(min(13, delta_mfcc.shape[0])):
                features[f'delta_mfcc_{i}_mean'] = float(np.mean(delta_mfcc[i]))
                features[f'delta_mfcc_{i}_std'] = float(np.std(delta_mfcc[i]))
        except Exception:
            for i in range(13):
                features[f'delta_mfcc_{i}_mean'] = 0.0
                features[f'delta_mfcc_{i}_std'] = 0.0

        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        default_features = {f'mfcc_{i}_mean': 0 for i in range(13)}
        default_features.update({f'mfcc_{i}_std': 0 for i in range(13)})
        default_features.update({
            'duration': 0, 'rms_energy': 0, 'zero_crossing_rate': 0,
            'spectral_centroid_mean': 0, 'spectral_centroid_std': 0,
            'spectral_rolloff_mean': 0, 'spectral_rolloff_std': 0,
            'chroma_mean': 0, 'chroma_std': 0, 'tempo': 0,
            'pitch_mean': 0, 'pitch_std': 0, 'pitch_range': 0,
            'silence_ratio': 0, 'energy_variation': 0,
            'log_mel_mean': 0.0, 'log_mel_std': 0.0, 'log_mel_p10': 0.0, 'log_mel_p50': 0.0, 'log_mel_p90': 0.0,
            'contrast_mean': 0.0, 'contrast_std': 0.0,
            'tonnetz_mean': 0.0, 'tonnetz_std': 0.0
        })
        for i in range(13):
            default_features[f'delta_mfcc_{i}_mean'] = 0.0
            default_features[f'delta_mfcc_{i}_std'] = 0.0
        return default_features

def extract_text_features_simple(audio_path):
    """Extract text-based features using audio proxies."""
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=60)
        features = {}
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.sum(np.abs(y_harmonic)) / (np.sum(np.abs(y)) + 1e-10)
        features['harmonic_ratio'] = harmonic_ratio
        
        rms = librosa.feature.rms(y=y)[0]
        rms_threshold = np.percentile(rms, 20)
        continuous_speech_ratio = np.sum(rms > rms_threshold) / len(rms)
        features['continuous_speech_ratio'] = continuous_speech_ratio
        
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['prosody_consistency'] = 1.0 / (np.std(spectral_centroids) + 1e-10)
        
        return features
    except Exception as e:
        return {
            'spectral_bandwidth_mean': 0, 'spectral_bandwidth_std': 0,
            'harmonic_ratio': 0, 'continuous_speech_ratio': 0,
            'prosody_consistency': 0
        }

# ============================================================================
# 3. Extract Features for Training Data
# ============================================================================
print("\n[3/8] Extracting audio features for training data...")
train_audio_features = []

for idx, row in train_df.iterrows():
    filename = row['filename']
    audio_path = f'datasets/audios/train/{filename}.wav'
    if not Path(audio_path).exists():
        audio_path = f'datasets/audios/train/audio_{filename.split("_")[-1]}.wav'
    
    features = extract_audio_features(audio_path)
    features['filename'] = filename
    train_audio_features.append(features)
    
    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{len(train_df)} files...")

train_features_df = pd.DataFrame(train_audio_features)

# Add text features
print("  Adding text-based features...")
train_text_features = []
for idx, row in train_df.iterrows():
    filename = row['filename']
    audio_path = f'datasets/audios/train/{filename}.wav'
    if not Path(audio_path).exists():
        audio_path = f'datasets/audios/train/audio_{filename.split("_")[-1]}.wav'
    
    features = extract_text_features_simple(audio_path)
    features['filename'] = filename
    train_text_features.append(features)

train_text_df = pd.DataFrame(train_text_features)
train_all_features = train_features_df.merge(train_text_df, on='filename', how='inner')
train_all_features = train_all_features.merge(train_df[['filename', 'label']], on='filename', how='inner')
print(f"  Total features extracted: {len(train_all_features.columns) - 2}")

# ============================================================================
# 4. Extract Features for Test Data
# ============================================================================
print("\n[4/8] Extracting audio features for test data...")
test_audio_features = []

for idx, row in test_df.iterrows():
    filename = row['filename']
    audio_path = f'datasets/audios/test/{filename}.wav'
    if not Path(audio_path).exists():
        audio_path = f'datasets/audios/test/audio_{filename}.wav'
    
    features = extract_audio_features(audio_path)
    features['filename'] = filename
    test_audio_features.append(features)
    
    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{len(test_df)} files...")

test_features_df = pd.DataFrame(test_audio_features)

# Add text features
print("  Adding text-based features...")
test_text_features = []
for idx, row in test_df.iterrows():
    filename = row['filename']
    audio_path = f'datasets/audios/test/{filename}.wav'
    if not Path(audio_path).exists():
        audio_path = f'datasets/audios/test/audio_{filename}.wav'
    
    features = extract_text_features_simple(audio_path)
    features['filename'] = filename
    test_text_features.append(features)

test_text_df = pd.DataFrame(test_text_features)
test_all_features = test_features_df.merge(test_text_df, on='filename', how='inner')

# ============================================================================
# 5. Feature Preprocessing
# ============================================================================
print("\n[5/8] Preprocessing features...")
feature_cols = [col for col in train_all_features.columns if col not in ['filename', 'label']]

X_train = train_all_features[feature_cols].copy()
y_train = train_all_features['label'].copy()
X_test = test_all_features[feature_cols].copy()

# Handle NaN and infinite values
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  Training shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

# ============================================================================
# 6. Model Training
# ============================================================================
print("\n[6/8] Training models (5-fold CV XGBoost + RF baseline)...")

# 5-fold CV XGBoost to get OOF predictions
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
xgb_models = []
oof_pred = np.zeros(len(y_train))
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_scaled), start=1):
    print(f"  Fold {fold}/5 - XGBoost")
    xgb_model = xgb.XGBRegressor(
        n_estimators=XGB_TREES,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled[tr_idx], y_train.iloc[tr_idx])
    oof_pred[va_idx] = xgb_model.predict(X_train_scaled[va_idx])
    xgb_models.append(xgb_model)

# Random Forest trained on full data for ensembling
rf_model_full = RandomForestRegressor(
    n_estimators=RF_TREES,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_model_full.fit(X_train_scaled, y_train)

# ============================================================================
# 7. Evaluation on Full Training Data
# ============================================================================
print("\n[7/8] Evaluating on full training data...")

# Aggregate CV XGB predictions on train
xgb_full_pred = np.zeros(len(y_train))
for m in xgb_models:
    xgb_full_pred += m.predict(X_train_scaled) / len(xgb_models)
rf_full_pred = rf_model_full.predict(X_train_scaled)
ensemble_full_pred = 0.6 * xgb_full_pred + 0.4 * rf_full_pred

xgb_full_rmse = np.sqrt(mean_squared_error(y_train, xgb_full_pred))
rf_full_rmse = np.sqrt(mean_squared_error(y_train, rf_full_pred))
ensemble_full_rmse = np.sqrt(mean_squared_error(y_train, ensemble_full_pred))

xgb_full_pearson, _ = pearsonr(y_train, xgb_full_pred)
rf_full_pearson, _ = pearsonr(y_train, rf_full_pred)
ensemble_full_pearson, _ = pearsonr(y_train, ensemble_full_pred)

print("\n" + "="*60)
print("FULL TRAINING DATA EVALUATION (REQUIRED FOR SUBMISSION)")
print("="*60)
print(f"\nXGBoost:")
print(f"  RMSE: {xgb_full_rmse:.4f}")
print(f"  Pearson Correlation: {xgb_full_pearson:.4f}")

print(f"\nRandom Forest:")
print(f"  RMSE: {rf_full_rmse:.4f}")
print(f"  Pearson Correlation: {rf_full_pearson:.4f}")

print(f"\nEnsemble (60% XGBoost + 40% Random Forest):")
print(f"  RMSE: {ensemble_full_rmse:.4f}")
print(f"  Pearson Correlation: {ensemble_full_pearson:.4f}")
print("="*60)

# Calibrate using OOF predictions vs true labels
calibrator = LinearRegression()
calibrator.fit(oof_pred.reshape(-1, 1), y_train.values)

final_train_rmse = xgb_full_rmse
final_train_pearson = xgb_full_pearson
model_name = "XGBoost (5-fold CV) + RF ensemble + linear calibration"

print(f"\nSelected Model: {model_name}")
print(f"Final Training RMSE: {final_train_rmse:.4f}")
print(f"Final Training Pearson Correlation: {final_train_pearson:.4f}")

# ============================================================================
# 8. Generate Test Predictions and Save Submission
# ============================================================================
print("\n[8/8] Generating test predictions...")

# XGB CV prediction + RF ensemble
xgb_test_pred = np.zeros(len(X_test_scaled))
for m in xgb_models:
    xgb_test_pred += m.predict(X_test_scaled) / len(xgb_models)
rf_test_pred = rf_model_full.predict(X_test_scaled)
test_predictions = 0.6 * xgb_test_pred + 0.4 * rf_test_pred

# Apply calibration
test_predictions = calibrator.predict(test_predictions.reshape(-1, 1))

# Clip to valid range of rubric
test_predictions = np.clip(test_predictions, 1, 5)

# Create submission
submission_df = pd.DataFrame({
    'filename': test_df['filename'],
    'label': test_predictions
})

submission_df.to_csv('submission.csv', index=False)

print("\n" + "="*70)
print("SUBMISSION FILE GENERATED SUCCESSFULLY!")
print("="*70)
print(f"\nFile: submission.csv")
print(f"Number of predictions: {len(submission_df)}")
print(f"Prediction range: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")
print(f"Mean prediction: {test_predictions.mean():.2f}")
print(f"Std prediction: {test_predictions.std():.2f}")
print("\nFirst 10 predictions:")
print(submission_df.head(10))
print("\n" + "="*70)


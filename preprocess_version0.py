import os
import librosa
import numpy as np
import pandas as pd

# Define emotion mapping
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Function to extract features from audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)  # 40 MFCCs
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
    return features

# Load data
data = []
labels = []
dataset_dir = 'Actors'

for actor_dir in os.listdir(dataset_dir):
    if actor_dir.startswith('Actor_'):
        actor_path = os.path.join(dataset_dir, actor_dir)
        for file in os.listdir(actor_path):
            if file.endswith('.wav'):
                file_path = os.path.join(actor_path, file)
                parts = file.split('-')
                emotion_code = parts[2]  # Third part is emotion
                if emotion_code in emotions:
                    features = extract_features(file_path)
                    data.append(features)
                    labels.append(emotions[emotion_code])

# Convert to DataFrame
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv('ravdess_features.csv', index=False)  # Save for later use
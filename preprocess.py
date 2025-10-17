import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  
import time

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
def extract_features(file_path, max_length=3.0):
    """
    Extract features with error handling and timeout
    """
    try:
        # Load audio with duration limit to avoid long files
        y, sr = librosa.load(file_path, sr=None, duration=max_length)
        
        # Extract features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        
        features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # Return zeros or skip - here we return zeros to maintain alignment
        return np.zeros(193)

def count_files(dataset_dir):
    """Count total wav files for progress tracking"""
    total_files = 0
    for actor_dir in os.listdir(dataset_dir):
        if actor_dir.startswith('Actor_'):
            actor_path = os.path.join(dataset_dir, actor_dir)
            if os.path.isdir(actor_path):
                wav_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
                total_files += len(wav_files)
    return total_files

# Main processing
def process_dataset(dataset_dir='Actors'):
    print(f"🔍 Scanning dataset in '{dataset_dir}'...")
    total_files = count_files(dataset_dir)
    print(f"📊 Found {total_files} audio files across {len([d for d in os.listdir(dataset_dir) if d.startswith('Actor_')])} actors")
    
    data = []
    labels = []
    valid_files = 0
    skipped_files = 0
    emotion_counts = {emotion: 0 for emotion in emotions.values()}
    
    start_time = time.time()
    
    # Progress bar for all files
    all_files = []
    for actor_dir in os.listdir(dataset_dir):
        if actor_dir.startswith('Actor_'):
            actor_path = os.path.join(dataset_dir, actor_dir)
            if os.path.isdir(actor_path):
                for file in os.listdir(actor_path):
                    if file.endswith('.wav'):
                        all_files.append((os.path.join(actor_path, file), file))
    
    print(f"🚀 Starting feature extraction for {len(all_files)} files...")
    print("⏳ This may take 10-30 minutes depending on your machine\n")
    
    # Process files with progress bar
    for file_path, filename in tqdm(all_files, desc="Processing audio files", unit="file"):
        try:
            # Parse emotion from filename
            parts = filename.split('-')
            if len(parts) >= 3 and parts[2] in emotions:
                emotion_code = parts[2]
                emotion_label = emotions[emotion_code]
                
                # Extract features
                features = extract_features(file_path)
                
                if not np.all(features == 0):  # Check if features were successfully extracted
                    data.append(features)
                    labels.append(emotion_label)
                    emotion_counts[emotion_label] += 1
                    valid_files += 1
                else:
                    skipped_files += 1
                
            else:
                print(f"⚠️  Skipping {filename}: Invalid emotion code {parts[2] if len(parts) >= 3 else 'N/A'}")
                skipped_files += 1
                
        except Exception as e:
            print(f"❌ Error with {filename}: {str(e)}")
            skipped_files += 1
            continue
    
    # Statistics
    end_time = time.time()
    processing_time = end_time - start_time
    
    print("\n" + "="*60)
    print("📈 PROCESSING SUMMARY")
    print("="*60)
    print(f"✅ Successfully processed: {valid_files} files")
    print(f"❌ Skipped/failed: {skipped_files} files")
    print(f"⏱️  Total time: {processing_time/60:.1f} minutes")
    print(f"⚡ Files per minute: {valid_files/(processing_time/60):.0f}")
    
    print("\n🎭 Emotion distribution:")
    for emotion, count in emotion_counts.items():
        percentage = (count/valid_files)*100 if valid_files > 0 else 0
        print(f"  {emotion}: {count} ({percentage:.1f}%)")
    
    print(f"\n💾 Feature matrix shape: {len(data)} samples × {len(data[0]) if data else 0} features")
    
    # Save data
    if data:
        df = pd.DataFrame(data)
        df['label'] = labels
        output_file = 'ravdess_features.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Features saved to '{output_file}'")
        
        # Save emotion counts for reference
        counts_df = pd.DataFrame(list(emotion_counts.items()), columns=['emotion', 'count'])
        counts_df.to_csv('emotion_distribution.csv', index=False)
        print(f"📊 Emotion distribution saved to 'emotion_distribution.csv'")
        
        return df
    else:
        print("❌ No valid data to save!")
        return None

# Run the processing
if __name__ == "__main__":
    dataset_dir = 'Actors'  # Update this path to your actual dataset folder
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory '{dataset_dir}' not found!")
        print("Please update the dataset_dir variable to point to your RAVDESS folder")
    else:
        df = process_dataset(dataset_dir)
        
        if df is not None:
            print(f"\n🎉 Feature extraction completed successfully!")
            print(f"Dataset shape: {df.shape}")
            print(f"Emotions detected: {df['label'].nunique()}")
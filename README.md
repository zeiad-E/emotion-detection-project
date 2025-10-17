# Emotion Recognition from Audio Project

This project preprocesses audio data from the RAVDESS dataset, extracts features, and trains a neural network model for emotion classification using TensorFlow/Keras.

## Prerequisites
- Python 3.8 or higher
- Git
- Access to download datasets

## Cloning the Project
1. Open a terminal and navigate to your desired directory.
2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   ```
3. Navigate to the project folder:
   ```bash
   cd your-repo-name
   ```

## Setting Up the Environment
1. Create a virtual environment (recommended to avoid dependency conflicts):
   - On Windows:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
2. Verify activation (you should see `(.venv)` in your prompt).

## Installing Required Libraries
1. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
2. This will install all necessary libraries (e.g., NumPy, Pandas, Librosa, TensorFlow, etc.).

## Downloading Datasets
- The project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.
- Since the `Actors` and `audio_speech_actors_01-24` directories are not included in the Git repository, download the dataset manually:
  1. Visit the official RAVDESS website or a reliable source (e.g., Kaggle or the University of Toronto's page).
  2. Download the audio files (e.g., `Actor_01`, `Actor_02`, etc., in WAV format).
  3. Create an `Actors` folder in the project root and place the actor subdirectories inside it (e.g., `Actors/Actor_01/`).
- Ensure the dataset structure matches the script expectations (e.g., `Actors/Actor_XX/*.wav` files).

## Running the Project
1. Preprocess the data:
   ```bash
   python preprocess.py
   ```
   - This extracts features from audio files and saves them to `ravdess_features.csv`.

2. Train the model:
   ```bash
   python prepare_data.py
   ```
   - This loads the features, trains the model, and saves it as `emotion_model.h5`.
   - Training plots will be displayed (requires Matplotlib).

3. (Optional) Evaluate or use the model further as needed.

## Project Structure
- `preprocess.py`: Handles audio feature extraction.
- `create_model.py`: Trains the emotion classification model.
- `requirements.txt`: Lists dependencies.
- `ravdess_features.csv`: Generated features (created after running `preprocess.py`).
- `emotion_model.h5`: Trained model (created after running `prepare_data.py`).
- `Actors/`: Dataset folder (not in Git; download separately).
- Other files: Outputs like `emotion_distribution.csv`.

## Notes
- If you encounter errors (e.g., missing modules), ensure your virtual environment is activated and re-run `pip install -r requirements.txt`.
- The model outputs to `emotion_model.h5` (HDF5 format; consider using `.keras` for newer TensorFlow versions).
- For issues or contributions, refer to the repository's issues section.



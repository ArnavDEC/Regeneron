import os
import numpy as np
import audioread
import zipfile
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import soundfile as sf  # Required for reading audio data after using audioread

# Install necessary dependencies
!apt-get install ffmpeg
!pip install soundfile

# Unzipping the dataset
zip_file_path = "/content/dataset.zip"
extract_path = "/content/dataset"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extracted files:", os.listdir(extract_path))

dataset_path = "/content/dataset"
nested_folder = os.path.join(dataset_path, "dataset")
if os.path.exists(nested_folder):
    dataset_path = nested_folder

# Feature extraction function using audioread
# Feature extraction function using audioread
def extract_features(file_path, sr=22050, duration=5):
    try:
        # Use audioread to open and read the file
        with audioread.audio_open(file_path) as audio_file:
            # Ensure we read only the first 'duration' seconds of the audio
            total_frames = int(audio_file.samplerate * duration)
            audio_data = np.array([])

            # Read the audio data using audioread
            for buf in audio_file:
                audio_data = np.append(audio_data, np.frombuffer(buf, dtype=np.int16))

            # Resample audio to match desired sample rate (if necessary)
            if audio_file.samplerate != sr:
                audio_data = librosa.resample(audio_data, orig_sr=audio_file.samplerate, target_sr=sr)

            # Compute MFCCs using soundfile for reading the final data
            audio_data = audio_data[:total_frames]  # truncate if necessary
            mfccs = librosa.feature.mfcc(y=audio_data.astype(np.float32), sr=sr, n_mfcc=40)
            return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


# Load dataset function
def load_dataset(dataset_path):
    features = []
    labels = []

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if not file.endswith(('.wav', '.mp3')) or not os.path.isfile(file_path):
                continue
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(folder)
            else:
                print(f"Feature extraction failed for {file_path}")

    if len(features) == 0 or len(labels) == 0:
        raise ValueError("No valid features or labels found. Check your dataset structure and audio files.")
    else:
        return np.array(features), np.array(labels)

# Load the dataset and preprocess
features, labels = load_dataset(dataset_path)

# Encode labels and convert to categorical
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels_categorical, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(labels_categorical.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save("alzheimers_audio_model.h5")

# Prediction function
def predict_alzheimers(audio_file):
    feature = extract_features(audio_file)
    if feature is None:
        return {"error": "Feature extraction failed. Ensure the audio file is valid."}
    feature = np.expand_dims(feature, axis=0)
    prediction = model.predict(feature)
    likelihood = prediction[0]
    label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return {"label": label, "likelihood": likelihood}

# Test prediction
test_audio = "/content/Pro-Files ｜ John Mackey.wav"
result = predict_alzheimers(test_audio)
print("Prediction:", result)

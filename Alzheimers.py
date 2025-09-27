import os
import numpy as np
import zipfile
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------------------
# Feature extraction: can return either time-independent (mean MFCCs)
# or time-dependent (MFCC sequence) depending on flag.
# -------------------------------------------------------------------
def extract_features(file_path, sr=22050, duration=5, n_mfcc=40, time_dependent=False, max_len=216):
    try:
        # Load audio (librosa handles resampling and trimming)
        y, sr = librosa.load(file_path, sr=sr, duration=duration)

        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        if time_dependent:
            # Pad/truncate to fixed length (max_len frames) for sequence models
            if mfccs.shape[1] < max_len:
                pad_width = max_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0,0),(0,pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :max_len]
            return mfccs.T  # shape: (time, n_mfcc)
        else:
            # Collapse across time → mean vector (time-independent)
            return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


# -------------------------------------------------------------------
# Load dataset: extract features and labels for all files
# -------------------------------------------------------------------
def load_dataset(dataset_path, time_dependent=False):
    features, labels = [], []

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if not file.endswith(('.wav', '.mp3')):
                continue
            file_path = os.path.join(folder_path, file)
            feature = extract_features(file_path, time_dependent=time_dependent)
            if feature is not None:
                features.append(feature)
                labels.append(folder)

    if len(features) == 0:
        raise ValueError("No valid features extracted. Check dataset and file types.")

    return np.array(features), np.array(labels)


# -------------------------------------------------------------------
# Choose mode: time-dependent (True) or time-independent (False)
# -------------------------------------------------------------------
TIME_DEPENDENT = True  # toggle this flag

dataset_path = "/content/dataset"
features, labels = load_dataset(dataset_path, time_dependent=TIME_DEPENDENT)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels_categorical, test_size=0.2, random_state=42)

# -------------------------------------------------------------------
# Define models
# -------------------------------------------------------------------
if TIME_DEPENDENT:
    # Model for sequential features (MFCC over time)
    model = Sequential([
        Masking(mask_value=0., input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(labels_categorical.shape[1], activation='softmax')
    ])
else:
    # Model for aggregated features (mean MFCCs)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(labels_categorical.shape[1], activation='softmax')
    ])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("alzheimers_audio_model.h5")


# -------------------------------------------------------------------
# Prediction function (uses the same TIME_DEPENDENT setting)
# -------------------------------------------------------------------
def predict_alzheimers(audio_file):
    feature = extract_features(audio_file, time_dependent=TIME_DEPENDENT)
    if feature is None:
        return {"error": "Feature extraction failed."}

    if TIME_DEPENDENT:
        feature = np.expand_dims(feature, axis=0)  # shape: (1, time, n_mfcc)
    else:
        feature = np.expand_dims(feature, axis=0)  # shape: (1, n_mfcc)

    prediction = model.predict(feature)
    label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return {"label": label, "likelihood": prediction[0]}


import os
import statistics
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
from sklearn.preprocessing import MinMaxScaler
import re


# ------------------- Acoustic Features -------------------
def Acoustic_features(voiceID, f0min=75, f0max=500, unit="Hertz"):
    sound = parselmouth.Sound(voiceID)
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    meanF0 = call(pitch, "Get mean", 0 ,0, unit)

    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return (
        localabsoluteJitter, localJitter, rapJitter, ddpJitter,
        localdbShimmer, localShimmer, apq3Shimmer, aqpq5Shimmer,
        hnr, meanF0
    )


# ------------------- Formant Features -------------------
def average_formant_frequency(voice_id, f0min=75, f0max=500):
    try:
        sound = parselmouth.Sound(voice_id)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
        numPoints = call(pointProcess, "Get number of points")

        f1_list, f2_list, f3_list, f4_list = [], [], [], []

        for point in range(1, numPoints + 1):
            t = call(pointProcess, "Get time from index", point)
            f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
            f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
            f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
            f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')

            if not np.isnan(f1): f1_list.append(f1)
            if not np.isnan(f2): f2_list.append(f2)
            if not np.isnan(f3): f3_list.append(f3)
            if not np.isnan(f4): f4_list.append(f4)

        if not all([f1_list, f2_list, f3_list, f4_list]):
            return None

        return statistics.median(f1_list + f2_list + f3_list + f4_list) / 4
    except Exception as e:
        print(f"[Formant Error] {voice_id}: {e}")
        return None


# ------------------- MFCC Features -------------------
def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)


# ------------------- Combine Features -------------------
def extract_features(path, filename):
    features = {}
    try:
        jitter_vals = Acoustic_features(path)
        formant = average_formant_frequency(path)
        mfccs = extract_mfcc(path)

        features.update({
            "path": path,
            "localabsoluteJitter": jitter_vals[0],
            "localJitter": jitter_vals[1],
            "rapJitter": jitter_vals[2],
            "ddpJitter": jitter_vals[3],
            "localdbShimmer": jitter_vals[4],
            "localShimmer": jitter_vals[5],
            "apq3Shimmer": jitter_vals[6],
            "aqpq5Shimmer": jitter_vals[7],
            "hnr": jitter_vals[8],
            "pitch": jitter_vals[9],
            "FundamentalFrequency": formant,
            "audio_id": filename.split("_")[0]
        })

        for i in range(len(mfccs)):
            features[f"MFCC{i}"] = mfccs[i]

        return features

    except Exception as e:
        print(f"[Error Processing {filename}]: {e}")

def scale(df):
    scaler = MinMaxScaler()
    numeric_cols = df.columns[1:-1] 
    df[numeric_cols] = df[numeric_cols].astype(float)
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# ------------------- Batch Process Folder -------------------
def process_folder(folder, save_dir=os.getenv("ACOUSTIC_FEATURES_DIR_PATH"), test=False):


    columns = [
        "path", "localabsoluteJitter", "localJitter", "rapJitter", "ddpJitter",
        "localdbShimmer", "localShimmer", "apq3Shimmer", "aqpq5Shimmer",
        "hnr", "pitch", "FundamentalFrequency"
    ] + [f"MFCC{i}" for i in range(13)] + ["audio_id"]

    rows = []  # collect feature dicts here

    for entry in os.listdir(folder):
        entry_path = os.path.join(folder, entry)

        if os.path.isdir(entry_path):
            # Training-style: process subfolder
            for filename in os.listdir(entry_path):
                if filename.endswith(".wav"):
                    path = os.path.join(entry_path, filename)
                    features = extract_features(path, filename)
                    if features:
                        rows.append({col: features.get(col, np.nan) for col in columns})

        elif test and entry.endswith(".wav"):
            # Test-style: wavs directly in folder
            features = extract_features(entry_path, entry)
            if features:
                rows.append({col: features.get(col, np.nan) for col in columns})

    if not rows:
        print(f"⚠️ No features extracted from {folder}")
        return

    # Build DataFrame once
    df = pd.DataFrame(rows, columns=columns)

    # Handle NaNs
    for col in df.columns[df.isna().any()].tolist():
        df[col] = df.groupby("audio_id")[col].transform(lambda x: x.fillna(x.median()))
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Normalize
    scale(df)

    # Save
    save_path = os.getenv("TMP_FEATURES_PATH") if test else os.path.join(save_dir, f"{os.path.basename(folder)}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
   

# ==============================
# Person ID Extractor
# ==============================
def extract_person_id(filepath):
    filename = os.path.basename(filepath).replace(' ', '')
    match = re.search(r'(.+?)_seg', filename)
    if match:
        full_id_string = match.group(1)
        if len(full_id_string) >= 5:
            return full_id_string[3:-2]
    return None
# ==============================

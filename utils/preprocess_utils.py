import os
import statistics
import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt
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
    df.loc[:, df.columns[1:-1]] = scaler.fit_transform(df.loc[:, df.columns[1:-1]])


# ------------------- Batch Process Folder -------------------
def process_folder(folder, save_dir=os.getenv("ACOUSTIC_FEATURES_DIR_PATH"),test=False):
    print(f"Processing: {folder}")

    columns = [
        "path", "localabsoluteJitter", "localJitter", "rapJitter", "ddpJitter",
        "localdbShimmer", "localShimmer", "apq3Shimmer", "aqpq5Shimmer",
        "hnr", "pitch", "FundamentalFrequency"
    ] + [f"MFCC{i}" for i in range(13)] + ["audio_id"]

    df = pd.DataFrame(columns=columns)

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".wav"):
                path = os.path.join(subfolder_path, filename)
                features = extract_features(path, filename)
                if features:
                    df.loc[len(df)] = [features.get(col, np.nan) for col in columns]

    # Handle NaNs (group-wise fill then column median)
    for col in df.columns[df.isna().any()].tolist():
        df[col] = df.groupby("audio_id")[col].transform(lambda x: x.fillna(x.median()))
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Normalize
    scale(df)
    # Save
    save_path = os.path.join(save_dir, f"{os.path.basename(folder)}.csv")
    if test:
        save_path =os.getenv("TMP_FEATURES_PATH")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

# ------------------- Spectrogram Parameters -------------------
n_fft = int(os.getenv("N_FFT", "2048"))
hop_length = int(os.getenv("HOP_LENGTH", "512"))
fmax = int(os.getenv("FMAX", "8000"))

# ------------------- Spectrogram Creation -------------------
def create_spectrogam(waveform, sample_rate=os.getenv("TARGET_SAMPLE_RATE","16000"), n_mels=os.getenv("N_MELS","128")):
    sr = sample_rate
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft, 
                                              hop_length=hop_length, n_mels=n_mels, fmax=fmax)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    mel_spec_tensor = torch.tensor(mel_spec_db).unsqueeze(0) 
    return mel_spec_tensor
    
# ------------------- Spectrogram Augmentation -------------------
def augment_spectrogram(waveform,time_mask_param=6, freq_mask_param=6):
    """Apply time masking & frequency masking on an audio waveform."""

    
    mel_spec_tensor = create_spectrogam(waveform)

    time_mask = T.TimeMasking(time_mask_param)
    freq_mask = T.FrequencyMasking(freq_mask_param)

    time_masked = time_mask(mel_spec_tensor)
    freq_masked = freq_mask(mel_spec_tensor)

    min_db = -80
    time_masked[time_masked == 0] = min_db
    freq_masked[freq_masked == 0] = min_db
    combined_masked = torch.minimum(time_masked, freq_masked)
    combined_masked[combined_masked == 0] = min_db

    return {
        "original": mel_spec_tensor,
        "time_masked": time_masked,
        "freq_masked": freq_masked,
        "combined": combined_masked
    }
# ------------------- Spectrogram saving -------------------
def save_spectrogram(spectrogram, file_path, sr, cmap='magma'):
    """Save spectrogram as an ultra-high-quality image."""
    plt.figure(figsize=(10, 10), dpi=600)  # Ultra-high resolution
    mel_spec_db_numpy = spectrogram.squeeze().detach().cpu().numpy()

    librosa.display.specshow(mel_spec_db_numpy, sr=sr, hop_length=hop_length,
                             x_axis=None, y_axis=None, cmap=cmap, fmax=fmax)

    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()



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

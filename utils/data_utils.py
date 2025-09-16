import os
import numpy as np
import pandas as pd
import torchaudio
from utils.segment_utils import segment_audio


# ------------------- Load 1D Data -------------------
def load_train_features(path):
    df = pd.read_csv(path)
    df["label"] = df["path"].apply(lambda x: x.split("/")[-2])
    df["binary_label"] = df["label"].apply(lambda x: 0 if x.lower() == "healthy" else 1)
    df["audio_id"] = df["path"].apply(lambda x: os.path.basename(x).split('_')[0][3:])  # group by audio

    features = [
        'localabsoluteJitter', 'localJitter', 'rapJitter', 'ddpJitter',
        'localdbShimmer', 'localShimmer', 'apq3Shimmer', 'aqpq5Shimmer',
        'hnr', 'FundamentalFrequency'
    ] + [f"MFCC{i}" for i in range(13)]

    X = df[features].values.astype(np.float32)
    y = df["binary_label"].values.astype(np.int32)
    groups = df["audio_id"].values

    print(f"Labels in {path}: {df['label'].unique()}")
    return X, y, groups

def load_test_features(path):
    df = pd.read_csv(path)
    features = [
        'localabsoluteJitter', 'localJitter', 'rapJitter', 'ddpJitter',
        'localdbShimmer', 'localShimmer', 'apq3Shimmer', 'aqpq5Shimmer',
        'hnr', 'FundamentalFrequency'
    ] + [f"MFCC{i}" for i in range(13)]

    X = df[features].values.astype(np.float32)
    return X

def segment_input(audio_path):
    segments = segment_audio(audio_path)
    for i, segment in enumerate(segments):
        try:
            os.makedirs("tmp/audio", exist_ok=True)
            segment_path = f"tmp/audio/segment_{i}.wav"
            torchaudio.save(segment_path, segment, 16000)
        except Exception as e:
            print(f"Error processing segment {i}: {e}")
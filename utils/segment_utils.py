# segment_utils.py
import os
import torch
import torchaudio
import torchaudio.transforms as T

# Global configs
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))
TARGET = int(os.getenv("TARGET_DURATION", "1"))


def segment_audio(audio_path, target_sample_rate=TARGET_SAMPLE_RATE, target=TARGET):
    """
    Splits an audio file into overlapping segments with optional silence trimming.

    Args:
        audio_path (str): Path to input .wav file
        target_sample_rate (int): Resampling rate
        target : Segment durations (in seconds)

    Returns:
        list[Tensor]: List of audio segments
    """
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    # Trim silence from start/end
    waveform = T.Vad(sample_rate=sample_rate)(waveform)

    segments = []
    target_duration = target * target_sample_rate
    step = target_duration // 2  # 50% overlap
    num_segments = max(1, (waveform.size(1) - target_duration) // step + 1)

    for i in range(num_segments):
        start = i * step
        end = start + target_duration
        segment = waveform[:, start:end]

        # Pad or truncate
        if segment.size(1) < target_duration:
            pad_amount = target_duration - segment.size(1)
            segment = torch.nn.functional.pad(segment, (0, pad_amount))
        elif segment.size(1) > target_duration:
            segment = segment[:, :target_duration]

        segments.append(segment)
    return segments


def process_healthy(path, save_dir="/media/data/1S/1AS/healthy"):
    """
    Process healthy control dataset, segment audio, and save.
    """
    os.makedirs(save_dir, exist_ok=True)

    for subfolder in os.listdir(path):
        subfolder_path = os.path.join(path, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing healthy: {subfolder}")
            for file in os.listdir(subfolder_path):
                if file.endswith(".wav") and file.startswith("V"):
                    file_path = os.path.join(subfolder_path, file)
                    segments = segment_audio(file_path)
                    for i, segment in enumerate(segments):
                        segment_path = os.path.join(save_dir, f"{file[:-4]}_seg{i}.wav")
                        torchaudio.save(segment_path, segment, TARGET_SAMPLE_RATE)


def process_parkinsons(path, save_base=os.getenv("SEGMENTS_PATH")):
    """
    Process Parkinson's dataset, segment audio, and save into 'mild'/'severe'.
    """
    for subfolder in os.listdir(path):
        if subfolder in ["1-5", "6-10"]:
            class_name = "mild"
        elif subfolder in ["11-16", "17-28"]:
            class_name = "severe"
        else:
            continue

        subfolder_path = os.path.join(path, subfolder)
        for sub_subfolder in os.listdir(subfolder_path):
            sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)
            if os.path.isdir(sub_subfolder_path):
                print(f"Processing Parkinson's {class_name}: {sub_subfolder}")
                for file in os.listdir(sub_subfolder_path):
                    if file.endswith(".wav") and file.startswith("V"):
                        file_path = os.path.join(sub_subfolder_path, file)
                        segments = segment_audio(file_path)
                        save_dir = os.path.join(save_base, class_name)
                        os.makedirs(save_dir, exist_ok=True)
                        for i, segment in enumerate(segments):
                            segment_path = os.path.join(save_dir, f"{file[:-4]}_seg{i}.wav")
                            torchaudio.save(segment_path, segment, TARGET_SAMPLE_RATE)


def run_segmentation(healthy_path, parkinson_path):
    process_healthy(healthy_path)
    process_parkinsons(parkinson_path)

import os
import re
import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import Resample
from torch.utils.data import Dataset


# ==============================
# Dataset
# ==============================
class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, sample_rate=16000):
        self.root_dir = root_dir
        self.transform = transform
        self.sample_rate = sample_rate
        self.audio_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for label in self.classes:
            class_dir = os.path.join(root_dir, label)
            for file in os.listdir(class_dir):
                if file.endswith((".wav", ".mp3")):
                    self.audio_paths.append(os.path.join(class_dir, file))
                    self.labels.append(self.class_to_idx[label])

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        else:
            waveform = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform

        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        label = self.labels[idx]
        return waveform, label


def get_class_distribution(dataset):
    class_counts = {}
    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts


def load_dataset(dataset_path):
    dataset = AudioDataset(root_dir=dataset_path, transform=None)
    print(f"Loaded {len(dataset)} samples from {dataset_path}")
    return dataset


# ==============================
# ResSE Block + SampleCNN
# ==============================
class ResSEBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, dropout=0.2, pool_kernel=3, pool_stride=3):
        super(ResSEBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # SE block
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(out_channels, out_channels // reduction)
        self.fc2 = nn.Linear(out_channels // reduction, out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Downsample if channels mismatch
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # SE
        w = self.global_pool(out).squeeze(-1)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        w = w.unsqueeze(-1)

        out = out * w
        residual = self.downsample(residual) 
        out = self.relu(out + residual)
        out = self.pool(out)

        return out


def create_ReSE(num_classes, n = 1536):
    class SampleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SampleCNN, self).__init__()

            self.initial_conv = nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=3, stride=3, bias=False),
                nn.ReLU()
            )

            # ResSE Blocks
            self.res_se_block1 = ResSEBlock1D(128, 128)
            self.res_se_block2 = ResSEBlock1D(128, 256)
            self.res_se_block3 = ResSEBlock1D(256, 256)
            self.res_se_block4 = ResSEBlock1D(256, 256)
            self.res_se_block5 = ResSEBlock1D(256, 256)
            self.res_se_block6 = ResSEBlock1D(256, 512)
            self.res_se_block7 = ResSEBlock1D(512, 512)

            self.final_conv_block = nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=1, stride=1, bias=False),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            
            self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)
            
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.initial_conv(x)

            x = self.res_se_block1(x)
            x = self.res_se_block2(x)
            x = self.res_se_block3(x)
            x = self.res_se_block4(x)
            x = self.res_se_block5(x)
            
            x = self.res_se_block6(x)
            pooled_level1 = self.adaptive_max_pool(x).squeeze(-1)

            x = self.res_se_block7(x)
            pooled_level2 = self.adaptive_max_pool(x).squeeze(-1)

            x = self.final_conv_block(x)
            pooled_level3 = self.adaptive_max_pool(x).squeeze(-1)

            x = torch.cat((pooled_level1, pooled_level2, pooled_level3), dim=1)
            x = self.fc(x)
            return x

    return SampleCNN(num_classes)


# ==============================
# Person ID extractor
# ==============================
def extract_person_id(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'(.+?)_seg', filename) 
    if match:
        full_id_string = match.group(1)
        if len(full_id_string) >= 5:
            return full_id_string[3:-2]
    return None

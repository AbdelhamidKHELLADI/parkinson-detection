
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets

# ==============================
# Hyperparams / Transforms
# ==============================
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=3)
])

# ==============================
# Dataset Loader
# ==============================
def load_dataset(dataset_path):
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return dataset


# ==============================
# CBAM Components
# ==============================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2)

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        conv = conv.repeat(1, x.size()[1], 1, 1)
        att = torch.sigmoid(conv)
        return att

    def agg_channel(self, x, pool="max"):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(0, 2, 1)
        if pool == "max":
            x = F.max_pool1d(x, c)
        elif pool == "avg":
            x = F.avg_pool1d(x, c)
        x = x.permute(0, 2, 1).view(b, 1, h, w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        middle = int(n_channels_in / float(reduction_ratio))
        self.bottleneck = nn.Sequential(
            nn.Linear(n_channels_in, middle),
            nn.ReLU(),
            nn.Linear(middle, n_channels_in)
        )

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel).view(x.size()[0], -1)
        max_pool = F.max_pool2d(x, kernel).view(x.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck
        sig_pool = torch.sigmoid(pool_sum).unsqueeze(2).unsqueeze(3)
        out = sig_pool.repeat(1, 1, kernel[0], kernel[1])
        return out


class CBAM(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        fp = self.channel_attention(f) * f
        fpp = self.spatial_attention(fp) * fp
        return fpp


# ==============================
# VGG16 + CBAM Classifier
# ==============================
class VGGCBAMClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VGGCBAMClassifier, self).__init__()
        base_model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        features = list(base_model.features)

        # Freeze original VGG conv layers
        for param in base_model.features.parameters():
            param.requires_grad = False

        # Insert CBAM after each block
        cbam1 = CBAM(64, reduction_ratio=16, kernel_size=7)
        cbam2 = CBAM(128, reduction_ratio=16, kernel_size=7)
        cbam3 = CBAM(256, reduction_ratio=16, kernel_size=7)
        cbam4 = CBAM(512, reduction_ratio=16, kernel_size=7)
        cbam5 = CBAM(512, reduction_ratio=16, kernel_size=7)

        modified_features_list = []
        current_idx = 0

        # Block 1
        modified_features_list.extend(features[current_idx:6])
        modified_features_list.append(cbam1)
        current_idx = 6

        # Block 2
        modified_features_list.extend(features[current_idx:13])
        modified_features_list.append(cbam2)
        current_idx = 13

        # Block 3
        modified_features_list.extend(features[current_idx:23])
        modified_features_list.append(cbam3)
        current_idx = 23

        # Block 4
        modified_features_list.extend(features[current_idx:33])
        modified_features_list.append(cbam4)
        current_idx = 33

        # Block 5
        modified_features_list.extend(features[current_idx:])
        modified_features_list.append(cbam5)

        self.features = nn.Sequential(*modified_features_list)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


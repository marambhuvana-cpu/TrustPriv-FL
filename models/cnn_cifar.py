import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFARCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self._init_args = (in_channels, num_classes)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # First convolution sequence
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolution sequence
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection adjustment
        self.skip_connection = None
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # First convolution sequence
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Second convolution sequence
        out = self.conv2(out)
        out = self.bn2(out)

        # Adjust skip connection if needed
        if self.skip_connection is not None:
            identity = self.skip_connection(x)

        # Add skip connection and apply ReLU
        out += identity
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.resblock1 = ResBlock(64, 64, stride=1)
        self.resblock2 = ResBlock(64, 128, stride=2)
        self.resblock3 = ResBlock(128, 256, stride=2)
        self.resblock4 = ResBlock(256, 512, stride=2)

        # Global average pooling and final layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet blocks
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        # Global average pooling and classification
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
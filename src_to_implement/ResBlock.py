import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

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

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip_connection is not None:
            identity = self.skip_connection(x)

        out += identity
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Mehr ResBlocks für bessere Feature-Extraktion
        self.resblock1_1 = ResBlock(64, 64, stride=1)
        self.resblock1_2 = ResBlock(64, 64, stride=1)
        
        self.resblock2_1 = ResBlock(64, 128, stride=2)
        self.resblock2_2 = ResBlock(128, 128, stride=1)
        
        self.resblock3_1 = ResBlock(128, 256, stride=2)
        self.resblock3_2 = ResBlock(256, 256, stride=1)
        
        self.resblock4_1 = ResBlock(256, 512, stride=2)
        self.resblock4_2 = ResBlock(512, 512, stride=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        
        # Erweiterte Classifier-Schicht
        self.fc1 = nn.Linear(512, 256)
        self.fc_dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.resblock1_1(x)
        x = self.resblock1_2(x)
        
        x = self.resblock2_1(x)
        x = self.resblock2_2(x)
        
        x = self.resblock3_1(x)
        x = self.resblock3_2(x)
        
        x = self.resblock4_1(x)
        x = self.resblock4_2(x)

        x = self.global_avg_pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x
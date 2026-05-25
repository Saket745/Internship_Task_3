import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """
    ResNet architecture for EMNIST (62 classes: ByClass Split).
    Input: (1, 28, 28)
    """
    def __init__(self, block, num_blocks, num_classes=62):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Initial layer for 1-channel 28x28 images
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out

def ResNet9(num_classes=62):
    """ResNet-9 is often superior for smaller 28x28 images."""
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)

def ResNet18(num_classes=62):
    """Standard ResNet-18."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def CharacterCNN(num_classes=62, version='resnet18'):
    """
    Factory function for character recognition model.
    Defaults to ResNet-18 for robustness.
    """
    if version.lower() == 'resnet9':
        return ResNet9(num_classes=num_classes)
    return ResNet18(num_classes=num_classes)

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Block 1: 1 → 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # H/2, W/2
        
        # Block 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # H/4, W/4
        
        # Block 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 1))  # H/8, W/4 (vertical pool only)
        
        # Block 4: 128 → 256 channels
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 1))  # H/16, W/4
        
        # Block 5: 256 → 512 channels
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        x = self.pool1(self.bn1(F.relu(self.conv1(x))))     # (B, 32, 16, 64)
        x = self.pool2(self.bn2(F.relu(self.conv2(x))))     # (B, 64, 8, 32)
        x = self.pool3(self.bn3(F.relu(self.conv3(x))))     # (B, 128, 4, 32)
        x = self.pool4(self.bn4(F.relu(self.conv4(x))))     # (B, 256, 2, 32)
        x = self.bn5(F.relu(self.conv5(x)))                 # (B, 512, 2, 32)
        
        assert x.shape[1] == 512, f"Expected 512 channels, got {x.shape[1]}"
        return x

class BiLSTMSequenceEncoder(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2):
        super().__init__()
        self.input_size = 2 * 512  # 1024
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0
        )
    
    def forward(self, features):
        batch_size = features.shape[0]
        # Reshape: (B, 512, 2, 32) → (B, 32, 1024)
        features_reshaped = features.permute(0, 3, 1, 2)  # (B, 32, 512, 2)
        features_reshaped = features_reshaped.contiguous().view(
            batch_size, features.shape[3], -1
        )  # (B, 32, 1024)
        
        assert features_reshaped.shape == (batch_size, 32, 1024)
        
        lstm_out, _ = self.lstm(features_reshaped)
        assert lstm_out.shape == (batch_size, 32, 2 * self.hidden_size)
        return lstm_out

class ClassificationHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, lstm_out):
        x = self.dropout(F.relu(self.fc1(lstm_out)))  # (B, T, 256)
        logits = self.fc2(x)                          # (B, T, num_classes)
        
        expected_shape = (lstm_out.shape[0], lstm_out.shape[1], self.num_classes)
        assert logits.shape == expected_shape
        return logits

class CRNNModel(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()
        self.num_classes = num_classes
        self.cnn = CNNFeatureExtractor()
        self.lstm = BiLSTMSequenceEncoder(hidden_size=256, num_layers=2)
        self.head = ClassificationHead(num_classes)
    
    def forward(self, x):
        features = self.cnn(x)              # (B, 512, 2, 32)
        lstm_out = self.lstm(features)      # (B, 32, 512)
        logits = self.head(lstm_out)        # (B, 32, num_classes)
        return logits

if __name__ == "__main__":
    print("Testing ResNet...")
    num_classes = 62
    model = CharacterCNN(num_classes=num_classes)
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    assert output.shape == (1, num_classes)
    print("ResNet architecture verified.")
    
    print("\nTesting CRNN...")
    crnn = CRNNModel(num_classes=37)
    dummy_crnn_input = torch.randn(4, 1, 32, 128)
    crnn_output = crnn(dummy_crnn_input)
    print(f"Input Shape: {dummy_crnn_input.shape}")
    print(f"Output Shape: {crnn_output.shape}")
    assert crnn_output.shape == (4, 32, 37)
    print("CRNN architecture verified.")

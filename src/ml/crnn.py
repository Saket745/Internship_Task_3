import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    """
    Extract spatial features from image
    Output shape: (batch, 512, height_reduced, width_reduced)
    
    Architecture:
    - Conv2D filters follow pattern: 32 → 64 → 128 → 256 → 512
    - All kernels: 3×3, padding=1 (preserve spatial dims)
    - Activation: ReLU after each conv
    - MaxPool: 2×2 after specific layers to reduce spatial dims
    - Batch normalization: After each conv layer
    """
    
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
        """
        Args:
            x: (batch, 1, 32, 128)
        Returns:
            features: (batch, 512, 2, 32)
        """
        x = self.pool1(self.bn1(F.relu(self.conv1(x))))     # (B, 32, 16, 64)
        x = self.pool2(self.bn2(F.relu(self.conv2(x))))     # (B, 64, 8, 32)
        x = self.pool3(self.bn3(F.relu(self.conv3(x))))     # (B, 128, 4, 32)
        x = self.pool4(self.bn4(F.relu(self.conv4(x))))     # (B, 256, 2, 32)
        x = self.bn5(F.relu(self.conv5(x)))                 # (B, 512, 2, 32)
        
        # Validation
        assert x.shape[1] == 512, f"Expected 512 channels, got {x.shape[1]}"
        
        return x

class BiLSTMSequenceEncoder(nn.Module):
    """
    Encode spatial features into sequence representation
    """
    
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
        
        # Reshape: collapse height dimension
        features_reshaped = features.permute(0, 3, 1, 2)  # (B, 32, 512, 2)
        features_reshaped = features_reshaped.contiguous().view(
            batch_size, features.shape[3], -1
        )  # (B, 32, 1024)
        
        assert features_reshaped.shape == (batch_size, 32, 1024)
        
        lstm_out, (h_n, c_n) = self.lstm(features_reshaped)
        
        assert lstm_out.shape == (batch_size, 32, 2 * self.hidden_size)
        
        return lstm_out

class ClassificationHead(nn.Module):
    """
    Dense layers to predict character class per timestep
    """
    
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
    """
    Complete CRNN: CNN + BiLSTM + Dense
    """
    
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
    
    def get_model_summary(self):
        print(self)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

"""
Parazit Yumurta Tespiti - Model Mimarisi (PyTorch Versiyonu)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ParasiteCNN(nn.Module):
    """CNN modeli"""
    
    def __init__(self, num_classes):
        super(ParasiteCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_cnn_model(num_classes):
    """CNN modeli oluşturur"""
    return ParasiteCNN(num_classes)


def create_transfer_learning_model(num_classes):
    """Transfer Learning modeli oluşturur (ResNet18)"""
    model = models.resnet18(pretrained=True)
    
    # Son katmanı değiştir
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    
    return model


def get_model_summary(model, input_size=(3, 224, 224)):
    """Model özetini yazdırır"""
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nToplam parametre sayısı: {total_params:,}")
    print(f"Eğitilebilir parametre sayısı: {trainable_params:,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_size)
        output = model(dummy_input)
        print(f"Çıkış boyutu: {output.shape}")







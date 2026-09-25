import torch
import torch.nn as nn
from torchvision import models

class FaceCNNModel(nn.Module):
    def __init__(self, num_classes=20, dropout_rate=0.4):
        super(FaceCNNModel, self).__init__()
        # Load CNN Backbone ResNet18 Pretrained trên ImageNet
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Đóng băng các tầng Convolutional ban đầu để học đặc trưng nét mặt nhanh hơn
        for param in list(self.backbone.parameters())[:-15]:
            param.requires_grad = False

        # Thay thế Classifier Head cho 20 class
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
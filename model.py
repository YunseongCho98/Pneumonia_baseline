import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class PneumoniaClassifier(nn.Module):
    def __init__(self, weights=ResNet18_Weights.DEFAULT):
        super().__init__()
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)

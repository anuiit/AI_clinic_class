# model.py
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def build_resnet18_multilabel(num_labels: int, dropout: float, hidden: int) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    
    if hidden == 0:
        model.fc = nn.Sequential( # type: ignore
        nn.Dropout(dropout),
        nn.Linear(in_features, num_labels)
    )
    else:
        model.fc = nn.Sequential(  # type: ignore
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels)
        )
    return model

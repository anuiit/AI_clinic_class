import torch
import torch.nn as nn
from typing import Optional, Type
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

class MultiLabelModel(nn.Module):
    def __init__(
        self,
        base_model: str,
        num_labels: int,
        dropout: float,
        hidden: list[int] = [],
        batch_size: int = 16,
        activation: nn.Module = nn.ReLU(inplace=True),
        normalization: Optional[Type[nn.Module]] = None,
        custom_head: bool = False,
        use_bottleneck: bool = False,
        bottleneck_dim: int = 1024
    ):
        """
        Args:
            base_model: Name of the base model architecture
            num_labels: Number of output labels
            dropout: Dropout rate
            hidden: List of hidden layer dimensions
            batch_size: Batch size (for reference)
            activation: Activation module instance
            normalization: Normalization layer class (e.g., nn.BatchNorm1d, nn.LayerNorm)
            custom_head: Whether to use legacy custom head architecture
            use_bottleneck: Whether to use bottleneck for dimensionality reduction
            bottleneck_dim: Dimension for bottleneck layer (only used with improved_head=True)
        """
        super(MultiLabelModel, self).__init__()

        self.base_model_name = base_model
        self.base_model = self.get_base_model(base_model)

        if 'resnet' in base_model.lower():
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif 'vgg' in base_model.lower():
            # VGG's classifier is Sequential, get last Linear layer
            in_features = self.base_model.classifier[-1].in_features
            self.base_model.classifier = nn.Identity()
        elif 'efficientnet' in base_model.lower():
            # EfficientNet's classifier is Sequential: [Dropout, Linear]
            # Access the Linear layer at index 1
            in_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        elif 'vit' in base_model.lower():
            in_features = self.base_model.head.in_features
            self.base_model.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported model architecture: {base_model}")

        if custom_head:
            self.head = build_custom_head_multilabel(in_features, num_labels)
        else:
            self.head = build_head_multilabel_v2(
                in_features=in_features,
                num_labels=num_labels,
                dropout=dropout,
                hidden=hidden,
                activation=activation,
                normalization=normalization,
                use_bottleneck=use_bottleneck,
                bottleneck_dim=bottleneck_dim
            )

    def forward(self, x):
        x = self.base_model(x)
        x = self.head(x)
        return x

    def freeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = True

    def get_base_model(self, base_model: str):
        model_dict = {
            "resnet18": build_resnet18_multilabel,
            "resnet50": build_resnet50_multilabel,
            "vgg16": build_vgg16_multilabel,
            "vit_b16": build_vit_b16_multilabel,
            "efficientnet_b7": build_efficientnet_b7_multilabel,
            "efficientnet_b0": build_efficientnet_b0_multilabel,
            "efficientnet_b4": build_efficientnet_b4_multilabel,
            "efficientnet_b3": build_efficientnet_b3_multilabel,
        }
        return model_dict[base_model]()
    
    def print_trainable_status(self):
        total_params = 0
        trainable_params = 0
        
        print("\n" + "="*60)
        print("Backbone (feature extractor):")
        backbone_trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        backbone_total = sum(p.numel() for p in self.base_model.parameters())
        print(f"  Trainable: {backbone_trainable:,} / {backbone_total:,}")
        
        print("\nHead (classifier):")
        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        head_total = sum(p.numel() for p in self.head.parameters())
        print(f"  Trainable: {head_trainable:,} / {head_total:,}")
        
        total_params = backbone_total + head_total
        trainable_params = backbone_trainable + head_trainable
        
        print("\nTotal:")
        print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
        print("="*60 + "\n")

def build_resnet18_multilabel() -> nn.Module:
    base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    return base_model

def build_resnet50_multilabel() -> nn.Module:
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    return base_model

def build_vgg16_multilabel() -> nn.Module:
    base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    return base_model

def build_vit_b16_multilabel() -> nn.Module:
    base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    return base_model

def build_efficientnet_b7_multilabel() -> nn.Module:
    from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
    base_model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)

    return base_model

def build_efficientnet_b3_multilabel() -> nn.Module:
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    base_model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

    return base_model

def build_efficientnet_b0_multilabel() -> nn.Module:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    return base_model

def build_efficientnet_b4_multilabel() -> nn.Module:
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
    base_model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

    return base_model

def build_head_multilabel(
    in_features: int,
    num_labels: int, 
    dropout: float,
    hidden: list[int] = [],
    activation: Optional[nn.Module] = None,
    normalization: Optional[Type[nn.Module]] = None,
    use_bottleneck: bool = False,
    bottleneck_dim: int = 1024
) -> nn.Module:
    """
    Unified multi-label classification head builder with full customization.
    
    Args:
        in_features: Input feature dimension from backbone
        num_labels: Number of output labels
        dropout: Dropout rate
        hidden: List of hidden layer dimensions. If empty and use_bottleneck=True,
                defaults to [bottleneck_dim, bottleneck_dim//2]
        activation: Activation module instance (default: ReLU)
        normalization: Normalization layer class (e.g., nn.BatchNorm1d, nn.LayerNorm)
        use_bottleneck: Whether to add initial bottleneck for dimensionality reduction
        bottleneck_dim: Dimension for bottleneck layer (only if use_bottleneck=True)
    """
    layers_list = []
    input_size = in_features
    
    # Optional bottleneck layer for dimensionality reduction
    if use_bottleneck:
        layers_list.append(nn.Linear(input_size, bottleneck_dim))
        if normalization is not None:
            layers_list.append(normalization(bottleneck_dim))
        layers_list.append(activation if activation is not None else nn.ReLU(inplace=True))
        layers_list.append(nn.Dropout(dropout))
        input_size = bottleneck_dim
        
        # If no hidden layers specified, use default progressive reduction
        if len(hidden) == 0:
            hidden = [bottleneck_dim // 2]
    
    # Simple head: direct to output
    if len(hidden) == 0:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_labels)
        )
    
    # Build hidden layers
    for i, hidden_dim in enumerate(hidden):
        layers_list.append(nn.Linear(input_size, hidden_dim))
        
        if normalization is not None:
            layers_list.append(normalization(hidden_dim))
        
        layers_list.append(activation if activation is not None else nn.ReLU(inplace=True))
        
        # Progressive dropout: increases slightly for deeper layers
        dropout_rate = dropout * (1 + i * 0.5)
        layers_list.append(nn.Dropout(dropout_rate))
        
        input_size = hidden_dim
    
    # Final output layer
    layers_list.append(nn.Linear(input_size, num_labels))
    
    return nn.Sequential(*layers_list)


def build_head_multilabel_v2(
    in_features: int,
    num_labels: int, 
    dropout: float,
    hidden: list[int] = [],
    activation: Optional[nn.Module] = None,
    normalization: Optional[Type[nn.Module]] = None,
    use_bottleneck: bool = False,
    bottleneck_dim: int = 1024
) -> nn.Module:
    layers_list = []
    input_size = in_features
    
    # Bottleneck with lighter dropout
    if use_bottleneck:
        layers_list.append(nn.Linear(input_size, bottleneck_dim))
        if normalization is not None:
            layers_list.append(normalization(bottleneck_dim))
        layers_list.append(activation if activation is not None else nn.ReLU(inplace=True))
        layers_list.append(nn.Dropout(dropout * 0.5))
        input_size = bottleneck_dim
        
        if len(hidden) == 0:
            hidden = [bottleneck_dim // 2]
    
    if len(hidden) == 0:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_labels)
        )
    
    for i, hidden_dim in enumerate(hidden):
        layers_list.append(nn.Linear(input_size, hidden_dim))
        
        if normalization is not None:
            layers_list.append(normalization(hidden_dim))
        
        layers_list.append(activation if activation is not None else nn.ReLU(inplace=True))
        
        dropout_rate = dropout * (1 + i * 0.3)  # ← CHANGED from 0.5
        dropout_rate = min(dropout_rate, 0.5)   # ← CAP at 0.5
        layers_list.append(nn.Dropout(dropout_rate))
        
        input_size = hidden_dim
    
    if normalization is not None and len(hidden) > 0:
        layers_list.append(normalization(input_size))
    
    layers_list.append(nn.Linear(input_size, num_labels))
    
    return nn.Sequential(*layers_list)

def build_custom_head_multilabel(in_features: int, num_labels: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_labels)
    )

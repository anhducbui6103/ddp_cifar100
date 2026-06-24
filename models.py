import torch.nn as nn
import timm


def build_model(model_name: str = "resnet50", num_classes: int = 100, pretrained: bool = True) -> nn.Module:
    """
    Build a CIFAR-100 classifier using timm.

    Recommended models:
    - resnet50: safest and fastest
    - convnext_tiny: better for report, heavier than ResNet-50
    - vit_base_patch16_224: heavier, only use if GPU is strong
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model

import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18_scratch(num_classes: int = 500) -> nn.Module:
    """
    Build a randomly initialized ResNet-18 model.

    Args:
        num_classes: Number of output classes.

    Returns:
        A ResNet-18 model with a 500-class output layer.
    """
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
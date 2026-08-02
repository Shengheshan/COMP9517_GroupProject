# Member 4: Transfer learning model definitions
# This file implements ImageNet-pretrained ResNet-18 for:
# 1. Frozen-backbone training
# 2. Full-network fine-tuning
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def create_resnet18(num_classes=500, mode="frozen"):
    if mode not in {"frozen", "finetune"}:
        raise ValueError("mode must be 'frozen' or 'finetune'")

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    if mode == "frozen":
        for parameter in model.parameters():
            parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_trainable_parameters(model):
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
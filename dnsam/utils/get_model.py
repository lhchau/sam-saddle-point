import sys

from ..models import *

def get_model(cfg, num_classes):
    if cfg["model"]["architecture"] == "resnet18":
        return ResNet18(num_classes=num_classes)
    elif cfg["model"]["architecture"] == "resnet34":
        return ResNet34(num_classes=num_classes)
    elif cfg["model"]["architecture"] == "resnet50":
        return ResNet50(num_classes=num_classes)
    elif cfg["model"]["architecture"] == "resnet101":
        return ResNet101(num_classes=num_classes)
    elif cfg["model"]["architecture"] == "resnet152":
        return ResNet152(num_classes=num_classes)
    elif cfg["model"]["architecture"] == "wrn28_10":
        return WideResnet28_10(num_classes=num_classes)
    elif cfg["model"]["architecture"] == "resnet18_noshort":
        return ResNet18_noshort(num_classes=num_classes)
    elif cfg["model"]["architecture"] == "resnet18_mnist":
        return ResNet18_MNIST(num_classes=num_classes)
    elif cfg["model"]["architecture"] == "resnet34_mnist":
        return ResNet34_MNIST(num_classes=num_classes)
    elif cfg["model"]["architecture"] == "resnet50_mnist":
        return ResNet50_MNIST(num_classes=num_classes)
    elif cfg["model"]["architecture"] == "wrn28_10_mnist":
        return WideResnet28_10_MNIST(num_classes=num_classes)
    else:
        raise ValueError("Invalid model!!!")
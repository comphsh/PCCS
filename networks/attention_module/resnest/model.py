import timm
import torch.nn as nn


def resnest50(n_classes=1000):
    model = timm.create_model('resnest50d', pretrained=False)
    model.fc = nn.Linear(2048, n_classes)
    return model

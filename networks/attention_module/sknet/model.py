import timm
import torch.nn as nn


def sknet50(n_classes=1000):
    model = timm.create_model('skresnet50', pretrained=False)
    model.fc = nn.Linear(2048, n_classes)
    return model


if __name__ == '__main__':
    import torch
    net = sknet50()
    net(torch.zeros(32, 3, 224, 224))
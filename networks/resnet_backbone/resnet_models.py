from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from collections import OrderedDict
import torch.nn as nn


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=False, bn_type=None,origional_in_channels=3):
        super(ResNet, self).__init__()
        self.inplanes = 128 if deep_base else 64
        if deep_base:
            self.resinit = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(origional_in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=False)),
                ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=False)),
                ('conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn3', nn.BatchNorm2d(self.inplanes)),
                ('relu3', nn.ReLU(inplace=False))]
            ))
        else:
            self.resinit = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(origional_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(self.inplanes)),
                ('relu1', nn.ReLU(inplace=False))]
            ))


        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change.
        # layers = [3, 4, 6, 3]
        self.layer1 = self._make_layer(block, 64, layers[0], bn_type=bn_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bn_type=bn_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bn_type=bn_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bn_type=bn_type)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # block.expansion = 4
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, bn_type=None):
        # plane=[3,4,6,3]
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, bn_type=bn_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_type=bn_type))

        return nn.Sequential(*layers)

    def forward(self, x):  #这个是分类forward推理
        x = self.resinit(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class NormalResnetBackbone(nn.Module):
    def __init__(self, orig_resnet):
        super(NormalResnetBackbone, self).__init__()

        self.num_features = 2048
        # take pretrained resnet, except AvgPool and FC
        self.resinit = orig_resnet.resinit
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        tuple_features = list()
        x = self.resinit(x)

        tuple_features.append(x)
        x = self.maxpool(x)

        tuple_features.append(x)
        x = self.layer1(x)

        tuple_features.append(x)
        x = self.layer2(x)

        tuple_features.append(x)
        x = self.layer3(x)

        tuple_features.append(x)
        x = self.layer4(x)

        tuple_features.append(x)

        return tuple_features


class DilatedResnetBackbone(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8, multi_grid=(1, 2, 4)):
        super(DilatedResnetBackbone, self).__init__()

        self.num_features = 2048
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            if multi_grid is None:
                orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnet.layer4[i].apply(partial(self._nostride_dilate, dilate=int(4 * r)))

        elif dilate_scale == 16:
            if multi_grid is None:
                orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
            else:
                for i, r in enumerate(multi_grid):
                    orig_resnet.layer4[i].apply(partial(self._nostride_dilate, dilate=int(2 * r)))

        # Take pretrained resnet, except AvgPool and FC
        self.resinit = orig_resnet.resinit
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        tuple_features = list()
        x = self.resinit(x)
        tuple_features.append(x)
        x = self.maxpool(x)
        tuple_features.append(x)
        x = self.layer1(x)
        tuple_features.append(x)
        x = self.layer2(x)
        tuple_features.append(x)
        x = self.layer3(x)
        tuple_features.append(x)
        x = self.layer4(x)
        tuple_features.append(x)

        return tuple_features

class ResNetModels(object):

    def __init__(self, backbone='', params=None):
        self.backbone = backbone
        self.params = params

    #        #         filters_34 = [64,64, 64, 128, 256, 512]
    #         #         filters_50 = [64, 64, 256, 512, 1024, 2048]
    def get_model(self):
        if self.backbone == 'resnet34':
            arch_net = self.resnet34()
            arch_net = NormalResnetBackbone(arch_net)
            arch_net.num_features = 512
            return arch_net

        elif self.backbone == 'resnet50':
            arch_net = self.resnet50()
            arch_net = NormalResnetBackbone(arch_net)
            return arch_net
        else :
            exit('not found backbone')

    def resnet34(self, **kwargs):
        """Constructs a ResNet-34 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(BasicBlock, [3, 4, 6, 3], deep_base=False, bn_type='torchbn' , origional_in_channels=self.params['in_chns'], **kwargs)
        # model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'))
        return model

    def resnet50(self, **kwargs):
        """Constructs a ResNet-50 model.
        Args:
            pretrained (bool): If True, returns a model pre-trained on Places
        """
        model = ResNet(Bottleneck, [3, 4, 6, 3], deep_base=False, bn_type='torchbn', origional_in_channels=self.params['in_chns'] , **kwargs)
        # model = ModuleHelper.load_model(model, pretrained=self.configer.get('network', 'pretrained'))

        return model

if __name__ == "__main__":
    import torch
    cfg = {
        "network": {"backbone": 'resnet34' }
    }

    model1 = ResNetModels(configer=cfg , params={'in_chns':1}).get_model()

    cfg = {
        "network": {"backbone": 'resnet50'}
    }
    model2 = ResNetModels(configer=cfg , params={'in_chns':1}).get_model()
    print(model1 , model2)
    input1 = torch.randn((16 , 1  ,224,224))
    input2 = torch.randn((16, 1, 224, 224))
    o1 = model1(input1)
    o2 = model2(input2)
    for item in o1:
        print(item.shape)
    for item in o2:
        print(item.shape)


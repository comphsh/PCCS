
from networks.unet import Encoder
import torch
import torch.nn as nn

from networks.normal_module import ASPP
import torch.nn.functional  as F

class Decoder_U2PL(nn.Module):
    def __init__(self ,
        in_planes,
        num_classes=19,
        inner_planes=256,
        dilations=(12, 24, 36),
        rep_head=True,
        low_conv_in=128,
        low_conv_out=128,
        **kwargs):
        super(Decoder_U2PL , self).__init__()
        norm_layer = nn.BatchNorm2d
        self.rep_head = rep_head

        self.low_conv_in = low_conv_in  # 原来256
        self.low_conv_out = low_conv_out  # 原来256
        self.low_conv = nn.Sequential(
            nn.Conv2d(self.low_conv_in, self.low_conv_out, kernel_size=1), norm_layer(self.low_conv_out),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP( in_planes, inner_planes=inner_planes, dilations=dilations)

        self.head_out = self.low_conv_out  # 原来256
        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                self.head_out,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(self.head_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.classifier_in = self.low_conv_out + self.head_out  # 原来512
        self.classifier_out = 128  # 原来256
        self.classifier = nn.Sequential(
            nn.Conv2d(self.classifier_in, self.classifier_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(self.classifier_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.classifier_out, self.classifier_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(self.classifier_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.classifier_out, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        if self.rep_head:
            self.representation_in = self.low_conv_out + self.head_out  # 原来512
            self.representation_out = 128  # 原来256
            self.representation = nn.Sequential(
                nn.Conv2d(self.representation_in, self.representation_out, kernel_size=3, stride=1, padding=1,
                          bias=True),
                norm_layer(self.representation_out),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(self.representation_out, self.representation_out, kernel_size=3, stride=1, padding=1,
                          bias=True),
                norm_layer(self.representation_out),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(self.representation_out, self.representation_out, kernel_size=1, stride=1, padding=0,
                          bias=True),
            )
    def forward(self , x):
        x1, x2, x3, x4 = x
        # x4=torch.Size([16, 256, 16, 16]) x3=torch.Size([16, 128, 32, 32]) x2=torch.Size([16, 64, 64, 64]) x1=torch.Size([16, 32, 128, 128])
        # print("x4={} x3={} x2={} x1={}".format(x4.shape, x3.shape, x2.shape, x1.shape))

        aspp_out = self.aspp(x4)
        # x4=torch.Size([8, 512, 8, 8]) x3=torch.Size([8, 256, 15, 15]) x2=torch.Size([8, 128, 29, 29]) x1=torch.Size([8, 64, 57, 57])
        # print("x4={} x3={} x2={} x1={}".format(x4.shape, x3.shape ,x2.shape ,x1.shape))
        # low_feat = self.low_conv(x1)
        low_feat = self.low_conv(x1)
        aspp_out = self.head(aspp_out)
        h, w = low_feat.size()[-2:]
        aspp_out = F.interpolate(aspp_out, size=(h, w), mode="bilinear", align_corners=True)

        # low_feat=torch.Size([16, 128, 64, 64])  aspp_out=torch.Size([16, 128, 64, 64])
        # print("low_feat={}  aspp_out={}".format(low_feat.shape ,aspp_out.shape))
        aspp_out = torch.cat((low_feat, aspp_out), dim=1)  #高频特征与低频特征合并

        # asppout=torch.Size([8, 256, 29, 29])
        # print("asppout={}".format(aspp_out.shape))  #128+128
        res = {"pred": self.classifier(aspp_out)}

        if self.rep_head:
            res["rep"] = self.representation(aspp_out)

        return res


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19):
        super(Aux_Module, self).__init__()

        norm_layer = nn.BatchNorm2d
        self.aux = nn.Sequential(
            nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        res = self.aux(x)
        return res


class UNet_U2PL(nn.Module):
    def __init__(self, in_chns , class_num ,cfg):
        super(UNet_U2PL, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [32, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder_u2pl = Decoder_U2PL(
            in_planes=params1['feature_chns'][-1],  #encoder的输出维度
            num_classes=class_num,
            inner_planes=cfg['network']['decoder']['kwargs']['inner_planes'],
            dilations=cfg['network']['decoder']['kwargs']['dilations'],
            rep_head=True,
            low_conv_in=cfg['network']['decoder']['kwargs']['low_conv_in'],
            low_conv_out=cfg['network']['decoder']['kwargs']['low_conv_out'],
        )
        self.cfg = cfg
        self.auxor = Aux_Module(cfg['network']['aux_loss']['aux_plane'] , class_num)


    def forward(self , x):
        if self.cfg['network']['bonetype']['name'] == 'std':
            # FPN
            _  ,f1, f2, feat1, feat2 = self.encoder(x)
            outs = self.decoder_u2pl([f1, f2, feat1, feat2])
        else:
            feat1, feat2 = self.encoder(x)
            outs = self.decoder_u2pl([feat1, feat2])

        # print("feat1={}".format(feat1.shape))
        pred_aux = self.auxor(feat1)   #feat1=torch.Size([8, 128, 32, 32]) ,输入的特征数量要对应
        outs['aux'] = pred_aux
        return outs


if __name__ == "__main__":
    input = torch.randn((16,1,256,256)).float().cuda()
    cfg = {}
    cfg['network'] = {'aux_loss' :  {'aux_plane':128}}
    cfg['network'].update({ 'bonetype' :{ 'name' :'std'} })


    model = UNet_U2PL(in_chns=1 , class_num=2 , cfg=cfg).cuda()
    for i in range(100):
        outdict = model(input)
    # pred torch.Size([16, 4, 128, 128])
    # rep torch.Size([16, 128, 128, 128])
    # aux torch.Size([16, 4, 32, 32])
    for k , v in outdict.items():
        print(k , v.shape)
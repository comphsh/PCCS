
from networks.unet import Encoder
import torch
import torch.nn as nn

from networks.normal_module import ASPP
import torch.nn.functional  as F

class MACBlock(nn.Module):
    def __init__(self, in_channels, out_channels ,reduction=16):
        super(MACBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)  #输出size1

        self.cSE1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.cSE2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, out_channels=1, kernel_size=1), nn.Sigmoid())
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )


    def forward(self, inputs):
        # avgx = torch.norm(inputs, p=2, dim=(2, 3), keepdim=True) # [1, 1, 1, c]
        # avgx = avgx / (avgx.sum(dim=1, keepdim=True) + 1e-6)
        # avgx = avgx / (inputs.shape[-2] * inputs.shape[-1])
        avgx = self.avgpool(inputs)
        maxx = self.maxpool(inputs)


        avg_attn = self.cSE1(avgx)
        max_attn = self.cSE2(maxx)
        # chns_attn = self.sSE(inputs)

        # print("avg_attn {}   max_attn {}".format(avg_attn.shape, max_attn.shape))
        # out = inputs * avg_attn + inputs * max_attn + inputs * chns_attn
        out = inputs * avg_attn + inputs * max_attn
        # out = inputs * avg_attn
        out = self.conv(out)
        return out

class Decoder_PPCN_Dif(nn.Module):
    def __init__(self ,
        in_planes,
        num_classes=4,
        inner_planes=256,
        dilations=(12, 24, 36),
        rep_head=True,
        low_conv_in=128,
        low_conv_out=128,
        seg_head_out_dim=64,
        **kwargs):
        super(Decoder_PPCN_Dif , self).__init__()
        norm_layer = nn.BatchNorm2d
        self.rep_head = rep_head

        self.low_conv_in = low_conv_in
        self.low_conv_out = low_conv_out
        self.low_conv = nn.Sequential(
            nn.Conv2d(self.low_conv_in, self.low_conv_out, kernel_size=1), norm_layer(self.low_conv_out),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP( in_planes, inner_planes=inner_planes, dilations=dilations)

        self.head_out = self.low_conv_out
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
        self.classifier_in = self.low_conv_out + self.head_out
        self.classifier_out = seg_head_out_dim
        self.classifier1 = nn.Sequential(
            nn.Conv2d(self.classifier_in, self.classifier_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(self.classifier_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.classifier2 = nn.Sequential(
            nn.Conv2d(self.classifier_out, self.classifier_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(self.classifier_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.seg_head = nn.Conv2d(self.classifier_out, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        # self.bound_classifier = nn.Sequential(
        #     nn.Conv2d(self.classifier_out, self.classifier_out, kernel_size=3, stride=1, padding=1, bias=True),
        #     norm_layer(self.classifier_out),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(0.1),
        # )
        self.bound_head = nn.Conv2d(self.classifier_out , num_classes - 1 , kernel_size=1 , stride=1 , padding=0 , bias=True)

        self.fuse_attn1 = MACBlock(self.classifier_out + 3 , self.classifier_out  , 16) #融入img
        # self.fuse_attn2 = MACBlock(self.classifier_out + 1, self.classifier_out   , 16)

        if self.rep_head:
            self.representation_in = self.low_conv_out + self.head_out
            self.representation_out = self.classifier_out
            self.representation = nn.Sequential(
                nn.Conv2d(self.representation_in, self.representation_out, kernel_size=3, stride=1, padding=1,    bias=True),
                norm_layer(self.representation_out),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                # nn.Conv2d(self.representation_out, self.representation_out, kernel_size=3, stride=1, padding=1,   bias=True),
                # norm_layer(self.representation_out),
                # nn.ReLU(inplace=True),
                # nn.Dropout2d(0.1),
                nn.Conv2d(self.representation_out, self.representation_out, kernel_size=1, stride=1, padding=0,  bias=True),
            )
    def forward(self , x , mode=''):
        if mode == 'main':
            x1, x2, x3, x4 = x
            # x4=torch.Size([16, 256, 16, 16]) x3=torch.Size([16, 128, 32, 32]) x2=torch.Size([16, 64, 64, 64]) x1=torch.Size([16, 32, 128, 128])
            # print("x4={} x3={} x2={} x1={}".format(x4.shape, x3.shape, x2.shape, x1.shape))

            # print(x4.shape)  #torch.Size([16, 256, 15, 15])
            aspp_out = self.aspp(x4)
            # x4=torch.Size([8, 512, 8, 8]) x3=torch.Size([8, 256, 15, 15]) x2=torch.Size([8, 128, 29, 29]) x1=torch.Size([8, 64, 57, 57])
            # print("x4={} x3={} x2={} x1={}".format(x4.shape, x3.shape ,x2.shape ,x1.shape))
            low_feat = self.low_conv(x1)
            aspp_out = self.head(aspp_out)
            h, w = low_feat.size()[-2:]
            aspp_out = F.interpolate(aspp_out, size=(h, w), mode="bilinear", align_corners=True)

            aspp_out = torch.cat((low_feat, aspp_out), dim=1)  # 高频特征与低频特征合并
            # print("asppout={}".format(aspp_out.shape))  #asppout=torch.Size([8, 256, 128, 128])
            dec_map = self.classifier1(aspp_out)
            # print("decc{}".format(dec_map.shape))
            seg_fea = self.classifier2(dec_map)
            seg = self.seg_head(seg_fea)

            # bound_fea = self.bound_classifier(dec_map)
            bound_mask = self.bound_head(seg_fea)
            res_dict = {"seg": seg, "seg_fea": seg_fea  ,"bound_mask": bound_mask}

            if self.rep_head:
                res_dict["embedding"] = self.representation(aspp_out)
            # print("res_dic embed {} ".format(res_dict["embedding"].shape))

            return res_dict
        elif mode=='ema':
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

            aspp_out = torch.cat((low_feat, aspp_out), dim=1)  # 高频特征与低频特征合并
            dec_map = self.classifier1(aspp_out)
            seg_fea = self.classifier2(dec_map)
            seg = self.seg_head(seg_fea)
            res_dict = {"seg": seg, "seg_fea": seg_fea}

            if self.rep_head:
                res_dict["embedding"] = self.representation(aspp_out)

            return res_dict
        else:
            exit('not founf mode-{}'.format(mode))

    def forward_main(self , x):
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

        aspp_out = torch.cat((low_feat, aspp_out), dim=1)  #高频特征与低频特征合并
        # print("asppout={}".format(aspp_out.shape))  #asppout=torch.Size([8, 256, 128, 128])
        dec_map = self.classifier1(aspp_out)
        seg_fea = self.classifier2(dec_map)
        seg = self.seg_head(seg_fea)

        bound_fea= self.bound_classifier(dec_map)
        bound_mask = self.bound_head(bound_fea)
        res_dict = {"seg": seg , "seg_fea":seg_fea ,"bound_mask":bound_mask}

        if self.rep_head:
            res_dict["embedding"] = self.representation(aspp_out)

        return res_dict

    def forward_co_complementary(self , x):
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

        aspp_out = torch.cat((low_feat, aspp_out), dim=1)  #高频特征与低频特征合并
        dec_map = self.classifier1(aspp_out)
        seg_fea = self.classifier2(dec_map)
        seg = self.seg_head(seg_fea)
        res_dict = {"seg": seg , "seg_fea":seg_fea}

        if self.rep_head:
            res_dict["embedding"] = self.representation(aspp_out)

        return res_dict

    def inference_unet(self , x):
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

        aspp_out = torch.cat((low_feat, aspp_out), dim=1)  # 高频特征与低频特征合并
        dec_map = self.classifier1(aspp_out)
        seg_fea = self.classifier2(dec_map)
        seg = self.seg_head(seg_fea)
        res_dict = {"seg": seg}
        return res_dict


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


class UNet_PPCN_Dif(nn.Module):
    def __init__(self, in_chns , class_num ,cfg):
        super(UNet_PPCN_Dif, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [32, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder_ppcn = Decoder_PPCN_Dif(
            in_planes=params1['feature_chns'][-1],  #encoder的输出维度
            num_classes=class_num,
            inner_planes=cfg['network']['decoder']['kwargs']['inner_planes'],
            dilations=cfg['network']['decoder']['kwargs']['dilations'],
            rep_head=True,
            low_conv_in=cfg['network']['decoder']['kwargs']['low_conv_in'],
            low_conv_out=cfg['network']['decoder']['kwargs']['low_conv_out'],
            seg_head_out_dim=cfg['network']['decoder']['kwargs']['seg_head_out_dim'],
        )
        self.cfg = cfg
        self.auxor = Aux_Module(cfg['network']['aux_loss']['aux_plane'] , class_num)

    def forward(self , x , mode='main'):

        _, f1, f2, feat1, feat2 = self.encoder(x)
        outs = self.decoder_ppcn([f1, f2, feat1, feat2] , mode=mode)
        # print("feat1={}".format(feat1.shape))
        pred_aux = self.auxor(feat1)  # feat1=torch.Size([8, 128, 32, 32]) ,输入的特征数量要对应
        outs['aux'] = pred_aux
        return outs

    def inference_unet(self , x):
        _, f1, f2, feat1, feat2 = self.encoder(x)
        # outs = self.decoder_ppcn([f1, f2, feat1, feat2])
        outs = self.decoder_ppcn.inference_unet([f1, f2, feat1, feat2])
        return outs


if __name__ == "__main__":
    input = torch.randn((16,1,256,256)).float().cuda()
    cfg = {}
    cfg['network'] = {'aux_loss' :  {'aux_plane':128}}
    cfg['network'].update({ 'bonetype' :{ 'name' :'std'} })


    model = UNet_PPCN_Dif(in_chns=1 , class_num=2 , cfg=cfg).cuda()
    for i in range(100):
        outdict = model(input)
    # pred torch.Size([16, 4, 128, 128])
    # rep torch.Size([16, 128, 128, 128])
    # aux torch.Size([16, 4, 32, 32])
    for k , v in outdict.items():
        print(k , v.shape)

from networks.unet import Encoder
import torch
import torch.nn as nn

from networks.normal_module import ASPP
import torch.nn.functional  as F

from segmentation_models_pytorch.encoders import get_encoder
from networks.mmseg.module import SegmentationHead ,ClassificationHead
from networks.mmseg.unet_decoder import UnetDecoder
import numpy as np


class ResUnet_SCP(nn.Module):
    def __init__(self, in_chns=2, class_num=4,
                 encoder_name='resnet34' ,
                 decoder_channels = (256, 128, 64, 32, 16),
                 encoder_depth = 5,
                 encoder_weights="imagenet",
                 activation=None,
                 ):
        super(ResUnet_SCP, self).__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_chns,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            # use_batchnorm=decoder_use_batchnorm,
            # center=True if encoder_name.startswith("vgg") else False,
            # attention_type=decoder_attention_type,
        )

        self.seg_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=class_num,
            activation=activation,
            kernel_size=3,
        )

    def forward(self, x):
        features = self.encoder(x)
        # torch.Size([16, 1, 224, 224])
        # torch.Size([16, 64, 112, 112])
        # torch.Size([16, 256, 56, 56])
        # torch.Size([16, 512, 28, 28])
        # torch.Size([16, 1024, 14, 14])
        # torch.Size([16, 2048, 7, 7])
        # for item in features:
        #     print(item.shape)
        output = self.seg_head(self.decoder(*features))
        mask = torch.softmax(output, dim=1)

        batch_pro = batch_prototype(x, mask)
        similarity_map = similarity_calulation(x, batch_pro)
        entropy_weight = agreementmap(similarity_map)
        self_simi_map = selfsimilaritygen(similarity_map)  # B*HW*C
        other_simi_map = othersimilaritygen(similarity_map)  # B*HW*C

        return output, self_simi_map, other_simi_map, entropy_weight

def masked_average_pooling(feature, mask):
    #print(feature.shape[-2:])
    mask = F.interpolate(mask, size=feature.shape[-2:], mode='bilinear', align_corners=True)
    #print((feature*mask).shape)
    masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                     / (mask.sum(dim=(2, 3)) + 1e-5)
    return masked_feature

def batch_prototype(feature,mask):  #return B*C*feature_size
    batch_pro = torch.zeros(mask.shape[0], mask.shape[1], feature.shape[1])
    for i in range(mask.shape[1]):
        classmask = mask[:,i,:,:]
        proclass = masked_average_pooling(feature,classmask.unsqueeze(1))
        batch_pro[:,i,:] = proclass
    return batch_pro

def entropy_value(p, C):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=2) / \
        torch.tensor(np.log(C))#.cuda()
    return y1

def agreementmap(similarity_map):
    score_map = torch.argmax(similarity_map,dim=3)
    #score_map =score_map.transpose(1,2)
    ##print(score_map.shape, 'score',score_map[0,0,:])
    gt_onthot = F.one_hot(score_map,6)
    avg_onehot = torch.sum(gt_onthot,dim=2).float()
    avg_onehot = F.normalize(avg_onehot,1.0,dim=2)
    ##print(gt_onthot[0,0,:,:],avg_onehot[0,0,:])
    weight = 1-entropy_value(avg_onehot,similarity_map.shape[3])
    ##print(weight[0,0])
    #score_map = torch.sum(score_map,dim=2)
    return weight

def similarity_calulation(feature,batchpro): #feature_size = B*C*H*W  batchpro= B*C*dim
    B = feature.size(0)
    feature = feature.view(feature.size(0), feature.size(1), -1)  # [N, C, HW]
    feature = feature.transpose(1, 2)  # [N, HW, C]
    feature = feature.contiguous().view(-1, feature.size(2))
    C = batchpro.size(1)
    batchpro = batchpro.contiguous().view(-1, batchpro.size(2))
    feature = F.normalize(feature, p=2.0, dim=1)
    batchpro = F.normalize(batchpro, p=2.0, dim=1).cuda()
    similarity = torch.mm(feature, batchpro.T)
    similarity = similarity.reshape(-1, B, C)
    similarity = similarity.reshape(B, -1, B, C)
    return similarity

def selfsimilaritygen(similarity):
    B = similarity.shape[0]
    mapsize = similarity.shape[1]
    C = similarity.shape[3]
    selfsimilarity = torch.zeros(B,mapsize,C)
    for i in range(similarity.shape[2]):
        selfsimilarity[i,:,:] = similarity[i,:,i,:]
    return selfsimilarity.cuda()

def othersimilaritygen(similarity):
    similarity = torch.exp(similarity)
    for i in range(similarity.shape[2]):
        similarity[i,:,i,:] =0
    similaritysum = torch.sum(similarity,dim=2)
    similaritysum_union = torch.sum(similaritysum,dim=2).unsqueeze(-1)
    #print(similaritysum_union.shape)
    othersimilarity = similaritysum/similaritysum_union
    #print(othersimilarity[1,1,:].sum())
    return othersimilarity
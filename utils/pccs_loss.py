from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from scipy.ndimage import distance_transform_edt as distance
import numpy as np
from skimage import segmentation as skimage_seg



class PCCS_Prototyoe_Contrastive(nn.Module,ABC):
    def __init__(self, cfg=None):
        super(PCCS_Prototyoe_Contrastive, self).__init__()
        self.configer = cfg
        self.ignore_index = -1
        self.labelbs = self.configer['dataset']['kwargs']['labeled_bs']

        self.num_classes = self.configer['dataset']['num_classes']

    def _loss_constrative_by_multilayer(self,proto_list,proto_num_list , preds_list=None):

        # proto_list=torch.Size([543, 320])  proto_num_list=[a,b,c,d]
        # preds_list充当原型的不确定性计算

        assert proto_list.shape[0] == preds_list.shape[0]
        # torch.mul(a, b)是矩阵a和b对应位相乘,a和b的维度必须相等  torch.mm才是矩阵乘法
        logits = torch.div( torch.mm(proto_list , proto_list.T) , self.configer['contras']['temperature'])

        # --------# 不确定性系数向量，每个分量对应一个原型的权重，不确定性越大，给与越大的权重---------------
        preds_list_uncertainty = -1.0 * torch.sum(preds_list * torch.log(preds_list), dim=1)

        preds_list_uncertainty = torch.exp(preds_list_uncertainty)  # 此时使用不确定性大的系数小
        uncertainty_comatrix = preds_list_uncertainty / preds_list_uncertainty.sum()
        # --------# 不确定性系数向量，每个分量对应一个原型的权重，不确定性越大，给与越大的权重---------------

        # 第1维，0-7表示第一张图，8-15表示第二张图，。。。
        # 第一维，0，1分别表示第一张图的第0类的边界原型和非边界原型  2 3 分别表示第一张图第1类的边界原型和非边界原型
        # print('logits={}'.format(logits.shape))
        # logits = logits.permute(1, 0, 2, 3)

        # 64 * 64 表示64个距离值与64个原型的结果。
        #     生成对角线都是0的对焦矩阵
        # IE_mask = torch.ones_like(logits)
        # IE_mask[torch.eye(logits.shape[0], dtype=torch.bool)] = 0
        IE_mask = torch.scatter(
            torch.ones_like(logits),
            1,  # 按列，即index的值作为列值索引 一般可以用来对标签进行one-hot 编码.
            torch.arange(logits.shape[0]).view(-1, 1).cuda(),
            0  # src的值都是0，用0来填充输出的位置
        )

        # ---------------------------(找多个正样本)--------------------
        judge_positive_mask = None
        for i in range(len(proto_num_list)):
            pre_num = int(np.sum(proto_num_list[0:i]))
            suf_num = int(np.sum(proto_num_list[i + 1:len(proto_num_list)]))
            zero_prefix = torch.zeros(pre_num)
            onet = torch.ones(proto_num_list[i])
            zero_sufix = torch.zeros(suf_num)
            one_img_row = torch.cat([zero_prefix, onet, zero_sufix], dim=0)
            one_row = torch.tile(one_img_row, dims=(proto_num_list[i], 1))
            if judge_positive_mask is None:
                judge_positive_mask = one_row
            else:
                judge_positive_mask = torch.cat([judge_positive_mask, one_row], dim=0)

        judge_positive_mask = judge_positive_mask.cuda()
        assert judge_positive_mask.shape == IE_mask.shape
        # ---------------------找多个正样本-----------------------

        # 多个正样本---计算类内紧凑--- 每行表示正样本的1 ，负样本0
        # positive_matri = judge_positive_mask * logits
        # diff_clsinner = (judge_positive_mask - positive_matri) ** 2
        # loss_contra_inner = diff_clsinner.sum(1) / judge_positive_mask.sum(1)

        # 消去自己和自己的内积
        exp_logits = torch.exp(logits) * IE_mask
        # print("explog={}  isnan={}".format(torch.unique(exp_logits)  , torch.any(torch.isnan(exp_logits))))
        #--------------------找对比系数(---相似度越大，则给予更大的权重----)矩阵------------------
        neg_mask = 1 - judge_positive_mask
        judge_positive_mask = judge_positive_mask * IE_mask
        postive_exp_logits = exp_logits * judge_positive_mask
        contra_coe_matri = postive_exp_logits / postive_exp_logits.sum(1,keepdim=True)

        # print("contra_coe_matri={}  isnan={}".format(torch.unique(contra_coe_matri), torch.any(torch.isnan(contra_coe_matri))))

        neg_logits = exp_logits * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        # print("exploggit {}".format(exp_logits.shape))  #[16, 16]
        inerexp = exp_logits + neg_logits
        midlog = torch.log(exp_logits + neg_logits + 1e-8)  #会出现-inf
        # print("exp_logits = {}".format(torch.unique(exp_logits), torch.any(torch.isnan(exp_logits))))
        # print("neg_logits = {}".format(torch.unique(neg_logits), torch.any(torch.isnan(neg_logits))))
        # print("inexp = {}".format(torch.unique(inerexp), torch.any(torch.isnan(inerexp))))
        # print("midlog = {}".format(torch.unique(midlog), torch.any(torch.isnan(midlog))))

        # log_prob = (logits - torch.log(exp_logits + neg_logits + 1e-8))  #指数变成整数
        log_prob = (torch.log(exp_logits + neg_logits + 1e-8)  - logits)  #翻过来，相当于-log(x/y) = log(y/x) = logy - logx
        # log_prob=tensor([-38.4207, -38.4207, -38.4207,  -0.0000], device='cuda:0')  isnan=False
        # print("log_prob={}  isnan={}".format(torch.unique(log_prob), torch.any(torch.isnan(log_prob))))

        # print("contra_coe_matri * log_prob={}  isnan={}".format(torch.unique(contra_coe_matri * log_prob), torch.any(torch.isnan(contra_coe_matri * log_prob))))
        # mean_log_prob = (judge_positive_mask * log_prob).sum(1) / judge_positive_mask.sum(1)
        mean_log_prob = (contra_coe_matri * log_prob).sum(1)   # 相似度系数
        # print("mean_log_prob={}  isnan={}".format(torch.unique(mean_log_prob),torch.any(torch.isnan(mean_log_prob))))
        loss_contra = mean_log_prob
        # print("loss_contra={}".format(loss_contra.shape))
        # print("uncertainty_comatrix.sum={}".format(uncertainty_comatrix.sum()))
        loss_contra = (loss_contra * uncertainty_comatrix).sum()  #不确定性系数加权
        # loss_contra = (loss_contra).sum()  # 无加权
        # print("loss_contra={} {}".format(loss_contra.shape , loss_contra))
        # print("mean_log_prob={}  loss_contra={} mean={}".format(mean_log_prob , loss_contra,loss_contra.mean()))

        return loss_contra.mean()
        # return loss_contra.mean() + 10.0 * loss_contra_inner.mean()

    def _loss_constrative(self,proto_list):

        # proto_list.shape=torch.Size([XXX, 4, 2, 320])
        # print('proto_list.shape={}'.format(proto_list.shape))
        # [8,4,2,fea_size] --> [8,fea_size , 4,2]
        contra_feature = proto_list.permute(0, 3, 1, 2)

        # contra_feature.shape=torch.Size([8, 320, 4, 2])
        # print('contra_feature.shape={}'.format(contra_feature.shape))

        # [8,4,2,fea_size] --> [outputchan:fea_size , inputchan:fea_size , 1 ,1 ]
        kernels = proto_list.reshape(-1, contra_feature.shape[1], 1, 1)
        # kernel=torch.Size([64, 320, 1, 1])
        # print("kernel={}".format(kernels.shape))
        temperature = 0.05
        logits = torch.div(F.conv2d(contra_feature, kernels), temperature)
        # print("1 logtis={}".format(logits.unique()))
        # logits=torch.Size([8, 64, 4, 2]) 64表示当前该点表示的原型与整个batch的其他原型计算距离的结果数
        # 第1维，0-7表示第一张图，8-15表示第二张图，。。。
        # 第一维，0，1分别表示第一张图的第0类的边界原型和非边界原型  2 3 分别表示第一张图第1类的边界原型和非边界原型
        # print('logits={}'.format(logits.shape))
        logits = logits.permute(1, 0, 2, 3)
        contra_coe_matri = F.softmax(logits, dim=-1)

        # 64 * 64 表示64个距离值与64个原型的结果。
        logits = logits.reshape(logits.shape[0], -1)
        contra_coe_matri = contra_coe_matri.reshape(contra_coe_matri.shape[0], -1)
        #     生成对角线都是0的对焦矩阵
        # IE_mask = torch.ones_like(logits)
        # IE_mask[torch.eye(logits.shape[0], dtype=torch.bool)] = 0
        IE_mask = torch.scatter(
            torch.ones_like(logits),
            1,  # 按列，即index的值作为列值索引 一般可以用来对标签进行one-hot 编码.
            torch.arange(logits.shape[0]).view(-1, 1).cuda(),
            0  # src的值都是0，用0来填充输出的位置
        )

        # 生成只留下自己的正样本的判断矩阵
        # judge_positive_mask = torch.zeros_like(logits)
        # for i in range(logits.shape[0]):  # 只是找一个正样本
        #     if i % 2 == 0:
        #         judge_positive_mask[i][i + 1] = 1
        #     else:
        #         judge_positive_mask[i][i - 1] = 1

        # ---------------------------(找多个正样本)--------------------
        img_num, fea_size, classes, bound_num = contra_feature.shape
        zerot = torch.zeros(bound_num)
        onet = torch.ones(bound_num)
        judge_positive_mask = None
        for i in range(logits.shape[0]):
            bound_id = i % (classes*bound_num)
            cls_id = bound_id // bound_num
            zero_prefix = zerot.tile(cls_id)
            zero_sufix = zerot.tile(classes - cls_id - 1)
            one_img_row = torch.cat([zero_prefix, onet, zero_sufix], dim=0)
            one_row = one_img_row.tile(img_num)
            # print("i={}  clsid={}  onerow={}".format(i, cls_id, one_row))
            if judge_positive_mask is None:
                judge_positive_mask = one_row.unsqueeze(0)
            else:
                judge_positive_mask = torch.cat([judge_positive_mask, one_row.unsqueeze(0)], dim=0)
        judge_positive_mask = judge_positive_mask.cuda()
        print("judge_positive_mask={}".format(judge_positive_mask.shape))
        assert judge_positive_mask.shape == IE_mask.shape
        # ---------------------找多个正样本及其对比系数矩阵-----------------------
        neg_mask = 1 - judge_positive_mask
        judge_positive_mask = judge_positive_mask * IE_mask
        contra_coe_matri = contra_coe_matri * judge_positive_mask

        # 多个正样本计算类内紧凑 每行表示正样本的1 ，负样本0
        positive_matri = judge_positive_mask * logits
        diff_clsinner = (judge_positive_mask - positive_matri) ** 2
        loss_contra_inner = diff_clsinner.sum(1) / judge_positive_mask.sum(1)

        # 消去自己和自己的内积
        exp_logits = torch.exp(logits) * IE_mask
        neg_logits = exp_logits * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        log_prob = logits - torch.log(exp_logits + neg_logits)
        # print("1 log_prob={}".format(log_prob.unique()))

        mean_log_prob = (contra_coe_matri * log_prob).sum(1) / judge_positive_mask.sum(1)
        # print("1 mean_log_prob={}".format(mean_log_prob.unique()))
        # print("mean_log_prob={}".format(mean_log_prob))
        loss_contra = mean_log_prob
        # print("mean_log_prob={}  loss_contra={} mean={}".format(mean_log_prob , loss_contra,loss_contra.mean()))
        return loss_contra.mean()
        # return loss_contra.mean() + 10.0 * loss_contra_inner.mean()

    def forward(self, preds, target):

        # batch , 320 , h112 ,w112用于确定原型
        prototype_embed = preds['prototype_embed']
        proto_num_list = preds['proto_num_list']
        preds_list = preds['preds_list']
        # 筛选出边界原型和非边界原型
        loss_contra_sup = self._loss_constrative_by_multilayer(prototype_embed , proto_num_list , preds_list)
        return loss_contra_sup



class ConsistencyLoss(nn.Module):
    def __init__(self,configer):
        super(ConsistencyLoss,self).__init__()

    def symmetric_mse_loss(self,input1, input2):
        """Like F.mse_loss but sends gradients to both directions

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to both input1 and input2.
        """
        assert input1.size() == input2.size()
        return torch.mean((input1 - input2) ** 2)

    def forward(self,seg_inputs,seg_proto_inputs):
        loss = nn.MSELoss()(seg_inputs,seg_proto_inputs)
        return loss


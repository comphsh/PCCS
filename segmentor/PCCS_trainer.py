import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss

from tqdm import tqdm
from einops import rearrange, repeat

from utils import losses, metrics, ramps, val_2d
from networks.net_factory import net_factory
from utils.pccs_loss import PCCS_Prototyoe_Contrastive

from dataloaders.acdc import get_acdc_data_loader
from dataloaders.acdc_by_volume import get_acdc_by_volume_data_loader
from dataloaders.busi import get_busi_data_loader
from dataloaders.breast import get_breast_data_loader



def get_current_consistency_weight(args , epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)



def train(args, snapshot_path , cfg=None, trainloader=None , valloader=None , db_val_len=1):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=args.in_chns,class_num=num_classes , cfg=cfg)
    ema_model = net_factory(net_type=cfg['network']['ema_net_type'], in_chns=args.in_chns,class_num=num_classes , cfg=cfg)
    #model.load_state_dict({k.replace('module.',''):v for k,v in torch.load('/home/zxzhang/MC-Net-Main/model/Prostate_EntropyProLearningAbL_cpcc_7_labeled/unet_pro/iter_16100_dice_0.7502.pth').items()})
    # for param in ema_model.encoder.parameters():
    #     param.detach_()
        # param.requires_grad_(False)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(ema_model.parameters() , lr=base_lr , momentum=0.9 , weight_decay=0.0001)



    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    pixel_prototype_loss = PCCS_Prototyoe_Contrastive(cfg)
    from utils.pccs_loss import ConsistencyLoss
    consis_criterion = ConsistencyLoss(configer=cfg)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    resize_size = cfg["dataset"]["kwargs"]["resize_size"]
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    best_min_loss = 1000000
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            print('hello')

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            consistency_weight = get_current_consistency_weight(args, iter_num // 150)

            if iter_num <= 200  :
                outputs_dict = model(volume_batch)
                outputs = outputs_dict["seg"]
                outputs_soft = torch.softmax(outputs, dim=1)
                loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:args.labeled_bs][:].long())
                loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))

                # 监督损失
                supervised_loss = (loss_ce + loss_dice) / 2

                u_pred_soft = outputs_soft[args.labeled_bs:]
                uncertainty = -torch.sum(u_pred_soft * torch.log(u_pred_soft + 1e-10), dim=1)
                uncertainty_avg = torch.mean(uncertainty, dim=[-2, -1]).mean()

                semi_supervised_loss = torch.tensor(0.0)
                loss_con, loss_auxce, loss_contra_sup = 0.0, 0.0, 0.0
                loss_sup_con, loss_unsup_con , loss_uncertainty = 0.0, 0.0 , 0.0
            elif iter_num <= 500 :
                outputs_ema_dict = ema_model(volume_batch)
                outputs_dict = model(volume_batch , use_prototype=True)

                outseg_main = outputs_dict["seg"]
                outputs_soft = torch.softmax(outseg_main, dim=1)
                loss_ce = ce_loss(outseg_main[:args.labeled_bs], label_batch[:args.labeled_bs][:].long())
                loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))

                u_pred_soft = outputs_soft[args.labeled_bs:]
                uncertainty = -torch.sum(u_pred_soft * torch.log(u_pred_soft + 1e-10), dim=1)
                uncertainty_avg = torch.mean(uncertainty, dim=[-2, -1]).mean()

                # 监督损失
                supervised_loss = (loss_ce + loss_dice) / 2

                cls_prob_ema = outputs_ema_dict['cls_prob_ema']

                pbs, _ , ph, pw = cls_prob_ema.shape
                # 计算不确定性
                outseg_main = F.interpolate(outseg_main , size=(ph , pw ) , mode="bilinear" , align_corners=True)

                positive_outseg_main = F.softmax(outseg_main , dim=1)
                unet_uncertainty_map = -1.0 * torch.sum(positive_outseg_main * torch.log(positive_outseg_main + 1e-8), dim=1,keepdim=True)
                positive_out_ema_proto = F.softmax(cls_prob_ema , dim=1)
                proto_uncertainty_map = -1.0 * torch.sum(positive_out_ema_proto * torch.log(positive_out_ema_proto + 1e-8),dim=1, keepdim=True)

                # 一致性：
                loss_sup_con = consis_criterion(outseg_main[:args.labeled_bs], cls_prob_ema[:args.labeled_bs])
                loss_unsup_con = consis_criterion(outseg_main[args.labeled_bs:], cls_prob_ema[args.labeled_bs:])
                # unet 不确定性减小正则化
                loss_unet_uncertainty = torch.sqrt_(torch.sum(unet_uncertainty_map ** 2))
                loss_proto_uncertainty = torch.sqrt_(torch.sum(proto_uncertainty_map ** 2))
                loss_uncertainty = (loss_unet_uncertainty + loss_proto_uncertainty) / 2.0
                loss_con = (loss_sup_con + loss_unsup_con) + cfg['loss']['uncertainty_coefficient'] * loss_uncertainty

                semi_supervised_loss =  loss_con * consistency_weight
                loss_auxce, loss_contra_sup =0.0, 0.0

            else :

                outputs_ema_dict = ema_model(volume_batch)
                outputs_dict = model(volume_batch , use_prototype=True)

                outseg_main = outputs_dict['seg']

                # 监督损失
                outputs_soft = torch.softmax(outseg_main, dim=1)
                loss_ce = ce_loss(outseg_main[:args.labeled_bs], label_batch[:args.labeled_bs][:].long())
                loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
                supervised_loss = (loss_ce + loss_dice ) / 2

                u_pred_soft = outputs_soft[args.labeled_bs:]
                uncertainty = -torch.sum(u_pred_soft * torch.log(u_pred_soft + 1e-10), dim=1)
                uncertainty_avg = torch.mean(uncertainty, dim=[-2, -1]).mean()

                pfea_main = outputs_dict['proto_featuremap']
                cls_kproto_prob_main = outputs_dict['cls*kproto_prob_main']

                cls_prob_ema = outputs_ema_dict['cls_prob_ema']
                cls_kproto_prob_ema = outputs_ema_dict['cls_kproto_prob_ema']
                pfea_ema = outputs_ema_dict['proto_featuremap']

                pbs,  _ , ph, pw = cls_prob_ema.shape
                pchns = pfea_ema.shape[-1]
                gt = F.interpolate(label_batch.unsqueeze(1).float() , size=(ph,pw) , mode='nearest')  # 用来适应指导中间特征图
                #                     计算不确定性
                outseg_main = F.interpolate(outseg_main , size=(ph , pw ) , mode="bilinear" , align_corners=True)
                positive_outseg_main = F.softmax(outseg_main , dim=1)
                unet_uncertainty_map = -1.0 * torch.sum(positive_outseg_main * torch.log(positive_outseg_main + 1e-8), dim=1,keepdim=True)
                threshold = (0.25 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2.7)
                # threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(step, self.max_iter)) * np.log(2)
                unet_uncertainty_mask = (unet_uncertainty_map > threshold)

                pseudo_main = torch.argmax(positive_outseg_main , dim=1 ,keepdim=True).float()
                #   pseudo_main[unet_uncertainty_mask] = -1
                pseudo_main[:args.labeled_bs] = gt[:args.labeled_bs]
                pseudo_main = pseudo_main.float()

                # 无标注数据使用伪标签， 有标注数据使用gt
                # 有标签计算一次原型
                gt_seg =  pseudo_main.float().view(-1)
                half_label = gt_seg.shape[0] // 2

                #计算原型不确定性
                positive_cls_prob_ema = F.softmax(cls_prob_ema , dim=1)
                proto_pred_ema = torch.argmax(positive_cls_prob_ema , dim=1 , keepdim=True).float()
                proto_uncertainty_map = -1.0 * torch.sum(positive_cls_prob_ema * torch.log(positive_cls_prob_ema + 1e-8),dim=1, keepdim=True)

                proto_pred_ema = proto_pred_ema.float().view(-1)

                # proto_featuremap_main = rearrange(pfea_main, 'b c h w -> (b h w) c')
                # proto_featuremap_main = F.normalize(proto_featuremap_main, p=2, dim=-1)

                cls_kproto_prob_main = cls_kproto_prob_main.permute(0 ,2 ,3 ,1)
                cls_kproto_1_prob_main = cls_kproto_prob_main.reshape(cls_kproto_prob_main.shape[0] ,
                                                  cls_kproto_prob_main.shape[1] ,
                                                  cls_kproto_prob_main.shape[2] ,
                                                  cls_kproto_prob_main.shape[3] // cfg['protoseg']['num_prototype'] ,
                                                  cfg['protoseg']['num_prototype'])
                cls_prob_main_from_proto = torch.amax(cls_kproto_1_prob_main , dim=-1)   #找出最大概率的子原型
                bs , h , w ,c = cls_prob_main_from_proto.shape
                cls_prob_main_from_proto = ema_model.mask_norm(cls_prob_main_from_proto.reshape(-1, cls_prob_main_from_proto.shape[-1]))  #从ema_model进行推理得出
                cls_prob_main_from_proto = cls_prob_main_from_proto.reshape(bs, h, w, c).permute(0, 3, 1, 2)
                cls_kproto_prob_main = cls_kproto_prob_main.reshape(-1,
                                              cfg['protoseg']['num_prototype'],
                                              cls_kproto_prob_main.shape[-1] // cfg['protoseg']['num_prototype'])

                # 过滤多层原型 ,也就是说，符号距离图的特征向量来自于model_main的特征头
                proto_list_map = filter_multilayer_proto(args , cfg ,
                                                              outputs_dict['proto_featuremap'],
                                                              pseudo_main,
                                                              unet_uncertainty_mask,
                                                              positive_outseg_main)

                #出队入队暂且不写
                outputs_dict['prototype_embed'] = proto_list_map['proto_list']
                outputs_dict['proto_num_list'] = proto_list_map['proto_num_list']
                outputs_dict['preds_list'] = proto_list_map['preds_list']

                # [37 , 256]  --> [4, 2, 256]

                c_k_boundary_proto = avg1d_for_cXk(proto_list_map['proto_list'] ,
                                                      nlist = proto_list_map['proto_num_list'] ,
                                                      proto_num = cfg["protoseg"]["num_prototype"] ,
                                                      ndim=pchns)

                # helf =524288
                # print("helf ={}".format(half_label))
                # 所谓的原型指导原型反而不好理解  # 本次 只计算 labels 改变label_contra_logits 长度
                labeled_contrast_logits, labeled_contrast_target = prototype_updating_prototype_direct_feaAndProto_teacherAsTarget(
                                                                                                        model, ema_model, cfg, args,
                                                                                                        # proto_featuremap_main[half_label:],
                                                                                                        pfea_ema[half_label:],
                                                                                                        cls_prob_main_from_proto[args.labeled_bs:],
                                                                                                        proto_pred_ema[half_label:],
                                                                                                        cls_kproto_prob_main[half_label:] ,
                                                                                                        c_k_boundary_proto)




                #     计算半监督损失
                #辅助损失
                auxce_logit = labeled_contrast_logits
                auxce_target = labeled_contrast_target
                loss_auxce = F.cross_entropy(auxce_logit,auxce_target.long(),ignore_index=-1)
                # loss_auxce = torch.tensor(0.0)

                # 一致性：
                loss_sup_con = consis_criterion(outseg_main[:args.labeled_bs], cls_prob_ema[:args.labeled_bs])
                loss_unsup_con = consis_criterion(outseg_main[args.labeled_bs:], cls_prob_ema[args.labeled_bs:])
                #  不确定性减小正则化
                loss_unet_uncertainty = torch.sqrt_(torch.sum(unet_uncertainty_map ** 2))
                loss_proto_uncertainty = torch.sqrt_(torch.sum(proto_uncertainty_map ** 2))
                loss_uncertainty = (loss_unet_uncertainty + loss_proto_uncertainty) / 2.0
                loss_con = (loss_sup_con + loss_unsup_con) + cfg['loss']['uncertainty_coefficient'] * loss_uncertainty



                #对比学习
                loss_contra_sup = pixel_prototype_loss(outputs_dict, pseudo_main.squeeze(1))

                semi_supervised_loss = loss_con * consistency_weight + cfg['loss']['contra_coefficient'] * loss_contra_sup    + cfg['loss']['auxce_coefficient'] * loss_auxce

            loss = supervised_loss + semi_supervised_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # schedule
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            # update_ema_variables
            # alpha = min(1 - 1 / (iter_num + 1), cfg["network"]["ema_alpha"])
            # for ema_param, param in zip(ema_model.encoder.parameters(), model.encoder.parameters()):
            #     ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            writer.add_scalar('info/semi_supervised_loss',semi_supervised_loss, iter_num)
            writer.add_scalar('info/loss_auxce', loss_auxce, iter_num)
            writer.add_scalar('info/loss_con', loss_con, iter_num)
            writer.add_scalar('info/loss_uncertainty', loss_uncertainty, iter_num)
            writer.add_scalar('info/loss_contra_sup', loss_contra_sup, iter_num)
            writer.add_scalar('info/consistency_weight',consistency_weight, iter_num)
            writer.add_scalar('info/uncertainty_avg', uncertainty_avg, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f  semi_supervised_loss: %f  uncertainty_avg:%f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item() , semi_supervised_loss.item() , uncertainty_avg.item()))


            if iter_num >= 0 and iter_num % 500 == 0:
                model.eval()
                performance, sec_val = val_val(model=model if not isinstance(model , nn.DataParallel) else model.module , valloader=valloader, writer=writer, cfg=cfg, num_classes=args.num_classes, db_val_len=db_val_len, iter_num=iter_num,   resize_size=resize_size)
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_model_{}_dice_{}.pth'.format(iter_num, round(  best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model1.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                if loss.item() < best_min_loss:
                    best_min_loss = loss.item()
                    save_best_path = os.path.join(snapshot_path, '{}_best_loss_model1.pth'.format(args.model))
                    torch.save(model.state_dict(), save_best_path)

                if "2D" == cfg["dataset"]["dimension"]:
                    logging.info(   'iteration %d :valset :  model1: mean_dice : %f mean_jaccard : %f' % (iter_num, performance, sec_val))
                else:
                    logging.info(  'iteration %d : valset :  model1:  mean_dice : %f mean_hd95 : %f' % (iter_num, performance, sec_val))

                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

def val_val(model=None, valloader=None, writer=None, cfg=None, num_classes=2 , db_val_len=1 ,  iter_num = 1, resize_size=[256,256] , in_chns=1):

    if "2D" == cfg["dataset"]["dimension"]:
        metric_list = np.zeros((num_classes - 1, 2))  # numclass类， 每类计算2个metric
        val_len = 0
        for _, sampled_batch in enumerate(valloader):
            # print(sampled_batch["image"].unique() , sampled_batch["label"].unique()) #torch.Size([1, 11, 256, 216]) torch.Size([1, 11, 256, 216]) , 0到1之间，归一化，， label=0,1,2,3
            #
            data, mask = sampled_batch["image"].float().cuda(), sampled_batch["label"].long().cuda()
            if len(data.shape) == 3:
                data = data.unsqueeze(1)  # 通道维度增加1
            elif len(data.shape) == 2:
                data = data.unsqueeze(0).unsqueeze(1)

            if len(mask.shape) == 4:
                mask = mask.squeeze(1)

            _, _, h, w = data.shape
            input = F.interpolate(data, size=(resize_size[0], resize_size[1]), mode="bilinear")

            output = model(input)

            if isinstance(output, dict):
                output = output["seg"]
            elif isinstance(output, tuple):
                output = output[0]

            output = F.interpolate(output, mask.shape[-2:], mode="bilinear", align_corners=True)
            out = torch.argmax(output, dim=1).squeeze(1)  # output去掉第一维 变成 [bs , h , w]
            out = out.cpu().detach().numpy()
            pred = out
            # pred = zoom(out, (1, h / resize_size[0], w / resize_size[0]), order=0)  # 回归原来大小，val数据集没有经过transform
            # print("pred=" , pred.shape)  [bs , h , w]
            mask = mask.cpu().detach().numpy()
            val_len += 1
            assert pred.shape == mask.shape
            for cls in range(1, num_classes):  # 一个batch 同类算一次
                val_i = cal_2d_metric(pred  == cls, mask == cls)  # 这里只算平均，，每个类的值，到了每一类都在这了
                metric_list[cls - 1] += np.array(val_i)

        metric_list = metric_list / val_len
        for class_i in range(num_classes - 1):
            writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
            writer.add_scalar('info/val_{}_jc'.format(class_i + 1), metric_list[class_i, 1], iter_num)

        performance = np.mean(metric_list, axis=0)[0]  # 所有0维(全部行)，加起来求均值
        sec_val = mean_jaccard = np.mean(metric_list, axis=0)[1]

        writer.add_scalar('info/val_mean_dice', performance, iter_num)
        writer.add_scalar('info/val_mean_jaccard', mean_jaccard, iter_num)

    elif "3D" == cfg["dataset"]["dimension"]:
        metric_list = 0.0
        for _, sampled_batch in enumerate(valloader):
            # print(sampled_batch["image"].unique() , sampled_batch["label"].unique()) #torch.Size([1, 11, 256, 216]) torch.Size([1, 11, 256, 216]) , 0到1之间，归一化，， label=0,1,2,3
            metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes , patch_size=resize_size)
            metric_list += np.array(metric_i)
        metric_list = metric_list / db_val_len

        for class_i in range(num_classes - 1):
            writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
            writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

        performance = np.mean(metric_list, axis=0)[0]
        sec_val = mean_hd95 = np.mean(metric_list, axis=0)[1]

        writer.add_scalar('info/val_mean_dice', performance, iter_num)
        writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
    else:
        exit("no found cal val!")


    return performance  , sec_val

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, model, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model.eval()
        with torch.no_grad():
            output = model(input)
            if isinstance( output , dict ):
                output = output["seg"]
            elif isinstance(output , tuple):
                output = output[0]

            out = torch.argmax(output, dim=1).squeeze(0)#torch.sigmoid(output).squeeze()
            #out = out>0.5
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

def cal_2d_metric(pred , gt): #单张图像单个类的值
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        return dice, jaccard
    else:
        return 0, 0


def prototype_updating_prototype_direct_feaAndProto_teacherAsTarget(model , ema_model , cfg , args, proto_featuremap, pred_prob, gt_seg, masks , c_k_boundary_proto=None):
    # [0]->val  [1]->indice
    # torch.Size([32768, 256])  torch.Size([8, 4, 64, 64])  torch.Size([32768])  torch.Size([32768, 2, 4])
    # print("{}  {}  {}  {}".format(proto_featuremap.shape , pred_prob.shape , gt_seg.shape , masks.shape))
    pred_seg = torch.max(pred_prob, 1)[1]
    mask = (gt_seg == pred_seg.view(-1))
    # proto_target = [100352] 样本点的label
    proto_target = gt_seg.clone().float()

    protos = ema_model.prototype.data.clone()
    seg_net_proto = model.prototype.weight.data.squeeze(-1).squeeze(-1)

    # cosine_similarity = torch.mm(proto_featuremap, self.seg_net_ema.prototype.view(-1, self.seg_net_ema.prototype.shape[-1]).t())
    proto_logits = torch.mm(proto_featuremap, seg_net_proto.t())

    # proto_logits=[100352,4*10] 样本点与所有原型计算内积
    for k in range(args.num_classes):
        #  masks=torch.Size([100352, 10, 4])
        init_q = masks[..., k]
        # 把像素中 真是第k类的 像素(对应10个原型的像素)挑出来
        init_q = init_q[gt_seg == k, ...]
        if init_q.shape[0] == 0:
            continue
        # ture initq=torch.Size([95299, 10])
        q, indexs = distributed_sinkhorn(init_q)  #通过 Sinkhorn 算法得到的概率来平滑地更新原型特征，
        # sinkhorn q.shape=torch.Size([95299, 10])
        # mask：是预测正确的像素从中跳出是k类的像素
        m_k = mask[gt_seg == k]
        # _c=torch.Size([100352, 896]) 从100352个像素中跳出类别k gt=true的特征向量
        c_k = proto_featuremap[gt_seg == k, ...]
        # c_k=torch.Size([91276, 896]) m_k=torch.Size([91276]) n_proto=10
        # tile 每列都一样
        m_k_tile = repeat(m_k, 'n -> n tile', tile=cfg['protoseg']['num_prototype'])
        # m_k_tile = torch.Size([91276, 10])
        # q.shape=torch.Size([91276, 10])
        # m_k :每列都一样，每一列是 预测正确的像素中跳出第k类的像素
        m_q = q * m_k_tile  # n x self.num_prototype
        # m_k.shape =91276
        c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
        # c_k_tile.shape = 896列m_k [91276,896]
        c_q = c_k * c_k_tile  # n x embedding_dim
        f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim
        # 这个操作实际上是在计算每个原型特征的加权平均，其中每个像素对于不同原型的贡献由 Sinkhorn 算法的输出概率 q 决定。
        n = torch.sum(m_q, dim=0)

        # update_prototype = true
        if torch.sum(n) > 0:
            f = F.normalize(f, p=2, dim=-1)
            new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],momentum=cfg['protoseg']['gamma'])

            # new_value.shape=torch.Size([10, 320])
            # protos.shape = torch.Size([4, 10, 320])
            protos[k, n != 0, :] = new_value
        proto_target[gt_seg == k] = indexs.float() + (cfg['protoseg']['num_prototype'] * k)

    # protos.shape = torch.Size([4, 10, 320])  seg_net.prototype.shape = (320 , 40)  将 (40,320,1,1) 转成 （4 ，10，320）
    seg_net_proto = seg_net_proto.view(protos.shape)

    assert protos.shape == seg_net_proto.shape and c_k_boundary_proto.shape == protos.shape
    # print(protos.device ,  c_k_boundary_proto.device)
    protos = cfg['network']['momentum_fromlearn'] * protos + (1 - cfg['network']['momentum_fromlearn']) * c_k_boundary_proto
    ema_model.prototype = nn.Parameter(F.normalize(protos, p=2, dim=-1), requires_grad=False)

    return proto_logits, proto_target

def momentum_update(old_value , new_value , momentum=0.9999):
    update = momentum * old_value + (1 - momentum) * new_value
    return update

def distributed_sinkhorn(out, sinkhorn_iterations=4, epsilon=0.05):
    Q = torch.exp(out / epsilon).t() # K x B
    B = Q.shape[1]
    K = Q.shape[0]

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for _ in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    Q = Q.t()

    indexs = torch.argmax(Q, dim=1)
    # Q = torch.nn.functional.one_hot(indexs, num_classes=Q.shape[1]).float()
    Q = F.gumbel_softmax(Q, tau=0.5, hard=True)

    return Q, indexs

from utils.boundary_func import img_to_sdf
def filter_multilayer_proto(args ,  cfg , proto_featuremap,target,unet_uncertainty_mask,preds=None):
    # profea=torch.Size([16, 320, 64, 64])  target=torch.Size([16, 1, 112, 112])
    assert unet_uncertainty_mask.shape == target.shape , print("uncertainty map = {}   target={}".format(unet_uncertainty_mask.shape , target.shape))
    assert proto_featuremap.shape[-2:] == target.shape[-2:]
    numclasses = args.num_classes
    # 样本特征归一化
    # proto_featuremap = F.normalize(proto_featuremap, p=2, dim=1)
    # unet_uncertainty_mask=torch.Size([16, 1, 112, 112])
    # 有标签数据使用GT辨识边界
    # 通过边界与mask分割出边界像素和非边界像素
    begin_labelbs = cfg['contras']['begin_contra']
    labeled_bs = cfg['contras']['end_contra']
    proto_list = None
    preds_list = None
    proto_featuremap = proto_featuremap.permute(0, 2, 3, 1)
    preds_map = preds.permute(0 ,2,3,1)
    fea_size = proto_featuremap.shape[-1]
    proto_num_list = [0 for _ in range(numclasses)]
    for cls in range(0 , numclasses):
        with torch.no_grad():
            labeled_real_gt_sdf = img_to_sdf(target[:labeled_bs].cpu().numpy() == cls,out_shape=target[:labeled_bs, 0, ...].shape)
            labeled_sdf_innerline = img_to_sdf(target[:labeled_bs].cpu().numpy() <= cls,out_shape=target[:labeled_bs, 0, ...].shape)
            labeled_sdf_outsideline = img_to_sdf(target[:labeled_bs].cpu().numpy() >= cls,out_shape=target[:labeled_bs, 0, ...].shape)

        for bat in range(begin_labelbs , labeled_bs):
            with torch.no_grad():
                min_dis_eq = np.min(labeled_real_gt_sdf[bat])
                # max_dis_eq = np.max(labeled_real_gt_sdf[bat])
                # min_dis_low = np.min(labeled_sdf_innerline[bat])
                max_dis_low = np.max(labeled_sdf_innerline[bat])
                # min_dis_upper = np.min(labeled_sdf_outsideline[bat])
                max_dis_upper = np.max(labeled_sdf_outsideline[bat])
                if min_dis_eq >= 0:
                    #     图像里没有该类/或者只有一层
                    continue
                if cls == 0:
                    # 单独计算背景层次原型
                    union_sdf_uncertainty = labeled_real_gt_sdf[bat] * unet_uncertainty_mask[bat, 0, ...].cpu().numpy() * (target[bat, 0].cpu().numpy() == cls)

                    far_dis = min_dis_eq // 2
                    nonbound_sdf = (union_sdf_uncertainty < far_dis) * (union_sdf_uncertainty < 0)
                    bound_sdf = (union_sdf_uncertainty >= far_dis) * (union_sdf_uncertainty < 0)
                    nonbound_coordinate = np.nonzero(nonbound_sdf)
                    bound_coordinate = np.nonzero(bound_sdf)
                    proto_nonbound = proto_featuremap[bat][nonbound_coordinate]
                    proto_bound = proto_featuremap[bat][bound_coordinate]
                    if proto_bound.shape[0] > 0:
                        mean_proto_bound = torch.mean(proto_bound, dim=0)
                        proto_num_list[cls] = proto_num_list[cls] + 1
                        if proto_list is None:
                            proto_list = mean_proto_bound
                        else:
                            proto_list = torch.cat([proto_list, mean_proto_bound], dim=0)

                        preds_patch = preds_map[bat][bound_coordinate]
                        mean_preds = torch.mean(preds_patch, dim=0)
                        if preds_list is None:
                            preds_list = mean_preds
                        else :
                            preds_list = torch.cat([preds_list, mean_preds], dim=0)

                    if proto_nonbound.shape[0] > 0:
                        mean_nonproto_bound = torch.mean(proto_nonbound, dim=0)
                        proto_num_list[cls] = proto_num_list[cls] + 1
                        if proto_list is None:
                            proto_list = mean_nonproto_bound
                        else:
                            proto_list = torch.cat([proto_list, mean_nonproto_bound], dim=0)

                        preds_patch = preds_map[bat][nonbound_coordinate]
                        mean_preds = torch.mean(preds_patch, dim=0)
                        if preds_list is None:
                            preds_list = mean_preds
                        else:
                            preds_list = torch.cat([preds_list, mean_preds], dim=0)
                    continue

                uniq_dis = np.unique(labeled_real_gt_sdf[bat])  # np.unique( ):返回其参数数组中所有不同的值,并按从小到大的顺序排列
                if max_dis_low <0 or max_dis_upper <0:
                #     说明此时全覆盖，不被两侧挤压
                    for dis in uniq_dis:
                        if dis >= 0:
                            break
                        bound_sdf = (labeled_real_gt_sdf[bat] == dis)
                        bound_coordinate = np.nonzero(bound_sdf)
                        proto_bound = proto_featuremap[bat][bound_coordinate]
                        if proto_bound.shape[0] > 0:
                            mean_proto_bound = torch.mean(proto_bound, dim=0)
                            proto_num_list[cls] = proto_num_list[cls] + 1
                            if proto_list is None:
                                proto_list = mean_proto_bound
                            else:
                                proto_list = torch.cat([proto_list, mean_proto_bound], dim=0)

                            preds_patch = preds_map[bat][bound_coordinate]
                            mean_preds = torch.mean(preds_patch, dim=0)
                            if preds_list is None:
                                preds_list = mean_preds
                            else:
                                preds_list = torch.cat([preds_list, mean_preds], dim=0)
                else:
                    # 说明此时被两侧挤压
                    for dis in uniq_dis:  #取内侧的像素
                        if dis >= 0:
                            break
                        bound_sdf = (labeled_sdf_innerline[bat] == dis)
                        bound_coordinate = np.nonzero(bound_sdf)
                        proto_bound = proto_featuremap[bat][bound_coordinate]
                        if proto_bound.shape[0] > 0:
                            mean_proto_bound = torch.mean(proto_bound, dim=0)
                            proto_num_list[cls] = proto_num_list[cls] + 1
                            if proto_list is None:
                                proto_list = mean_proto_bound
                            else:
                                proto_list = torch.cat([proto_list, mean_proto_bound], dim=0)

                            preds_patch = preds_map[bat][bound_coordinate]
                            mean_preds = torch.mean(preds_patch, dim=0)
                            if preds_list is None:
                                preds_list = mean_preds
                            else:
                                preds_list = torch.cat([preds_list, mean_preds], dim=0)

                    for dis in uniq_dis:#取外侧的像素
                        if dis >= 0:
                            break
                        bound_sdf = (labeled_sdf_outsideline[bat] == dis)
                        bound_coordinate = np.nonzero(bound_sdf)
                        proto_bound = proto_featuremap[bat][bound_coordinate]
                        if proto_bound.shape[0] > 0:
                            mean_proto_bound = torch.mean(proto_bound, dim=0)
                            proto_num_list[cls] = proto_num_list[cls]  + 1
                            if proto_list is None:
                                proto_list = mean_proto_bound
                            else:
                                proto_list = torch.cat([proto_list, mean_proto_bound], dim=0)

                            preds_patch = preds_map[bat][bound_coordinate]
                            mean_preds = torch.mean(preds_patch, dim=0)
                            if preds_list is None:
                                preds_list = mean_preds
                            else:
                                preds_list = torch.cat([preds_list, mean_preds], dim=0)

    if proto_list is not None:
        # protolist predlist pronum torch.Size([3840]) torch.Size([30]) [15, 0]
        # print("protolist predlist pronum {} {} {}".format(proto_list.shape, preds_list.shape, proto_num_list))
        proto_list = proto_list.reshape(-1,fea_size)
        preds_list = preds_list.reshape(-1,numclasses)
        # protolist predlist pronum torch.Size([15, 256]) torch.Size([15, 2]) [15, 0]
        # print("protolist predlist pronum {} {} {}".format(proto_list.shape, preds_list.shape, proto_num_list))
    else:
        # map , gt  mask torch.Size([16, 56, 56, 256]) torch.Size([16, 1, 56, 56]) torch.Size([16, 56, 56, 2])
        # print("map , gt  mask {} {} {}".format(proto_featuremap.shape , target.shape , preds_map.shape))
        preds_list = None
        proto_num_list = [0 for _ in range(numclasses)]
        assert len(proto_featuremap.shape) == len(target.shape), f"{proto_featuremap.shape} , {target.shape}"
        proto_featuremap = proto_featuremap.permute(0 , 3 , 1,2)
        preds_map = preds_map.permute(0 , 3, 1, 2)
        for cls in range(0, numclasses):
            mask = (target[:labeled_bs]  == cls).float()
            cls_map = proto_featuremap * mask
            preds = preds_map * mask
            for bat in range(begin_labelbs, labeled_bs):
                sn = torch.sum(mask[bat])

                proto_num_list[cls] = proto_num_list[cls] + 1
                list_vec = torch.mean(preds[bat] , dim=[-2,-1])
                if preds_list is None:
                    preds_list = list_vec
                else:
                    preds_list = torch.cat([preds_list, list_vec], dim=0)

                # print("list_vec={}  {}".format(list_vec , list_vec.shape))
                if sn == 0:
                    cls_net_prototype = torch.zeros(fea_size).cuda()
                    # noise = torch.clamp(torch.randn_like(cls_net_prototype) * 1 + 0, -0.2, 0.2).cuda()  # 标准差 1 ，均值0
                    cls_net_prototype = F.normalize(cls_net_prototype, p=2, dim=0)
                    if proto_list is None:
                        proto_list = cls_net_prototype
                    else:
                        proto_list = torch.cat([proto_list, cls_net_prototype], dim=0)
                else:
                    cls_net_prototype = torch.sum(cls_map[bat], dim=[-2, -1]) / sn
                    cls_net_prototype = F.normalize(cls_net_prototype, p=2, dim=0)

                    if proto_list is None:
                        proto_list = cls_net_prototype
                    else:
                        proto_list = torch.cat([proto_list, cls_net_prototype], dim=0)


        proto_list = proto_list.reshape(-1, fea_size)
        preds_list = preds_list.reshape(-1, numclasses)

        # modify protois torch.Size([16, 256])  predlist torch.Size([16, 2])
        # print("modify protois {}  predlist {}".format(proto_list.shape , preds_list.shape))

    #
    #



    # bs , numclass , pnum , dim
    # proto_list.shape=torch.Size([8, 4, 2, 320])
    # print('proto_list.shape={}'.format(proto_list.shape))
    proto_list = F.normalize(proto_list, p=2, dim=-1)

    proto_list_map = {
        'proto_list': proto_list.cuda(),
        'proto_num_list': proto_num_list,
        'preds_list': preds_list.cuda()
    }
    return proto_list_map

def avg1d_for_cXk(feamap , nlist = [] , proto_num = 1 , ndim=20):

    # nlist = [5,0,2,6]  # 4 类， 没类有2个原型
    begin_n = 0
    # 将边界原型(NUM , 10) --> (4类 ， 2原型 ， 10)
    cXk_fea = torch.zeros((len(nlist) , proto_num , ndim)).cuda()
    for i , inum in enumerate(nlist):
        if inum == 0:
            pass
        elif inum < proto_num:
            meanfea = feamap[begin_n:begin_n+inum].mean(dim=0)
            newfeamap = torch.cat( [feamap[begin_n:begin_n+inum] , meanfea.repeat(inum % proto_num , 1) ] , dim=0)
            # newfeamap.shape = [h , chns] ,这样avg识别不了， 转成[1 , chns , h]
            newfeamap = newfeamap.permute(1, 0).unsqueeze(0)
            res1 = F.avg_pool1d(newfeamap, kernel_size=1, stride=1)
            # TMD ,还要把res1转回来
            res1 = res1.permute(0 , 2 ,1).squeeze(0)   #res1=torch.Size([3, 10])
            cXk_fea[i] = res1
        else:  #必定足够划分 , 减掉尾部
            newfeamap = feamap[begin_n:begin_n+inum - (inum % proto_num)].permute(1, 0).unsqueeze(0)
            res1 = F.avg_pool1d(newfeamap , kernel_size=inum // proto_num ,stride=inum // proto_num )
            res1 = res1.permute(0, 2, 1).squeeze(0)
            cXk_fea[i] = res1

        begin_n += inum

    return cXk_fea
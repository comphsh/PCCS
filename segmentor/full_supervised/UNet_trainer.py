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

from utils import losses, metrics, ramps, val_2d
from networks.net_factory import net_factory

from dataloaders.acdc import get_acdc_data_loader
from dataloaders.busi import get_busi_data_loader
from dataloaders.breast import get_breast_data_loader
from dataloaders.acdc_by_volume import get_acdc_by_volume_data_loader

def get_current_consistency_weight(args , epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def reverse_T(volume_batch  , rot_times):
    if abs(rot_times) < 4:
        image = torch.rot90(volume_batch, -rot_times, [-2, -1])
    else:
        if abs(rot_times) == 4:
            image = torch.flip(volume_batch, [2])  # 图像上下反转
        else:
            image = torch.flip(volume_batch, [3])  # 图像左右反转
    return image

def no_missinformation_T(volume_batch , label_batch):
    rot_times = random.randrange(0, 6)
    if rot_times < 4:
        # 左转90 次数
        image = torch.rot90(volume_batch, rot_times, [-2, -1])
        label = torch.rot90(label_batch, rot_times, [-2, -1])
    else:
        if rot_times == 4:
            image = torch.flip(volume_batch, [2])  # 图像上下反转
            label = torch.flip(label_batch, [2])
        else:
            image = torch.flip(volume_batch, [3])  # 图像左右反转
            label = torch.flip(label_batch, [3])
    return image , label , rot_times

def train(args, snapshot_path , cfg=None , trainloader=None, valloader=None, db_val_len=1):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=args.in_chns,class_num=num_classes , cfg=cfg)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,  momentum=0.9, weight_decay=0.0001)



    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    resize_size = cfg["dataset"]["kwargs"]["resize_size"]
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_train_perf = 0.0
    best_min_loss = 1000000
    iterator = tqdm(range(max_epoch), ncols=70)

    # from test_2d import show_single_mask_
    # for i_batch, sampled_batch in enumerate(valloader):
    #     print('hello')
    #     volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
    #
    #     for ind in range(len(sampled_batch['img_name'])):
    #         img_name = sampled_batch['img_name'][ind]
    #
    #         if img_name == 'benign (10)':
    #             # show_single_mask_(img * 255 ,label, FLAGS.num_classes)
    #             show_single_mask_(255 * volume_batch[ind].squeeze()[0, :, :], label_batch[ind].squeeze(), num_classes)
    #             exit()
    #         else:
    #             pass
    # exit()
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            print('hello')
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            if len(label_batch.shape) == 4:
                label_batch = label_batch.squeeze(1)

            # print("volume_batch = {}  {}".format(volume_batch.unique() , label_batch.unique()))

            if "semi" in cfg["dataset"]["exp"]:
                outputs_dict = model(volume_batch[:args.labeled_bs])
            elif 'full' in cfg["dataset"]["exp"]:
                outputs_dict = model(volume_batch)
            else:
                exit('not found exp')

            print("labels" , args.labeled_bs)
            if isinstance(outputs_dict , dict):
                outputs = outputs_dict['seg']
            elif isinstance(outputs_dict , tuple):
                outputs = outputs_dict[0]
            else:
                outputs = outputs_dict
            # output torch.Size([16, 2, 56, 56])  label torch.Size([16, 1, 224, 224])
            # print("output {}  label {}".format(outputs.shape , label_batch.shape))

            outputs = F.interpolate(outputs , size=label_batch.shape[-2:] , mode='bilinear' , align_corners=True)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs[:args.labeled_bs],label_batch[:args.labeled_bs][:].long())
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            consistency_weight = get_current_consistency_weight(args , iter_num // 150)

            # 监督损失
            supervised_loss = (loss_ce + loss_dice ) / 2
            loss = supervised_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # schedule
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            # writer.add_scalar('info/loss_contra', loss_contra, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item() ))


            if iter_num >= 0 and iter_num % 500 == 0:
                model.eval()
                performance, sec_val = val_val(model=model if not isinstance(model , nn.DataParallel) else model.module , valloader=valloader, writer=writer, cfg=cfg, num_classes=args.num_classes, db_val_len=db_val_len, iter_num=iter_num,   resize_size=resize_size)
                if performance > best_performance1:
                    best_performance1 = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_model1_{}_dice_{}.pth'.format(iter_num, round(  best_performance1, 4)))
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

                # val to train set
                # performance, sec_val = val_train(model=model if not isinstance(model, nn.DataParallel) else model.module,
                #                                valloader=trainloader, writer=writer, cfg=cfg,
                #                                num_classes=args.num_classes, iter_num=iter_num,
                #                                resize_size=resize_size)
                # if performance > best_train_perf:
                #     best_train_perf = performance
                #
                # if "2D" == cfg["dataset"]["dimension"]:
                #     logging.info( 'iteration %d : trainset : model1: mean_dice : %f mean_jaccard : %f' % (iter_num, performance, sec_val))
                # else:
                #     logging.info( 'iteration %d : trainset :  model1:  mean_dice : %f mean_hd95 : %f' % (iter_num, performance, sec_val))

                model.train()


            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


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
            #  torch.Size([1, 8, 256, 216]) torch.Size([1, 8, 256, 216])
            # print(sampled_batch["image"].unique() , sampled_batch["label"].unique() , sampled_batch["image"].shape ,sampled_batch["label"].shape ) #torch.Size([1, 11, 256, 216]) torch.Size([1, 11, 256, 216]) , 0到1之间，归一化，， label=0,1,2,3
            image_batch , label_batch = sampled_batch["image"], sampled_batch["label"]

            metric_i = test_single_volume(image_batch, label_batch, model, classes=num_classes , patch_size=resize_size)
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

def val_train(model=None, valloader=None, writer=None, cfg=None, num_classes=2 , iter_num = 1, resize_size=[256,256]):

    if "2D" == cfg["dataset"]["dimension"]:
        metric_list = np.zeros((num_classes - 1, 2))  # numclass类， 每类计算2个metric
        val_len = 0
        for _, sampled_batch in enumerate(valloader):
            data_batch, mask_batch = sampled_batch["image"].float().cuda(), sampled_batch["label"].long().cuda()
            for ind in range(data_batch.shape[0]):

                data = data_batch[ind].unsqueeze(0)
                mask = mask_batch[ind].unsqueeze(0)

                if len(mask.shape) == 4:
                    mask = mask.squeeze(1)

                _, _, h, w = data.shape
                input = F.interpolate(data, size=(resize_size[0], resize_size[1]), mode="bilinear")
                output = model(input)
                if isinstance(output, dict):
                    output = output["seg"]
                elif isinstance(output, tuple):
                    output = output[0]

                # <class 'torch.Tensor'> torch.Size([1, 2, 256, 256])
                # print(type(output) , output.shape)
                output = F.interpolate(output, mask.shape[-2:], mode="bilinear", align_corners=True)
                out = torch.argmax(output, dim=1).squeeze(1)  # output去掉第一维 变成 [bs , h , w]
                out = out.cpu().detach().numpy()
                pred = out
                # pred = zoom(out, (1, h / resize_size[0], w / resize_size[0]), order=0)  # 回归原来大小，val数据集没有经过transform
                # print("pred=" , pred.shape)  [bs , h , w]
                mask = mask.cpu().detach().numpy()
                val_len += mask.shape[0]
                assert pred.shape == mask.shape
                for bs in range(mask.shape[0]):
                    for cls in range(1, num_classes):  # 一个batch 同类算一次
                        val_i = cal_2d_metric(pred[bs, ...] == cls, mask[bs, ...] == cls)  # 这里只算平均，，每个类的值，到了每一类都在这了
                        metric_list[cls - 1] += np.array(val_i)

        metric_list = metric_list / val_len
        for class_i in range(num_classes - 1):
            writer.add_scalar('info/val_train_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
            writer.add_scalar('info/val_train_{}_jc'.format(class_i + 1), metric_list[class_i, 1], iter_num)

        performance = np.mean(metric_list, axis=0)[0]  # 所有0维(全部行)，加起来求均值
        sec_val = mean_jaccard = np.mean(metric_list, axis=0)[1]

        writer.add_scalar('info/val_train_mean_dice', performance, iter_num)
        writer.add_scalar('info/val_train_mean_jaccard', mean_jaccard, iter_num)

    elif "3D" == cfg["dataset"]["dimension"]:
        metric_list = np.zeros((num_classes - 1, 2))  # numclass类， 每类计算2个metric
        val_len = 0
        for _, sampled_batch in enumerate(valloader):
            data_batch, mask_batch = sampled_batch["image"].float().cuda(), sampled_batch["label"].long().cuda()
            for ind in range(data_batch.shape[0]):

                data = data_batch[ind].unsqueeze(0)
                mask = mask_batch[ind].unsqueeze(0)

                if len(mask.shape) == 4:
                    mask = mask.squeeze(1)

                _, _, h, w = data.shape
                input = F.interpolate(data, size=(resize_size[0], resize_size[1]), mode="bilinear")
                output = model(input)
                if isinstance(output, dict):
                    output = output["seg"]
                elif isinstance(output, tuple):
                    output = output[0]

                # <class 'torch.Tensor'> torch.Size([1, 2, 256, 256])
                # print(type(output) , output.shape)
                output = F.interpolate(output, mask.shape[-2:], mode="bilinear", align_corners=True)
                out = torch.argmax(output, dim=1).squeeze(1)  # output去掉第一维 变成 [bs , h , w]
                out = out.cpu().detach().numpy()
                pred = out
                # pred = zoom(out, (1, h / resize_size[0], w / resize_size[0]), order=0)  # 回归原来大小，val数据集没有经过transform
                # print("pred=" , pred.shape)  [bs , h , w]
                mask = mask.cpu().detach().numpy()
                val_len += mask.shape[0]
                assert pred.shape == mask.shape
                for bs in range(mask.shape[0]):
                    for cls in range(1, num_classes):  # 一个batch 同类算一次
                        val_i = cal_2d_metric(pred[bs, ...] == cls, mask[bs, ...] == cls)  # 这里只算平均，，每个类的值，到了每一类都在这了
                        metric_list[cls - 1] += np.array(val_i)

        metric_list = metric_list / val_len
        for class_i in range(num_classes - 1):
            writer.add_scalar('info/val_train_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
            writer.add_scalar('info/val_train_{}_jc'.format(class_i + 1), metric_list[class_i, 1], iter_num)

        performance = np.mean(metric_list, axis=0)[0]  # 所有0维(全部行)，加起来求均值
        sec_val = mean_jaccard = np.mean(metric_list, axis=0)[1]

        writer.add_scalar('info/val_train_mean_dice', performance, iter_num)
        writer.add_scalar('info/val_train_mean_jaccard', mean_jaccard, iter_num)
    else:
        exit("no found cal val!")


    return performance  , sec_val

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

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
            elif isinstance( output , tuple):
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

def contrast(embeding ,  label_batch , num_classes):
    bs , chns , h,w = embeding.shape
    scale_label = F.interpolate(label_batch.unsqueeze(1).float() , size=(h,w) , mode='nearest').squeeze(1).long()
    # enbebdtorch.Size([16, 256, 16, 16])
    embeding = embeding.permute(0 , 2,3 ,1)
    feature_list = None
    cls_num_list = []
    for cls in range(0 , num_classes):
        cls_mask = (scale_label == cls)
        feature_ = embeding[cls_mask , :]
        # print(feature_.shape , cls)
        if cls == 0:
            A = list(range(feature_.shape[0]))
            list_mask = torch.zeros(feature_.shape[0]).long()
            random.shuffle(A)
            list_mask[A[0:64]] = 1      #随机取64个进行对比
            # print("list_mask={} {} {}  {}".format(list_mask ,torch.sum(list_mask) , list_mask.shape  , feature_.shape))
            feature_ = feature_[list_mask == 1, :]
            print(feature_.shape, cls)

        if feature_list is None:
            feature_list = feature_
        else:
            feature_list = torch.cat([feature_list , feature_] , dim=0)
        cls_num_list.append(feature_.shape[0])

    # print("clsnulst={} {}  {}".format(cls_num_list  ,  np.sum(cls_num_list) , feature_list.shape)) #clsnulst=[64, 94, 60, 29] 247  torch.Size([247, 256])
    loss_contra = _loss_constrative(feature_list , cls_num_list)

    return loss_contra

def _loss_constrative(proto_list ,proto_num_list):

    # proto_list=torch.Size([543, 320])  proto_num_list=[a,b,c,d]
    assert len(proto_list.shape) == 2
    proto_list = F.normalize(proto_list , p=2 ,dim=-1)
    # torch.mul(a, b)是矩阵a和b对应位相乘,a和b的维度必须相等  torch.mm才是矩阵乘法
    logits = torch.div( torch.mm(proto_list , proto_list.T) , 0.05)

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
    # 消去自己和自己的内积

    exp_logits = torch.exp(logits) * IE_mask

    neg_mask = 1 - judge_positive_mask
    judge_positive_mask = judge_positive_mask * IE_mask  #不包括自身的正样本，
    postive_exp_logits = exp_logits * judge_positive_mask  ##不包括自身的正样本，


    neg_logits = exp_logits * neg_mask
    sum_neg_logits = neg_logits.sum(1, keepdim=True)

    # 分母
    log_prob = torch.log(postive_exp_logits + sum_neg_logits + 1e-8) - logits #翻过来，相当于-log(x/y) = log(y/x) = logy - logx
    log_prob = log_prob * judge_positive_mask
    # 每个样本对比的结果
    loss_contra = log_prob.mean(1)
    # 所有样本求平均
    loss_contra = loss_contra.mean()
    return loss_contra
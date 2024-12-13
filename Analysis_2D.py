import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from skimage.measure import label
from PIL import Image
from networks.net_factory import net_factory
import yaml
import glob
import cv2
import matplotlib.pyplot as plt
from BUSI_center_crop import crop_by_ratio
import matplotlib
# matplotlib.use('TkAgg')
from sklearn.metrics import accuracy_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='', help='Name of Experiment')  # if prostate num-class equals 2
parser.add_argument('--exp', type=str, default='', help='experiment_name')
parser.add_argument('--model', type=str, default='', help='model_name')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=-1, help='labeled data')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def pixel_four_metric(y_pred, y_true):
    # acc = (TP+TN)/(TP+TN+FP+FN)
    # recall = TP / (TP+FN)  #也 敏感性
    #     pre = TP / (TP+FP)
    # Specificity = TN / (TN + FP)

    # 将真实标签展平为一维数组
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # 计算像素准确率
    # accuracy = accuracy_score(y_true_flat, y_pred_flat)
    TP = np.sum(y_true_flat * y_pred_flat)
    TN = np.sum((1 - y_true_flat) * (1 - y_pred_flat))
    FP = np.sum((1 - y_true_flat) * y_pred_flat)  # 预测为正例，实际错了
    FN = np.sum((y_true_flat) * (1 - y_pred_flat))  # 而假负例是模型未能正确识别的正例。
    # precision_1 = metric.binary.precision(y_pred, y_true)
    # recall_1 = metric.binary.recall(y_pred , y_true)
    # specificity_1 = metric.binary.specificity(y_pred, y_true)

    # nan  nan  nan  nan  nan  nan  63.72  52.79  60.51  1.21  31.40  1.63  57.21  47.88  54.55  1.11  33.65  1.63  60.35  50.25  57.42  1.16  32.57  1.63
    # nan  nan  nan  nan  nan  nan  63.72  52.79  60.51  74.96  97.21  1.63  57.21  47.88  54.55  67.78  95.36  1.63  60.35  50.25  57.42  71.24  96.25  1.63
    #     nan  nan  nan  nan  nan  nan  63.72  52.79  60.51  74.96  97.21  96.78  57.21  47.88  54.55  67.78  95.36  94.83  60.35  50.25  57.42  71.24  96.25  95.77   吓死人

    pre = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # return precision_1, recall_1, specificity_1, accuracy
    return pre, recall, specificity, accuracy




# unet_ppcnt : [array([0.83447432, 0.72909656, 7.70549365, 2.40363324]), array([0.81210591, 0.68873811, 8.68602049, 2.24937856]), array([ 0.86750365,  0.78359178, 13.27629953,  4.6399703 ])]
# [0.83802796 0.73380882 9.88927122 3.0976607 ]
#  unet_trf ppcnt: [array([ 0.42744387,  0.28977571, 28.0925473 ,  7.1006574 ]), array([ 0.28835128,  0.17280659, 12.3143378 ,  4.43743559]), array([ 0.45557125,  0.31935285, 12.76129413,  4.65230611])]
# [ 0.39045547  0.26064505 17.72272641  5.3967997 ]



# plt 认为的顺序是 R  G   B

# color1 = [0, 255, 255]  #紫色
# # color2 = [255, 106, 106]
# color2 = [106, 106, 255]  #红色分量 106、绿色分量 106、蓝色分量 255
# color3 = [255, 250, 240]
#
# color4 = [255, 0, 0]   #纯色红
# color5 = [0 , 255 , 0]  #纯色绿
# color7 = [	127,255,0] #查特酒绿

# #cv2 认为的顺序是 B  G   Rcv B G R 颜色向量
color3 = [0, 255, 255]  # 黄色
color4 = [0, 0, 255]  # 纯色红
color5 = [0, 255, 0]  # 纯色绿
color6 = [255, 0, 255]  # 纯色蓝

# pred gt 分开mask时都用
color44 = [0, 69, 255]  # 橙红
color66 = [255, 144, 30]  # 道奇蓝
color7 = [0, 255, 127]  # 查特酒绿

# pred gt作为边界时 ，最好用纯色
color8 = [0, 255, 0]  # 纯绿
color9 = [0, 255, 255]  # 纯黄
color10 = [0, 0, 255]  # 纯红

from utils.boundary_func import img_to_sdf


def show_uncertainty_sdm(pred, uncertainty_map, img_name):
    plt.xticks([])
    plt.yticks([])
    # plt.imshow(uncertainty_map)
    # signed distance map
    # pred边界
    expred = np.expand_dims(pred, axis=0) == 1
    if np.sum(expred) > 0:
        sdf_pred = img_to_sdf(expred, expred.shape)
        sdf_pred = sdf_pred.squeeze(axis=0)
        print("sdf {} {}".format(np.unique(sdf_pred), sdf_pred.shape))
    else:
        print("pred . is null ")

        # 显示图形
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.tight_layout()
    # plt.figure(figsize=(6, 4))  # 设置宽度为6英寸，高度为4英寸
    save_options = dict(format='png', dpi=1200, bbox_inches='tight', pad_inches=0.0)

    stage_list = [500, 1000, 2000, 5000, 10000]
    state_time = stage_list[4]
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(pred )
    # plt.savefig(rf'pred_{img_name}_{state_time}.png', **save_options)
    # plt.clf()  # 清除当前图形

    plt.xticks([])
    plt.yticks([])

    ratio = 50
    minv = np.min(uncertainty_map)
    maxv = np.max(uncertainty_map)
    uncer_val = np.percentile(uncertainty_map.flatten(), 98.5)
    print("{} {} {} , min={}  max={} ".format(ratio, uncer_val, np.unique(uncertainty_map), minv, maxv))
    uncertainty_map[uncertainty_map < uncer_val] = minv

    plt.imshow(uncertainty_map)
    plt.savefig(rf'uncertainty_{img_name}_{state_time}.png', **save_options)
    plt.show()
    plt.clf()  # 清除当前图形

    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(sdf_pred)
    # plt.savefig(rf'sdf_{img_name}_{state_time}.png', **save_options)
    # plt.clf()  # 清除当前图形

    exit()


def show_uncertainty(uncertainty_map, img_name, cur_step):
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.tight_layout()
    # plt.figure(figsize=(6, 4))  # 设置宽度为6英寸，高度为4英寸
    save_options = dict(format='png', dpi=1200, bbox_inches='tight', pad_inches=0.0)
    state_time = cur_step

    # ratio = 50
    # minv = np.min(uncertainty_map)
    # maxv = np.max(uncertainty_map)
    # uncer_val = np.percentile(uncertainty_map.flatten(), 98.5)
    # print("{} {} {} , min={}  max={} ".format(ratio, uncer_val, np.unique(uncertainty_map), minv, maxv))
    # uncertainty_map[uncertainty_map < uncer_val] = minv

    plt.imshow(uncertainty_map)
    plt.savefig(rf'uncertainty_{img_name}_{state_time}.png', **save_options)
    plt.show()
    plt.clf()  # 清除当前图形

    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(sdf_pred)
    # plt.savefig(rf'sdf_{img_name}_{state_time}.png', **save_options)
    # plt.clf()  # 清除当前图形
    #
    # exit()


def show_pred_prob(pred_prob):
    # 显示图形
    # pred_prob = pred_prob.mean(dim=1)

    pred_prob = pred_prob[:, 1, ...]
    pred_prob = pred_prob.squeeze().cpu().numpy()
    # pred_prob =

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.tight_layout()
    # plt.figure(figsize=(6, 4))  # 设置宽度为6英寸，高度为4英寸
    save_options = dict(format='png', dpi=1200, bbox_inches='tight', pad_inches=0.0)
    plt.xticks([])
    plt.yticks([])
    print("pred_prob={}".format(pred_prob.shape))  # pred_prob=torch.Size([1, 2, 256, 256])
    # ratio = 50
    # minv = np.min(pred_prob)
    # maxv = np.max(pred_prob)
    # uncer_val = np.percentile(pred_prob.flatten(), ratio)
    # ratio[ratio < uncer_val] = minv

    plt.imshow(pred_prob)
    # plt.savefig(rf'uncertainty_{img_name}_{state_time}.png', **save_options)
    plt.show()
    plt.clf()  # 清除当前图形

    exit()


def show_single_mask_(image, label, num_classes, opacity=0.5, palette=[[0, 0, 0], color7, color66, color44]):
    assert len(image.shape) == 2 and len(label.shape) == 2
    sdf_label = img_to_sdf(np.expand_dims(label, axis=0), np.expand_dims(label, axis=0).shape)
    sdf_label = sdf_label.squeeze(axis=0)
    sdf_label_ord = np.where(sdf_label == 0)
    label_bound = np.zeros_like(sdf_label)
    label_bound[sdf_label_ord] = 1

    color_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    color_label_bound = np.zeros((label_bound.shape[0], label_bound.shape[1], 3), dtype=np.uint8)
    for cls in range(num_classes):
        color_label[cls == label, :] = palette[cls]  # 意思是找出这类像素，赋值颜色向量
        color_label_bound[cls == label_bound, :] = palette[cls]

    img_3D = np.stack([image] * 3, axis=-1)
    img_label = img_3D * (1 - opacity) + color_label * opacity  # img与pred_bound合并
    img_label = img_label.astype(np.uint8)

    img_labelbound = img_3D * (1 - opacity) + color_label_bound * opacity  # img与pred_bound合并
    img_labelbound = img_labelbound.astype(np.uint8)
    print("good")
    plt.imshow(img_label)
    print("hrere")
    plt.show()
    exit()


def save_result_hXw(img, label, pred, file_path,
                    img_name, num_classes, opacity=0.5, palette=[[0, 0, 0], color44, color7, color9],
                    bound_palatee=[[0, 0, 0], color66, color66, color66]):
    assert len(img.shape) == 2 and len(label.shape) == 2 and len(pred.shape) == 2
    # [0] [0 2 3] shape(256, 256) (256, 216) (256, 216)  ///     # (256,256)  (256,256)  (256,256)
    # print("{} {} {} shape{} {} {}".format(np.unique(img), np.unique(label), np.unique(pred), np.array(img).shape , np.array(label).shape , np.array(pred).shape))

    label_cls_bound = []
    pred_cls_bound = []
    for cls in range(1, num_classes):
        # gt 分割边界
        exlabel = np.expand_dims(label, axis=0) == cls
        if np.sum(exlabel) > 0:
            sdf_label = img_to_sdf(exlabel, exlabel.shape)
            sdf_label = sdf_label.squeeze(axis=0)
            sdf_label_ord = np.where(sdf_label == 0)
            label_bound = np.zeros_like(sdf_label)
            label_bound[sdf_label_ord] = 1

            label_cls_bound.append(label_bound)
        else:
            label_cls_bound.append(np.zeros(label.shape))

        # pred边界
        expred = np.expand_dims(pred, axis=0) == cls
        if np.sum(expred) > 0:
            sdf_pred = img_to_sdf(expred, expred.shape)
            sdf_pred = sdf_pred.squeeze(axis=0)
            sdf_pred_ord = np.where(sdf_pred == 0)
            pred_bound = np.zeros_like(sdf_pred)
            pred_bound[sdf_pred_ord] = 1

            pred_cls_bound.append(pred_bound)
        else:
            pred_cls_bound.append(np.zeros(pred.shape))

    label_cls_bound = np.array(label_cls_bound)
    pred_cls_bound = np.array(pred_cls_bound)  # [cls , h , w] 0 1 矩阵

    color_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    color_label_bound = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

    color_pred = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    color_pred_bound = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    # fig , axs = plt.subplots(3, 1)
    # axs[0].imshow(img , cmap="gray")
    # axs[1].imshow((label / num_classes) * 255.0 , cmap="gray")
    # axs[2].imshow((pred / num_classes) * 2555.0 , cmap="gray")
    # plt.show()

    for cls in range(1, num_classes):
        color_pred[cls == pred, :] = palette[cls]  # 意思是找出这类像素，复制颜色向量 ， mask用的颜色 ， 和边界用颜色不一样
        color_pred_bound[pred_cls_bound[cls - 1] == 1, :] = bound_palatee[cls]

        color_label[cls == label, :] = palette[cls]  # 意思是找出这类像素，复制颜色向量
        color_label_bound[label_cls_bound[cls - 1] == 1, :] = bound_palatee[cls]

    # print("{} {}".format(color_pred_bound.shape , img.shape)) #(256, 256, 3) (256, 256)
    # convert to BGR
    # color_seg = color_seg[..., ::-1]
    # img_3D = np.expand_dims(img , axis=2)
    # img_3D = np.repeat(img_3D , 3 , axis=2)
    img_3D = np.stack([img] * 3, axis=-1)
    label_3D = np.stack([label] * 3, axis=-1)
    pred_3D = np.stack([pred] * 3, axis=-1)
    img_predbound = img_3D * (1 - opacity) + color_pred_bound * opacity  # img与pred_bound合并
    img_predbound = img_predbound.astype(np.uint8)

    img_pred = img_3D * (1 - opacity) + color_pred * opacity  # img与pred合并
    img_pred = img_pred.astype(np.uint8)

    img_labelbound_pred = img_3D * (1 - opacity) + (
                0.5 * color_label_bound + 0.5 * color_pred) * opacity  # img与maskbound和predmask合并
    img_labelbound_pred = img_labelbound_pred.astype(np.uint8)

    img_label = img_3D * (1 - opacity) + color_label * opacity
    img_label = img_label.astype(np.uint8)

    # print("{} {} {} {} {}".format(img_3D.shape , img_label.shape , img_pred.shape , img_predbound.shape , img_labelbound_pred.shape))
    # print("img3D={} {}".format(np.unique(img_3D[:,:,0]) , np.unique(img_3D)))
    # plt.imshow(img_3D)  #plt擅长处理0-1之间的图
    # plt.imshow(img_label)
    # plt.imshow(img_pred)
    # plt.imshow(img_predbound)
    # plt.imshow(img_labelbound_pred)
    # fig, axs = plt.subplots(1, 1)
    # axs[0].imshow(img_3D)
    # axs[1].imshow(img_label)
    # axs[2].imshow(img_pred)
    # axs[3].imshow(img_predbound)
    # axs[4].imshow(img_labelbound_pred)

    # axs[0].axis('off')
    # axs[1].axis('off')
    # axs[2].axis('off')
    # axs[3].axis('off')
    # axs[4].axis('off')
    # plt.show()

    # img_3D / 255.0
    # label_3D / num_classes

    # cv2.imshow('a', pred_3D / num_classes)
    # cv2.waitKey(0)
    # exit()

    # dir_name = os.path.abspath(os.path.dirname(file_path))
    dir_name = file_path
    # os.makedirs(dir_name, exist_ok=True)
    # print("filepath={}".format(file_path))
    # print("dirname={}".format(dir_name))

    # if not os.path.exists(os.path.join(dir_name, 'img')):
    #     os.makedirs(os.path.join(dir_name, 'img'))
    # if not os.path.exists(os.path.join(dir_name, 'label')):
    #     os.makedirs(os.path.join(dir_name, 'label'))
    # if not os.path.exists(os.path.join(dir_name, 'pred')):
    #     os.makedirs(os.path.join(dir_name, 'pred'))
    if not os.path.exists(os.path.join(dir_name, 'img_label')):
        os.makedirs(os.path.join(dir_name, 'img_label'))
    if not os.path.exists(os.path.join(dir_name, 'img_pred')):
        os.makedirs(os.path.join(dir_name, 'img_pred'))
    if not os.path.exists(os.path.join(dir_name, 'img_labelbound_pred')):  # 2类以上数据集不适合
        os.makedirs(os.path.join(dir_name, 'img_labelbound_pred'))

    if not os.path.exists(os.path.join(dir_name, 'pred_gray')):
        os.makedirs(os.path.join(dir_name, 'pred_gray'))
    if not os.path.exists(os.path.join(dir_name, 'label_gray')):  # 2类以上数据集不适合
        os.makedirs(os.path.join(dir_name, 'label_gray'))

    # print("img_3D = {}  {}".format(np.unique(img_3D) , img_3D.shape))
    # cv2.imwrite(os.path.join(dir_name, 'img', f"{img_name}.png"), img_3D )
    # cv2.imwrite(os.path.join(dir_name, 'label', f"{img_name}.png"), label_3D / num_classes )
    # cv2.imwrite(os.path.join(dir_name, 'pred', f"{img_name}.png"), pred_3D / num_classes )
    cv2.imwrite(os.path.join(dir_name, 'img_label', f"{img_name}.png"), img_label)
    cv2.imwrite(os.path.join(dir_name, 'img_pred', f"{img_name}.png"), img_pred)
    cv2.imwrite(os.path.join(dir_name, 'img_labelbound_pred', f"{img_name}.png"), img_labelbound_pred)

    cv2.imwrite(os.path.join(dir_name, 'pred_gray', f"{img_name}.png"), pred * 255)
    cv2.imwrite(os.path.join(dir_name, 'label_gray', f"{img_name}.png"), label * 255)

def cal_2d_metric(pred, gt):  # 单张图像单个类的值
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        return dice, jaccard
    else:
        return 0, 0

def cal_2d_by_metricname(pred, gt, metric_name=None):  # 单张图像单个类的值
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    metric_val = np.zeros((len(metric_name)))

    pre, recall, specificity, accuracy = pixel_four_metric(pred, gt)
    if pred.sum() > 0:
        for i in range(len(metric_name)):
            val = 0.
            if metric_name[i] == 'Dice':
                val = metric.binary.dc(pred, gt)
            elif metric_name[i] == 'Jaccard':
                val = metric.binary.jc(pred, gt)
            elif metric_name[i] == 'HD95':
                val = metric.binary.hd95(pred, gt)
            # 'Precision' , 'Recall' , 'Specificity' , 'Accuracy'
            elif metric_name[i] == 'Precision':
                val = pre
            elif metric_name[i] == 'Recall':
                val = recall
            elif metric_name[i] == 'Specificity':
                val = specificity
            elif metric_name[i] == 'Accuracy':
                val = accuracy
            else:
                exit("not found metric function")
            metric_val[i] = val
        # print("metri_val == ", metric_val)

        return metric_val

    else:
        return metric_val







from test_2d_data_load import get_busi_data_loader, get_breast_data_loader



def Analysis_2D(cfg=None, FLAGS=None, output_path=None, pth_name=None, methods_name=None,
                divide_name=None, divide_result=None, metric_name='',resize_size=[0, 0],cur_step=None, loop_mode=''):
    data_mode = 'test'

    #读取单张图片
    if loop_mode == 'single_analysis':
        # L5_00031_US L4_00075_US  L2_n1010_T2_cut1 L2_n1044_T2_cut1 L3_n1058_T2_cut1  L3_n1176_T2_cut1 L4_n1060_T2_cut1 L4_n1037_T2_cut1 L5_n1013_T2_cut1
        img_name = 'L2_n1042_T2_cut1'
        img_path = os.path.join(FLAGS.root_path , data_mode , 'image' , f'{img_name}.png')
        mask_path = os.path.join(FLAGS.root_path , data_mode , 'mask' , f'{img_name}.png')
        img = Image.open(img_path).convert('L')  # signal channel
        mask = Image.open(mask_path)
        img = np.array(img) / 255.0
        mask = np.array(mask)
        # 放缩到0 和 1
        mask[mask <= np.max(mask) / 2.0] = 0  # 妥协写法
        mask[mask > np.max(mask) / 2.0] = 1

        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        print(img.shape , mask.shape)
        sample_dict = {'image': img, 'label': mask}
    else:
        pass
        # 批量数据集
        # if 'BUSI' in FLAGS.exp:
        #     data_loader = get_busi_data_loader(args=FLAGS, cfg=cfg, data_mode=data_mode)
        # elif 'breast' in FLAGS.exp:
        #     data_loader = get_breast_data_loader(args=FLAGS, cfg=cfg, data_mode=data_mode)

    net = net_factory(net_type=FLAGS.model, in_chns=FLAGS.in_chns, class_num=FLAGS.num_classes, cfg=cfg).cuda()
    save_model_path = f"{output_path}/{pth_name}"  # os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))#"/home/zxzhang/MC-Net-Main/data/ACDCSASSNetACDCclass3/data/ACDC_SASSNetACDC_7_labeled/unetsdf//iter_6200_dice_0.8938.pth"#os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))#os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))#os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    state_dict = torch.load(save_model_path)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict, strict=True)
    # print("init weight from {}".format(save_model_path))
    net.eval()

    # 测试不需要增强
    # input =torch.Size([1, 3, 224, 224])  label=torch.Size([1, 1, 224, 224])  srimg torch.Size([1, 256, 256, 3])  srlable=torch.Size([1, 256, 256]) imgname ['benign (111)']
    input, label = sample_dict['image'].float().cuda(), sample_dict['label']

    label = label.squeeze().numpy()
    input = F.interpolate(input.float(), size=resize_size, mode='bilinear', align_corners=True)

    assert resize_size[0] == input.shape[-2:][0] and resize_size[1] == input.shape[-2:][1], print(
        "resize={}  input={}".format(resize_size, input.shape))

    with torch.no_grad():
        out_dict = net(input)
        if isinstance(out_dict, dict):
            if 'seg' in out_dict:
                out_main = out_dict['seg']
            elif 'pred' in out_dict:
                out_main = out_dict['pred']
            elif 'out' in out_dict:
                out_main = out_dict['out']
        elif isinstance(out_dict, tuple):
            out_main = out_dict[0]
        else:
            out_main = out_dict
        out_main1 = F.interpolate(out_main, size=label.shape[-2:], mode="bilinear", align_corners=True)
        out_main = torch.softmax(out_main1, dim=1)
        out = torch.argmax(out_main, dim=1).squeeze(0).squeeze(1)
        out = out.cpu().detach().numpy()
        pred = out

        # pred = zoom(out , ( x / resize_size[0] , y /resize_size[1]) , order=0)
        # 预测后续操作
        # show_uncertainty_sdm()
        unet_uncertainty_map = -1.0 * torch.sum(out_main * torch.log(out_main), dim=1, keepdim=True)
        unet_uncertainty_map = unet_uncertainty_map.squeeze().cpu().numpy()
        show_uncertainty(unet_uncertainty_map, img_name, cur_step)


def Inference_metric(cfg, FLAGS, output_path, pth_name, methods_name, divide_name, divide_result, cur_step=0,
                     loop_mode=None):
    if cfg["dataset"][
        "dimension"] == "3D":  # 专指的是ACDC这种，有4类 ,也有两种显示方式，一种方式是变成tar.gz ,一种方式是，变成多个gt img pred img_gtbound_pred
        # metric, test_save_path = Inference_tar_gz(output_path ,pth_name, FLAGS, cfg=cfg)
        metric, test_save_path = Inference_dir_slices(output_path, pth_name, FLAGS, cfg=cfg,
                                                      methods_name=methods_name)  # 只是输出保存的方式不一样。。。
        print("methods_name=={}".format(methods_name))
        print("metric results=", end='  ')
        for item in metric:
            for pos in range(2):
                print("{:.2f}  ".format(100.0 * item[pos]), end='')
            for pos in range(2, len(item)):
                print("{:.2f}  ".format(item[pos]), end='')

        print("\naverage = ", end='  ')
        avg_item = (metric[0] + metric[1] + metric[2]) / 3
        for pos in range(2):
            val = 100.0 * avg_item[pos]
            print("{:.2f}  ".format(val), end=' ')
        for pos in range(2, len(avg_item)):
            val = avg_item[pos]
            print("{:.2f}  ".format(val), end=' ')
        print("\n")
        with open(test_save_path + '../performance.txt', 'w') as f:
            f.writelines('methods_name = {}metric is {} \n'.format(methods_name, metric))
            f.writelines('average metric is {}\n'.format((metric[0] + metric[1] + metric[2]) / 3))  # 4类的打印方式

    elif cfg["dataset"]["dimension"] == "2D":
        metric_name = ["Dice", "Jaccard", 'Precision', 'Recall', 'Specificity', 'Accuracy']
        if loop_mode is None or loop_mode == '':

            metric_list, avg_metric, test_save_path = Inference_2D(output_path, pth_name, FLAGS, metric_name,
                                                                   resize_size=[FLAGS.patch_size[0],
                                                                                FLAGS.patch_size[1]], cfg=cfg,
                                                                   methods_name=methods_name, divide_name=divide_name,
                                                                   cur_step=cur_step)
            print("methods_name=={}".format(methods_name))
            print(metric_list)
            print(avg_metric)
            divide_result.extend(avg_metric)
            with open(test_save_path + '../performance.txt', 'w') as f:
                f.writelines('methods_name={} metric is {} \n'.format(methods_name, metric_list))
                f.writelines('average metric is {}\n'.format(avg_metric))
        else:
            # #             不计算图像的metric，而是为了方便单张图像显示
            Analysis_2D(output_path, pth_name, FLAGS, metric_name,
                        resize_size=[FLAGS.patch_size[0], FLAGS.patch_size[1]], cfg=cfg,
                        methods_name=methods_name, divide_name=divide_name,
                        cur_step=cur_step)





if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    BU_MRI_merge_list = [
        # '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_4_2_11/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled_24_0304_1435/res_unet_bcp_3_attn/iter_model1_3600_dice_0.6409.pth',
        # '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_4_2_12/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled_24_0304_1435/res_unet_bcp_3_attn/iter_model1_8000_dice_0.6346.pth',
        # '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_4/model/breast_US1st_MRI2nd_no_SAG_T1_full_supervised_1.0_labeled/resunet/iter_model1_3200_dice_0.6509.pth',
        # '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_4/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled_24_0303_2040/resunet/iter_model1_2400_dice_0.6307.pth',
        # '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_2/model/breast_US1st_MRI2nd_no_SAG_T1_US_full_supervised_1.0_labeled/unet/iter_model1_8200_dice_0.5458.pth',
        # '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_4_2_10/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled_24_0303_2032/res_unet_bcp/iter_model1_11400_dice_0.6376.pth',
        # '/data/heshihuan/program/mapFromWin/BCP-work-005_4_2_13/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled/res_unet_scp/iter_model1_3200_dice_0.6206.pth',
        # '/data/heshihuan/program/mapFromWin/BCP-work-005_4_2_13/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled/std_unet_bcp/iter_model1_13000_dice_0.5978.pth',
        # '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_4_2_3/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled_24_0303_2040/res_unet_bcp/iter_model1_5400_dice_0.6467.pth'

        # '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_4/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled_24_0303_2040/resunet/iter_model1_2200_dice_0.6236.pth',
        '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_4_2_3/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled_24_0303_1001/res_unet_bcp/iter_model1_3000_dice_0.6419.pth',
        # '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_4_2_10/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled_24_0303_2032/res_unet_bcp/iter_model1_11400_dice_0.6376.pth'
    ]

    BU_MRI_merge_diff_stage_list = [
        '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_4_2_11/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled_24_0304_1435/res_unet_bcp_3_attn/iter_model1_3600_dice_0.6409.pth',
        '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_4_2_11/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled_24_0304_1435/res_unet_bcp_3_attn/iter_model1_3200_dice_0.6258.pth',
        '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_4_2_11/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled_24_0304_1435/res_unet_bcp_3_attn/iter_model1_1000_dice_0.5646.pth',
        '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_4_2_11/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled_24_0304_1435/res_unet_bcp_3_attn/iter_model1_400_dice_0.4887.pth',
        '/data/heshihuan/program/mapFromWin/BCP-series/BCP-work-005_4_2_11/model/breast_US1st_MRI2nd_no_SAG_T1_semi_0.1_labeled_24_0304_1435/res_unet_bcp_3_attn/iter_model1_200_dice_0.4246.pth',
    ]
    # 遍历多次取不确定性图
    times_list = [10000, 5000, 2000, 1000, 500]
    for pos, methods_output_path in enumerate(BU_MRI_merge_diff_stage_list):
        cur_step = times_list[pos]

        # divide_list = ['benign', 'malignant', '(']
        # divide_list = ['US' ]
        # divide_list =['US' ,'DWI' , 'SAG' , 'T1a' , 'T2' , 'L']
        # divide_list = ['L']
        divide_list = ['L']
        divide_result = []
        for divide_name in divide_list:
            output_path, pth_name = os.path.split(methods_output_path)
            print(methods_output_path)
            methods_name = output_path.split('/')[-1]
            yaml_file_list = glob.glob(f"{output_path}/*_config.yaml")
            print("flist={}".format(yaml_file_list))
            cfg = yaml.load(open(yaml_file_list[0], "r"), Loader=yaml.Loader)
            assert cfg is not None
            print("cfg={}".format(cfg))
            FLAGS.root_path = cfg["dataset"]["root_dir"]  # 1 74.80  65.00 , 2:
            # FLAGS.root_path = r'/data/heshihuan/datasets/BUS1st_MRI2nd_merge/BUS1st_MRI2nd_merge_all'

            FLAGS.model = cfg["network"]["net_type"]
            FLAGS.exp = cfg["dataset"]["exp"]  # ACDC_semi , BUSI_semi , breast_semi
            FLAGS.num_classes = cfg["dataset"]["num_classes"]
            FLAGS.labeled_num = cfg["dataset"]["kwargs"]["labeled_num"]
            FLAGS.batch_size_val = cfg["dataset"]["kwargs"]["batch_size_val"]
            FLAGS.patch_size = cfg['dataset']['kwargs']['resize_size']

            FLAGS.in_chns = cfg['dataset']['in_channels']
            print("args", FLAGS)
            Analysis_2D(cfg=cfg, FLAGS=FLAGS, output_path=output_path, pth_name=pth_name, methods_name=methods_name, divide_name=divide_name, divide_result=divide_result, resize_size=[FLAGS.patch_size[0] , FLAGS.patch_size[1]],
                        cur_step=cur_step, loop_mode='single_analysis')
            print("pth={}".format(methods_output_path))
            print('\n\n')



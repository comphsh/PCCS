from scipy.ndimage import distance_transform_edt as distance
import numpy as np
from skimage import segmentation as skimage_seg

def img_to_sdf(img_gt, out_shape, mode=None):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """
    #    ,,img_gt.shape=[8, 1, 256, 256] out_shape = (batchsize,h,w)
    # print("img_gt.shape={0}".format(img_gt.shape))

    img_gt = img_gt.astype(np.uint8)
    # normalized_sdf = np.zeros(out_shape)
    gt_sdf = np.zeros(out_shape)
    for b in range(out_shape[0]):  # batch size
        # posmask.shape = [1,256,256]
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            # [True,False] --> [False,True]
            negmask = ~posmask
            # distance 即 distance_transform_edt是scipy库里的一个函数，用于距离转换，计算图像中非零点到最近背景点（即0）的距离。
            posdis = distance(posmask)

            negdis = distance(negmask)

            # bool 转 uint8,true和1表示边界
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            # boundary = skimage_seg.find_boundaries(posmask, mode=mode ).astype(np.uint8)
            # negdis:背景距离边界/差值    posdis:分割内距离边界/差值
            # sdf对应背景为正，，对应分割为负
            # sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis))

            solid_sdf = negdis - posdis

            # boundary == 1的位置就是边界
            # sdf[boundary == 1] = 0

            solid_sdf[boundary == 1] = 0

            # print("sdf", sdf[0][xx[0]][yy[0]])
            # normalized_sdf[b] = sdf
            gt_sdf[b] = solid_sdf

    return  gt_sdf
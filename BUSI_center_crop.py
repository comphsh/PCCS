import cv2
import numpy as np
from PIL import Image
import math
from scipy.ndimage.interpolation import zoom
import cv2
import os
import matplotlib.pyplot as plt

# # 示例使用 这种是粗糙的切割，不是我们所写那种精细切割

# image = cv2.imread("your_image.png")  # 读取图像
# mask = cv2.imread("your_mask.png", cv2.IMREAD_GRAYSCALE)  # 读取掩码
#
# # 使用 crop_around_mask 进行切割
# cropped_image, cropped_mask = crop_around_mask(image, mask)
#
# # 保存切割结果
# cv2.imwrite("cropped_image.png", cropped_image)
# cv2.imwrite("cropped_mask.png", cropped_mask)

def crop_around_mask(image, mask, target_ratio=0.3):
    # 寻找掩码中心坐标
    center_y, center_x = np.where(mask > 0)  # row , col
    center_y, center_x = int(np.mean(center_y)), int(np.mean(center_x))

    # 计算切割框的大小
    h, w = image.shape[:2]
    target_h, target_w = int(h * target_ratio), int(w * target_ratio)

    # 计算切割框的左上角和右下角坐标
    start_y = max(0, center_y - target_h // 2)
    start_x = max(0, center_x - target_w // 2)
    end_y = min(h, start_y + target_h)
    end_x = min(w, start_x + target_w)

    # 切割图像和掩码
    cropped_image = image[start_y:end_y, start_x:end_x, ...]
    cropped_mask = mask[start_y:end_y, start_x:end_x]
    return cropped_image, cropped_mask

def cal_pic_pixel_lesion_ratio(data_path):
    mask = Image.open(data_path)
    mask = np.array(mask)
    print(mask.shape)
    mask[mask <= np.max(mask) /2.0] = 0
    mask[mask > np.max(mask) / 2.0] = 1
    print(np.unique(mask))
    print(np.sum(mask > 0))
    print(mask.shape[0]*mask.shape[1])
    val = np.sum(mask > 0) / (1.0 * mask.shape[0]*mask.shape[1])
    print("ratio={}".format(100.0 *val))

def kernel_crop_code(image , mask , pred , cur_ratio , target_ratio = 0.2 , resize=None):

    if cur_ratio >= target_ratio:
        target_image , target_mask , target_pred= image , mask , pred
        return target_image , target_mask , target_pred
    else :
        row_cord, col_cord = np.where(mask > 0)
        # print("row , col" , row_cord , col_cord)
        #     左上角是 [0，0]
        mini = np.min(row_cord)
        minj = np.min(col_cord)

        maxi = np.max(row_cord)
        maxj = np.max(col_cord)

        out_min_i = 0
        out_min_j = 0

        out_max_i = mask.shape[0] - 1
        out_max_j = mask.shape[1] - 1

        #         至少减少  S - x/0.2 个像素 才能让达到目标率
        demand_pixelnum = (mask.shape[0] * mask.shape[1] - np.sum(mask > 0) / target_ratio)
        demand_pixelnum = math.ceil(demand_pixelnum)  # 大于给定数字的第一个整数

        erase_num = 0  # 记录已经删除多少个数字
        while erase_num < demand_pixelnum:
            #                        上面行宽                左边列宽               下面行宽                   右边列宽
            nplist = np.array([abs(mini - out_min_i), abs(minj - out_min_j), abs(out_max_i - maxi), abs(out_max_j - maxj)])
            max_value = np.amax(nplist)
            max_indice = np.argwhere(nplist == max_value)  # 坐标，二维数据[n , 1]
            if (max_value <= 3):
                break  # 间隔太小，直接退出
            if resize is not None:
                if abs(out_max_i - out_min_i + 1) <= resize[0] or abs(out_max_j - out_min_j + 1) <= resize[1]:
                    break
            # 从多个最大值中随机挑选一个来消除
            pos = np.random.randint(0, len(max_indice))
            id = max_indice[pos][0]
            if id == 0:  # 减当前框 上面1行所有列个
                erase_num += (out_max_j - out_min_j + 1)
                out_min_i += 1

            elif id == 1:  # 减左边1列
                erase_num += (out_max_i - out_min_i + 1)
                out_min_j += 1

            elif id == 2:  # 减下面1行
                erase_num += (out_max_j - out_min_j + 1)
                out_max_i -= 1

            else:  # 减 右边1列
                erase_num += (out_max_i - out_min_i + 1)
                out_max_j -= 1

    # 最终出来的box就是要求的 病灶像素占 target_ratio的个数
    target_mask = mask[out_min_i:out_max_i + 1, out_min_j:out_max_j + 1]
    target_image = image[out_min_i:out_max_i + 1, out_min_j:out_max_j + 1]
    target_pred = pred[out_min_i:out_max_i+1 , out_min_j : out_max_j + 1]
    return target_image , target_mask , target_pred

def crop_by_ratio(image , mask , pred, target_ratio = 0.2 , resize = None):
    if np.sum(mask > 0) == 0:
        target_image, target_mask, target_pred = image , mask , pred
        if resize is not None:
            # print("resize")
            target_image = zoom(target_image, (resize[0] / image.shape[0], resize[1] / image.shape[1]), order=0)
            target_mask = zoom(target_mask, (resize[0] / mask.shape[0], resize[1] / mask.shape[1]), order=0)
            target_pred = zoom(target_pred, (resize[0] / pred.shape[0], resize[1] / pred.shape[1]), order=0)
        return target_image , target_mask ,target_pred

    cur_ratio = np.sum(mask > 0) / (1.0 * mask.shape[0] * mask.shape[1])

    if cur_ratio >= target_ratio:
        target_image , target_mask ,target_pred = image , mask , pred
    elif cur_ratio >= 0.15:
        target_image , target_mask,target_pred = kernel_crop_code(image , mask , pred, cur_ratio , target_ratio)
    elif cur_ratio >= 0.1:
        target_image, target_mask,target_pred = kernel_crop_code(image, mask, pred, cur_ratio , 0.15)
    elif cur_ratio >= 0.01:
        target_image, target_mask,target_pred = kernel_crop_code(image, mask, pred, cur_ratio, 0.1)
    else: #对于小于0.01， 翻10倍
        if cur_ratio * 10 <= 0.01:
            target_image, target_mask,target_pred = kernel_crop_code(image, mask, pred, cur_ratio, 0.01)
        elif cur_ratio * 10 <= 0.05:
            target_image, target_mask,target_pred = kernel_crop_code(image, mask, pred, cur_ratio, 0.05)
        else :
            target_image, target_mask,target_pred = kernel_crop_code(image, mask, pred, cur_ratio, 0.1)

    # plt.imshow(target_mask)
    # plt.show()
    # print("crop shape = {}".format(target_image.shape))
    # 再变成 [256,256]
    if resize is not None:
        # print("resize")  使用双线性插值时一定要小心
        target_image = zoom(target_image , (resize[0] / target_image.shape[0] , resize[1] / target_image.shape[1]) , order=0)
        target_mask = zoom(target_mask , (resize[0] / target_mask.shape[0] , resize[1] / target_mask.shape[1]) , order=0)
        target_pred = zoom(target_pred , (resize[0] / target_pred.shape[0] , resize[1] / target_pred.shape[1]) , order=0)

    # 双线性插值会把0-255 变成 0-1 ，
    # print("bilibli {} {}  {}".format(np.unique(target_image), np.unique(target_mask) , np.unique(target_pred)))
    # target_image = target_image.astype(np.uint8)  target只是用来显示，不是保存，不需要变成
    target_mask = target_mask.astype(np.uint8)
    target_pred = target_pred.astype(np.uint8)
    # print("bilibli {} {}  {}".format(np.unique(target_image), np.unique(target_mask) , np.unique(target_pred)))

    pval = 100.0 * cur_ratio
    sval = 100.0 *np.sum(target_mask > 0) / (1.0 * target_mask.shape[0] * target_mask.shape[1])
    # print("new shape , {} {} ratio pre={}　crop={} increase:{}倍" , target_image.shape , target_mask.shape , pval   , sval , sval / pval)
    return target_image , target_mask ,target_pred

def test_crop():
    datapath = r"E:\dataset\BUSI\archive_original\Dataset_BUSI_with_GT\malignant"  #1.73  8.49  12.60  30.3434 benign (262)_mask.png
    # cal_pic_pixel_lesion_ratio(datapath)
    imagepath = os.path.join(datapath  ,"malignant (145).png")
    name = imagepath.split(r"/")[-1]
    prename = name.split(".")[0]
    extend = name.split(".")[1]
    maskpath = os.path.join(datapath , f"{prename}_mask.{extend}")
    image = Image.open(imagepath).convert("L")
    mask = Image.open(maskpath)
    image = np.array(image)
    mask = np.array(mask)

    # <0.01的到达翻新十倍(如果还不到0.01，取0.01) ， 大于 0.01 小于0.1 翻新十倍？？
    # 最大限度是20%
    target_image , target_mask = crop_by_ratio(image, mask, target_ratio=0.2, resize=[144,144])

    fig , axs = plt.subplots(2 ,2 )
    axs[0][0].imshow(image , cmap="gray")
    axs[0][1].imshow(mask , cmap="gray")
    axs[1][0].imshow(target_image , cmap="gray")
    axs[1][1].imshow(target_mask, cmap="gray")
    plt.show()
    target_mask = np.stack([target_mask]*3 , axis=-1)
    print(target_mask.shape)
    # cv2.namedWindow("img" , cv2.WINDOW_NORMAL)
    # cv2.imshow('a', target_mask)
    # cv2.waitKey(1)

def crop_and_move(inputpath , outpath):
    imagepath = os.path.join(inputpath, "image")
    imgnamelist = os.listdir(imagepath)
    for i , item in enumerate(imgnamelist):
        prename = item.split(".")[0]
        extend = item.split(".")[1]
        imagefilepath = os.path.join(inputpath , "image" , f"{prename}.{extend}")
        maskfilepath = os.path.join(inputpath,"mask" ,f"{prename}_mask.{extend}")

        image = Image.open(imagefilepath).convert("L")
        mask = Image.open(maskfilepath)
        image = np.array(image)
        mask = np.array(mask)

        # <0.01的到达翻新十倍(如果还不到0.01，取0.01) ， 大于 0.01 小于0.1 翻新十倍？？
        # 最大限度是20%
        target_image, target_mask = crop_by_ratio(image, mask, target_ratio=0.2, resize=[256, 256])
        tg_img = Image.fromarray(target_image)
        tg_mask = Image.fromarray(target_mask)

        dirimg = os.path.join(outpath, "image")
        dirmask = os.path.join(outpath , "mask")
        if not os.path.exists( dirimg):
            os.makedirs(dirimg)
        if not os.path.exists(dirmask):
            os.makedirs(dirmask)

        img_outpath = os.path.join(outpath , "image" , f"{prename}.{extend}")
        mask_outpath = os.path.join(outpath ,"mask" , f"{prename}_mask.{extend}")
        print(img_outpath , mask_outpath)
        tg_img.save(img_outpath)
        tg_mask.save(mask_outpath)
        # print(target_image.shape ,target_mask.shape ,np.unique(target_image) , np.unique(target_mask))
        # exit()

if __name__ == "__main__":
    # malignant  benign
    #
    # inputpath = r"E:\dataset\BUSI\after_divided_image_mask\malignant"
    # outpath = r"E:\dataset\BUSI\aug__center_crop_256X256\orignal\malignant"

    # elastic_brightness_contras  flip_brightness_contras  gaussian_noise_brightness_contras  rotate_brightness_contras
    # shift_brightness_contras
    inputpath = r"E:\dataset\BUSI\after_aug_after_divided\malignant\shift_brightness_contras"
    outpath = r"E:\dataset\BUSI\aug__center_crop_256X256\aug\malignant\shift_brightness_contras"
    crop_and_move(inputpath , outpath)




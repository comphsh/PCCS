import math
import numpy as np
from torch import nn
import torch

from skimage.measure import label


def get_largest_cc(segmentation):
    labels = label(segmentation)
    if labels.max() == 0:
        return segmentation
    # assert (labels.max() != 0)  # assume at least 1 CC
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_cc

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class BaseModel2D(nn.Module):

    def __init__(self):
        super().__init__()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def inference(self, x, **kwargs):
        logits = self(x)
        preds = torch.argmax(logits['seg'], dim=1, keepdim=True).to(torch.float)
        return preds


class BaseModel3D(BaseModel2D):

    def __init__(self):
        super(BaseModel3D, self).__init__()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.backbone is not None:
            for m in self.backbone.modules():
                if isinstance(m, nn.Conv3d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def inference_slide3d(self,
                          images,
                          device='cuda',
                          turnoff_drop=False,
                          patch_size=(112, 112, 80),
                          stride_xy=18,
                          stride_z=4,
                          nms=True):

        images = images.squeeze(1)
        preds = []
        for image in images:
            image = image
            w, h, d = image.shape
            # 判断输入3D图像是否比patch尺寸小，若小则pad
            add_pad = False
            if w < patch_size[0]:
                w_pad = patch_size[0] - w
                add_pad = True
            else:
                w_pad = 0
            if h < patch_size[1]:
                h_pad = patch_size[1] - h
                add_pad = True
            else:
                h_pad = 0
            if d < patch_size[2]:
                d_pad = patch_size[2] - d
                add_pad = True
            else:
                d_pad = 0
            wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
            hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
            dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2

            if add_pad:
                image = np.pad(image.cpu().numpy(), [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
                               mode='constant', constant_values=0)
                image = torch.tensor(image).to(device)

            ww, hh, dd = image.shape
            # 根据stride确定3个方向上裁出patch的数量
            sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
            sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
            sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

            score_map = torch.zeros((self.num_classes,) + image.shape).to(torch.float).to(device)
            cnt = torch.zeros(image.shape).to(torch.float).to(device)

            # 预测每个patch的结果然后拼接到最终预测结果上
            for x in range(sx):
                xs = min(stride_xy * x, ww - patch_size[0])
                for y in range(sy):
                    ys = min(stride_xy * y, hh - patch_size[1])
                    for z in range(sz):
                        zs = min(stride_z * z, dd - patch_size[2])

                        test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                        test_patch = test_patch.unsqueeze(0).unsqueeze(0)

                        with torch.no_grad():
                            y_ = self.inference(test_patch)

                        # y = y[0].cpu().data.numpy()
                        score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                            = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y_[0]
                        cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                            = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1

            score_map = score_map / cnt.unsqueeze(0)
            label_map = score_map[0].to(torch.int).cpu().numpy()

            if add_pad:
                label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
            if nms:
                label_map = get_largest_cc(label_map)
            preds.append(label_map)

        return torch.tensor(preds).unsqueeze(1).to(device)

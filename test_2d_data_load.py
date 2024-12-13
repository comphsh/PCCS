
import random

from dataloaders.dataset import RandomGenerator, TwoStreamBatchSampler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from dataloaders._transforms import build_transforms
import torch

class BUSIDataSets(Dataset):
    def __init__(self , base_dir=None, split='train', num=None, transform=None , in_chns=1):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        # self.transform = build_transforms(transform)
        self.transform = transform
        self.in_chns = in_chns

        self.img_paths, self.mask_paths = self._build_path()


    def _build_path(self):
        img_root_dir = os.path.join(self._base_dir, self.split,'image')
        mask_root_dir = os.path.join(self._base_dir, self.split ,'mask')
        print("img={}".format(img_root_dir))
        image_list = os.listdir(img_root_dir)
        mask_list = os.listdir(mask_root_dir)

        ###--------检查image 和 mask是否一一对应

        image_list = sorted([item for item in image_list])
        mask_list = sorted([item for item in mask_list])
        print("sorted test list=", image_list)
        print("sorted test list=", mask_list)
        for i in range(len(image_list)):
            img = image_list[i]
            msk = mask_list[i]
            imgname = img.split(".")[0]
            maskname = msk.split(".")[0]
            if imgname not in maskname:
                print("not one to one reflection ,between img and mask , {} != {}".format(img, msk))
                exit()
        print("imglist == masklist")
        ###--------检查image 和 mask是否一一对应

        img_paths = [os.path.join(img_root_dir, img) for img in image_list]
        mask_paths = [os.path.join(mask_root_dir, mask) for mask in mask_list]
        return img_paths, mask_paths

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_name = img_path.split('/')[-1].split('.')[0]
        if self.in_chns == 1:
            img = Image.open(img_path).convert('L')  # signal channel
        elif self.in_chns == 3:
            img = Image.open(img_path).convert('RGB')  # signal channel , (256,256)-->(256,256,3)
        else:
            exit('not found in_chns')

        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path)
        img = np.array(img) / 255.0   #这里不归一，留待F.to_tensor归一,训练归一，验证不归一，后果很严重
        mask = np.array(mask)
        # 放缩到0 和 1
        mask[mask <= np.max(mask) / 2.0] = 0  # 妥协写法
        mask[mask > np.max(mask) / 2.0] = 1


        sample = {'image': img, 'label': mask}
        # 但是需要变换维度以适应测试
        if len(sample['image'].shape) == 2:
            sample['image'] = np.expand_dims(sample['image'], axis=0)

        if len(sample['label'].shape) == 2:
            sample['label'] = np.expand_dims(sample['label'], axis=0)

        sample['img_name'] = img_name
        return sample

    def get_img_list(self):
        return self.img_paths

    def __len__(self):
        return len(self.img_paths)

def get_busi_data_loader(args=None , cfg=None , data_mode='test'):

    # 用增强的训练83，用增强测试：82dice  ,  用增强训练83，用不增强的测试，87。。！！！！！！！！！！！
    print(args.patch_size)
    db_train = BUSIDataSets(base_dir=args.root_path, split=data_mode, num=None, in_chns=args.in_chns)

    loader = DataLoader(db_train, batch_size=1, shuffle=False,num_workers=1)
    return  loader


class BreastDataSets(Dataset):
    def __init__(self , base_dir=None, split='train', num=None, transform=None , in_chns=1):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform

        self.img_paths, self.mask_paths = self._build_path()


    def _build_path(self):
        img_root_dir = os.path.join(self._base_dir, self.split,'image')
        mask_root_dir = os.path.join(self._base_dir, self.split ,'mask')
        print("img={}".format(img_root_dir))
        image_list = os.listdir(img_root_dir)
        mask_list = os.listdir(mask_root_dir)

        ###--------检查image 和 mask是否一一对应
        image_list = sorted([item for item in image_list])
        mask_list = sorted([item for item in mask_list])

        print("sorted test list=", image_list)
        print("sorted test list=", mask_list)
        for i in range(len(image_list)):
            img = image_list[i]
            msk = mask_list[i]
            imgname = img.split(".")[0]
            maskname = msk.split(".")[0]
            if imgname not in maskname:
                print("not one to one reflection ,between img and mask , {} != {}".format(img, msk))
                exit()
        print("imglist == masklist")
        ###--------检查image 和 mask是否一一对应

        img_paths = [os.path.join(img_root_dir, img) for img in image_list]
        mask_paths = [os.path.join(mask_root_dir, mask) for mask in mask_list]
        return img_paths, mask_paths

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_name = img_path.split('/')[-1]

        img = Image.open(img_path).convert('L')  # signal channel
        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path)
        img = np.array(img) / 255.0
        mask = np.array(mask)
        # 放缩到0 和 1
        mask[mask <= np.max(mask) / 2.0] = 0  # 妥协写法
        mask[mask > np.max(mask) / 2.0] = 1

        sample = {'image': img, 'label': mask}

        # 但是需要变换维度以适应测试
        if len(sample['image'].shape) == 2:
            sample['image'] = np.expand_dims(sample['image'], axis=0)

        if len(sample['label'].shape) == 2:
            sample['label'] = np.expand_dims(sample['label'], axis=0)

        sample['img_name'] = img_name

        return sample

    def get_img_list(self):
        return self.img_paths

    def __len__(self):
        return len(self.img_paths)

def get_breast_data_loader(args=None , cfg=None , data_mode='test'):

    # 用增强的训练83，用增强测试：82dice  ,  用增强训练83，用不增强的测试，87。。！！！！！！！！！！！
    print(args.patch_size)
    db_train = BreastDataSets(base_dir=args.root_path, split=data_mode, num=None, in_chns=args.in_chns)

    loader = DataLoader(db_train, batch_size=16, shuffle=False,num_workers=4)
    return  loader

if __name__ == '__main__':
    pass

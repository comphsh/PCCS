import random

from dataloaders.dataset import RandomGenerator, TwoStreamBatchSampler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class BUSIDataSets(Dataset):
    def __init__(self , base_dir=None, split='train', num=None, transform=None , in_chns=1):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
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
        assert len(image_list) == len(mask_list) , print(len(image_list) , len(mask_list))
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
            img = Image.open(img_path).convert('RGB')  # signal channel
        else:
            exit('not found in_chns')

        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path)
        img = np.array(img) / 255.0
        mask = np.array(mask)
        # 放缩到0 和 1
        mask[mask <= np.max(mask) / 2.0] = 0  # 妥协写法
        mask[mask > np.max(mask) / 2.0] = 1

        # print("img={}  msak={}".format(np.unique(img) , np.unique(mask)))
        sample = {'image': img, 'label': mask}

        if self.split == "train":
            sample = self.transform(sample)
        sample['img_name'] = img_name
        return sample

    def get_img_list(self):
        return self.img_paths

    def __len__(self):
        return len(self.img_paths)

def get_busi_data_loader(args , cfg=None):
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BUSIDataSets(base_dir=args.root_path, split="train", num=None,  transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]) , in_chns=args.in_chns)

    db_val = BUSIDataSets(base_dir=args.root_path, split="val" , in_chns=args.in_chns)

    if args.labeled_num < 1.:
        label_num = int(len(db_train) * args.labeled_num)
        labeled_idxs = list(range(0,label_num))  #0-label_num
        unlabeled_idxs = list(range(label_num, len(db_train)))

        print("Total silices is: {}, labeled slices is: {}".format(len(db_train) , label_num))
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)
        trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=16,
                                  pin_memory=True, worker_init_fn=worker_init_fn)
    else:
        trainloader = DataLoader(db_train, batch_size=args.batch_size, num_workers=16,pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=1)

    return trainloader, valloader, len(db_val)



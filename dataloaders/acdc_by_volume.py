

import random

from dataloaders.dataset import RandomGenerator, TwoStreamBatchSampler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py
import numpy as np

# /data/heshihuan/datasets/ACDC/sup_and_semisup
class ACDCDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None , semi_mode="by_volume" ,label_ratio=-1):
        self._base_dir = base_dir
        self.sample_list = []

        self.split = split
        self.transform = transform
        if self.split == 'train':
            if semi_mode == 'by_volume':
                with open(self._base_dir + f'/sup_and_semisup/{label_ratio}/train_slice_sup.txt', 'r') as f1:
                    self.sup_list = f1.readlines()
                self.sup_list = [item.replace('\n', '') for item in self.sup_list]

                with open(self._base_dir + f'/sup_and_semisup/{label_ratio}/train_slice_unsup.txt', 'r') as f1:
                    self.unsup_list = f1.readlines()
                self.unsup_list = [item.replace('\n', '') for item in self.unsup_list]
                self.sample_list = self.sup_list + self.unsup_list

            else :
                with open(self._base_dir + '/train_slices.list', 'r') as f1:
                    self.sample_list = f1.readlines()
                self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def get_sup_num(self):
        return len(self.sup_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
      #  label[label==1]=0
      #  label[label==3]=0
    #    label[label==2]=1we
    #     print("acdc img={}  label={}".format(np.unique(image) , np.unique(label)))
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 47, "4": 111, "7": 191,
                    "11": 306, "14": 391, "18": 478, "35": 940}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_acdc_by_volume_data_loader(args , cfg=None):
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_val = ACDCDataSets(base_dir=args.root_path, split="val")
    if args.labeled_bs < args.batch_size:
        db_train = ACDCDataSets(base_dir=args.root_path, split="train", num=None,
                                transform=transforms.Compose([
            RandomGenerator(args.patch_size)
        ]) , semi_mode="by_volume" , label_ratio=args.labeled_num / 140)

        total_slices = len(db_train)
        labeled_slice = db_train.get_sup_num()

        print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
        labeled_idxs = list(range(0, labeled_slice))  # 0-512
        unlabeled_idxs = list(range(labeled_slice, total_slices))  # 512- ()
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

        trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                                 num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    else:

        db_train = ACDCDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
            RandomGenerator(args.patch_size)
        ]) , semi_mode="Nonesemi" , label_ratio=-1)

        trainloader = DataLoader(db_train , batch_size=args.batch_size ,num_workers=16 , pin_memory=True ,worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=1)
    return trainloader , valloader , len(db_val)


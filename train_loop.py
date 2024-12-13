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

import yaml
from dataloaders.acdc import get_acdc_data_loader
from dataloaders.acdc_by_volume import get_acdc_by_volume_data_loader
from dataloaders.acdc_train_val import get_acdc_train_val_data_loader
from dataloaders.busi import get_busi_data_loader
from dataloaders.breast import get_breast_data_loader

from segmentor import  PCCS_trainer
from segmentor.full_supervised import UNet_trainer

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/heshihuan/datasets/ACDC', help='Name of Experiment')

parser.add_argument('--exp', type=str,default='ACDC_semi', help='experiment_name')
# mcnet2d_v1  unet_cct unet_urpc unet_scp unet_ppcn  unet_mt  unet_uamt
# unet_cps unet_u2pl unet_em  unet_slc  unet_ssnet   unet_ugcl

parser.add_argument('--model', type=str,default='unet_ppcn', help='model_name')

parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')

parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=14,
                    help='labeled data')
# costs
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--gpu', type=str, default='2', help='GPU to use')
args = parser.parse_args()

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

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    from configPathSet import select_dataset_methods

    config_path_list = select_dataset_methods(nums=20)

    proj_path = os.path.dirname(__file__)
    for i in range(len(config_path_list)):
        torch.cuda.empty_cache()
        config_path = os.path.join(proj_path , config_path_list[i][0])
        print("configpath = {}".format(config_path))
        import logging
        cfg = yaml.load(open(config_path, "r"), Loader=yaml.Loader)

        args.root_path = cfg["dataset"]["root_dir"]
        args.model = cfg["network"]["net_type"]
        args.exp = cfg["dataset"]["exp"]  # ACDC_semi , BUSI_semi , breast_semi
        args.num_classes = cfg["dataset"]["num_classes"]
        cfg["lr"]["base_lr"] = args.base_lr   #最省力的更换学习率
        # cfg['loss']['consistency_rampup'] =args.consistency_rampup

        args.in_chns = cfg['dataset']['in_channels']
        cfg["dataset"]["kwargs"]["resize_size"] = [224 , 224]
        args.patch_size =  cfg["dataset"]["kwargs"]["resize_size"]
        args.labeled_num = cfg["dataset"]["kwargs"]["labeled_num"]
        args.labeled_bs = cfg["dataset"]["kwargs"]["labeled_bs"]
        args.batch_size = cfg["dataset"]["kwargs"]["batch_size"]
        args.batch_size_val = cfg["dataset"]["kwargs"]["batch_size_val"]

        args.max_iterations = 15000

        snapshot_path = "./model/{}_{}_labeled/{}".format(args.exp, args.labeled_num, args.model)
        snapshot_path = os.path.join(proj_path , snapshot_path)
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

        with open(config_path, "w") as file:
            yaml.dump(cfg, file, sort_keys = False , default_flow_style=False)
        print("cfg=",cfg)

        shutil.copy(config_path, snapshot_path)

        logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,  format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))

        if "ACDC" in args.exp:
            # trainloader, valloader, db_val_len = get_acdc_data_loader(args , cfg)
            trainloader, valloader, db_val_len = get_acdc_by_volume_data_loader(args)
            # trainloader, valloader, db_val_len = get_acdc_train_val_data_loader(args, cfg)
        elif "breast" in args.exp:
            trainloader, valloader, db_val_len = get_breast_data_loader(args, cfg)
        elif "BUSI" in args.exp:
            trainloader, valloader, db_val_len = get_busi_data_loader(args, cfg)
        else:
            logging.error("not found dataset!!")
            exit()

        # PPCNet_trainer.train(args, snapshot_path)  #  可以
        if "full_supervised" in config_path or "unet" in config_path:
            UNet_trainer.train(args , snapshot_path , cfg , trainloader, valloader, db_val_len)
        elif "pccs" in config_path:
            PCCS_trainer.train(args, snapshot_path, cfg, trainloader, valloader, db_val_len)
        else:
            print("error , not found model name")


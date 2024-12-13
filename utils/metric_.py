"""
preds.shape: (N,1,H,W) || (N,1,H,W,D) || (N,H,W) || (N,H,W,D)
labels.shape: (N,1,H,W) || (N,1,H,W,D) || (N,H,W) || (N,H,W,D)
"""
import torch
import numpy as np
from medpy import metric

class Dice:
    def __init__(self, name='Dice', class_indexs=[1], class_names=['xx']) -> None:
        super().__init__()
        self.name = name
        self.class_indexs = class_indexs
        self.class_names = class_names

    def __call__(self, preds, labels):
        res = {}
        for class_index, class_name in zip(self.class_indexs, self.class_names):
            preds_ = (preds == class_index).to(torch.int)
            labels_ = (labels == class_index).to(torch.int)
            intersection = (preds_ * labels_).sum()
            if  (preds_.sum() + labels_.sum()).item() > 0:
                res[class_name] = (2 * intersection) / (preds_.sum() + labels_.sum()).item()
            else :
                # res[class_name] = 0.0
                res[class_name] = torch.tensor(0.0,device='cuda')
            # res[class_name] = metric.dc(preds_.cpu().numpy(), labels_.cpu().numpy())
        return res


class Jaccard:
    def __init__(self, name='Jaccard', class_indexs=[1], class_names=['xx']) -> None:
        super().__init__()
        self.name = name
        self.class_indexs = class_indexs
        self.class_names = class_names

    def __call__(self, preds, labels):
        res = {}
        for class_index, class_name in zip(self.class_indexs, self.class_names):
            preds_ = (preds == class_index).to(torch.int)
            labels_ = (labels == class_index).to(torch.int)
            intersection = (preds_ * labels_).sum()
            union = ((preds_ + labels_) != 0).sum()
            if union > 0 :
                res[class_name] = intersection / union
            else :
                # res[class_name] = 0.0
                res[class_name] = torch.tensor(1.0, device='cuda')
                # res[class_name] = metric.jc(preds_.cpu().numpy(), labels_.cpu().numpy())
        return res
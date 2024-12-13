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
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        model.eval()
        with torch.no_grad():
            output = model(input)
            # if len(output)>1:
            #     # output0 = torch.Size([1, 4, 256, 256])
            #     # ouput1 = torch.Size([1, 65536, 4])
            #     # ouput1 = torch.Size([1, 65536])
            #     output = output[3].unsqueeze(1).reshape(output[0].shape[0],-1 , 256,256) + output[1].transpose(1,2).reshape(output[0].shape[0] ,-1, 256,256) + output[0] #.transpose(0,1).reshape(4,256,256)
            if isinstance( output , dict ):
                output = output["seg"]
            out = torch.argmax(output, dim=1).squeeze(0)#torch.sigmoid(output).squeeze()
            #out = out>0.5
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

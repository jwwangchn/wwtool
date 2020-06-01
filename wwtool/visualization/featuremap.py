import cv2
import numpy as np

import torch
import wwtool

def rescale(band, minval, maxval):
    band = 255 * (band - minval) / (maxval - minval)
    band = band.astype("int")
    band[band < 0] = 0
    band[band > 255] = 255
    return band

def show_featuremap(featuremap, win_name='feature_map'):
    if featuremap.type() == 'torch.FloatTensor':
        print("original feature map shape: ", featuremap.shape)
        featuremap = torch.mean(featuremap, dim=1)
        print("mean feature map shape: ", featuremap.shape)
        featuremap = featuremap.detach()
        featuremap = featuremap.numpy()[0]
        print("numpy feature map shape: ", featuremap.shape)
        print("max num: {}, min num: {}".format(featuremap.max(), featuremap.min()))
        min_num = np.percentile(featuremap, 5)
        max_num = np.percentile(featuremap, 90)
        print("percentile max: {}, percentile min: {}".format(max_num, min_num))
        featuremap = rescale(featuremap, min_num, max_num)
    elif featuremap.type() == 'torch.ByteTensor':
        featuremap = featuremap.detach()
        featuremap = featuremap.numpy()[0, 0]

    heatmap = wwtool.show_grayscale_as_heatmap(featuremap, win_name=win_name, wait_time=500, return_img=True)

    cv2.imwrite('/home/jwwangchn/Documents/Nutstore/100-Work/110-Projects/2019-DOTA/05-CVPR/supplementary/heatmap/save.png', heatmap)

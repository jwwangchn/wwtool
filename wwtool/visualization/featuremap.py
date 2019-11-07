import cv2
import numpy as np

import torch
import wwtool


def show_featuremap(featuremap, win_name='feature_map'):
    print("Feature map shape: ", featuremap.shape)
    print("Feature part value: ", featuremap[0, 0:2, 0:10, 0:10])
    if featuremap.type() == 'torch.FloatTensor':
        featuremap = torch.mean(featuremap, dim=1)
        featuremap = featuremap.detach()
        featuremap = featuremap.numpy()[0]
    elif featuremap.type() == 'torch.ByteTensor':
        featuremap = featuremap.detach()
        featuremap = featuremap.numpy()[0, 0]
    wwtool.show_grayscale_as_heatmap(featuremap, win_name=win_name, wait_time=10000)

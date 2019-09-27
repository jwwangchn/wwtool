import cv2
import numpy as np

import torch
import wwtool


def show_featuremap(featuremap, win_name='featuremap'):
    print("Feature map shape: ", featuremap.shape)
    if featuremap.type() == 'torch.FloatTensor':
        featuremap = torch.mean(featuremap, dim=1)
        featuremap = torch.sigmoid(featuremap)
        featuremap = featuremap.numpy()[0]
    elif featuremap.type() == 'torch.ByteTensor':
        featuremap = featuremap.numpy()[0]
    wwtool.show_grayscale_as_heatmap(featuremap, win_name)
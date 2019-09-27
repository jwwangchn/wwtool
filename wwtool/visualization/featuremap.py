import cv2
import numpy as np

import torch
import wwtool


def show_featuremap(featuremap):
    featuremap = torch.mean(featuremap, dim=1, keepdim=True)
    featuremap = featuremap.numpy()
    wwtool.show_grayscale_as_heatmap(featuremap)
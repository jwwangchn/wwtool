import numpy as np
import torch

from .bbox_overlaps_cpu import bbox_overlaps_cpu


def iou(bboxes1, bboxes2, mode='iou', is_aligned=False):
    if isinstance(bboxes1, torch.Tensor):
        bboxes1 = bboxes1.numpy()
        bboxes2 = bboxes2.numpy()
        is_numpy = False
    elif isinstance(bboxes1, np.ndarray):
        is_numpy = True
    else:
        raise TypeError('bboxes1, bboxes2 must be either a Tensor or numpy array, but got {}'.format(type(bboxes1)))

    ious = bbox_overlaps_cpu(bboxes1, bboxes2, mode, is_aligned)

    ious = torch.tensor(ious)
    
    return ious

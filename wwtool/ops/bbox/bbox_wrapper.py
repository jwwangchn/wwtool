import numpy as np
import torch

from .bbox_overlaps_cpu import bbox_overlaps_cpu


def iou(bboxes1, bboxes2, mode='iou', is_aligned=False):
    ious = bbox_overlaps_cpu(bboxes1, bboxes2, mode, is_aligned)

    return ious

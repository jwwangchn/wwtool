# cython: language_level=3, boundscheck=False

import numpy as np
cimport numpy as np
from libcpp cimport bool

def bbox_overlaps_cpu(
    np.ndarray[float, ndim=2] bboxes1,
    np.ndarray[float, ndim=2] bboxes2,
    str mode,
    bool is_aligned):
    """
    Args:
        bboxes1: (N, 4) ndarray of float
        bboxes2: (K, 4) ndarray of float
    Return:
        iou: (N, K) ndarry of overlap between bboxes1 and bboxes2
    """

    assert mode in ['iou', 'iof']

    cdef np.ndarray[float, ndim=1] area1, area2
    cdef np.ndarray[float, ndim=2] overlap, ious
    cdef np.ndarray[float, ndim=2] lt_aligned, rb_aligned, wh_aligned
    cdef np.ndarray[float, ndim=3] lt_no_aligned, rb_no_aligned, wh_no_aligned

    cdef unsigned int rows = bboxes1.shape[0]
    cdef unsigned int cols = bboxes2.shape[0]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        # cdef np.ndarray[float, ndim=2] lt, rb, wh
        lt_aligned = np.maximum(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb_aligned = np.minimum(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh_aligned = (rb_aligned - lt_aligned + 1).clip(min=0)  # [rows, 2]
        overlap = wh_aligned[:, 0] * wh_aligned[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt_no_aligned = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb_no_aligned = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh_no_aligned = (rb_no_aligned - lt_no_aligned + 1).clip(min=0)  # [rows, cols, 2]
        overlap = wh_no_aligned[:, :, 0] * wh_no_aligned[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])
    return ious
import cv2
import os.path as osp
import numpy as np
from mmcv.utils import is_str, mkdir_or_exist
from .color import color_val


def imshow_bboxes(img_or_path,
                  bboxes,
                  colors='red',
                  thickness=3,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """
    """
    if is_str(img_or_path):
        img = cv2.imread(img_or_path)
    else:
        img = img_or_path

    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    
    if bboxes.shape[0] == 1:
        bboxes = np.array([bboxes])

    colors = color_val(colors)

    for idx, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.int32)
        xmin, ymin, xmax, ymax = bbox

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=colors, thickness=thickness)

    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        dir_name = osp.abspath(osp.dirname(out_file))
        mkdir_or_exist(dir_name)
        cv2.imwrite(out_file, img)
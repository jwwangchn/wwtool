import cv2
import os.path as osp
import numpy as np

from mmcv.utils import is_str, mkdir_or_exist
from .color import color_val


def imshow_bboxes(img_or_path,
                  bboxes,
                  labels=None,
                  scores=None,
                  score_threshold=0.0,
                  colors='red',
                  thickness=3,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None,
                  origin_file=None,
                  return_img=False):
    """ Draw horizontal bounding boxes on image

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (list or ndarray): A ndarray of shape (N, 4)
        labels (list or ndarray): A ndarray of shape (N, 1)
        scores (list or ndarray): A ndarray of shape (N, 1)
    """
    if is_str(img_or_path):
        img = cv2.imread(img_or_path)
        img_origin = img.copy()
    else:
        img = img_or_path
        img_origin = img.copy()

    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)

    if bboxes.ndim == 1:
        bboxes = np.array([bboxes])

    if labels is None:
        labels_vis = np.array(['ins'] * bboxes.shape[0])
    else:
        labels_vis = np.array(labels)
        if labels_vis.ndim == 0:
            labels_vis = np.array([labels_vis])

    if scores is None:
        scores_vis = np.array([1.0] * bboxes.shape[0])
    else:
        scores_vis = np.array(scores)
        if scores_vis.ndim == 0:
            scores_vis = np.array([scores_vis])

    colors = color_val(colors)

    for bbox, label, score in zip(bboxes, labels_vis, scores_vis):
        if score < score_threshold:
            continue
        bbox = bbox.astype(np.int32)
        xmin, ymin, xmax, ymax = bbox

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=colors, thickness=thickness)
        if labels is not None:
            cv2.putText(img, label, (xmin, ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.0, color = colors, thickness = 2, lineType = 8)
        if scores is not None:
            cv2.putText(img, "{:.2f}".format(score), (xmin, ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.0, color = colors, thickness = 2, lineType = 8)

    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        dir_name = osp.abspath(osp.dirname(out_file))
        mkdir_or_exist(dir_name)
        cv2.imwrite(out_file, img)
    if origin_file is not None:
        dir_name = osp.abspath(osp.dirname(origin_file))
        mkdir_or_exist(dir_name)
        cv2.imwrite(origin_file, img_origin)
    if return_img:
        return img


def imshow_rbboxes(img_or_path,
                  rbboxes,
                  labels=None,
                  scores=None,
                  score_threshold=0.0,
                  colors='red',
                  thickness=3,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None,
                  return_img=False):
    """ Draw oriented bounding boxes on image

    Args:
        img (str or ndarray): The image to be displayed.
        rbboxes (list or ndarray): A ndarray of shape (N, 8)
        labels (list or ndarray): A ndarray of shape (N, 1)
        scores (list or ndarray): A ndarray of shape (N, 1)
    """
    if is_str(img_or_path):
        img = cv2.imread(img_or_path)
    else:
        img = img_or_path

    if isinstance(rbboxes, list):
        rbboxes = np.array(rbboxes)

    if rbboxes.ndim == 1:
        rbboxes = np.array([rbboxes])

    if labels is None:
        labels_vis = np.array(['ins'] * rbboxes.shape[0])
    else:
        labels_vis = np.array(labels)
        if labels_vis.ndim == 0:
            labels_vis = np.array([labels_vis])

    if scores is None:
        scores_vis = np.array([1.0] * rbboxes.shape[0])
    else:
        scores_vis = np.array(scores)
        if scores_vis.ndim == 0:
            scores_vis = np.array([scores_vis])

    colors = color_val(colors)

    for rbbox, label, score in zip(rbboxes, labels_vis, scores_vis):
        if score < score_threshold:
            continue
        rbbox = rbbox.astype(np.int32)

        cx = np.mean(rbbox[::2])
        cy = np.mean(rbbox[1::2])

        for idx in range(-1, 3, 1):
            cv2.line(img, (int(rbbox[idx*2]), int(rbbox[idx*2+1])), (int(rbbox[(idx+1)*2]), int(rbbox[(idx+1)*2+1])), colors, thickness=thickness)

        if labels is not None:
            cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.0, color = colors, thickness = 2, lineType = 8)
        if scores is not None:
            cv2.putText(img, "{:.2f}".format(score), (cx, cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.0, color = colors, thickness = 2, lineType = 8)

    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        dir_name = osp.abspath(osp.dirname(out_file))
        mkdir_or_exist(dir_name)
        cv2.imwrite(out_file, img)
    if return_img:
        return img

#TODO: show both ground truth and detection results

def show_centerness(centerness, 
                    show=True,
                    win_name='',
                    wait_time=0,
                    return_img=False):
    centerness_max = np.max(centerness)
    centerness_min = np.min(centerness)
    centerness = 255 * (centerness - centerness_min) / (centerness_max - centerness_min)
    centerness = centerness.astype(np.uint8)
    img_color = cv2.applyColorMap(centerness, cv2.COLORMAP_JET)

    if show:
        cv2.imshow(win_name, img_color)
        cv2.waitKey(wait_time)
    
    if return_img:
        return img_color
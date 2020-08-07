import cv2
import os.path as osp
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mmcv.utils import is_str, mkdir_or_exist
from .color import color_val
from wwtool.transforms import thetaobb2pointobb, impad, hobb2pointobb

def imshow_bboxes(img_or_path,
                  bboxes,
                  labels=None,
                  scores=None,
                  score_threshold=0.0,
                  show_label=False,
                  show_score=False,
                  thickness=2,
                  show=False,
                  win_name='',
                  wait_time=0,
                  out_file=None,
                  origin_file=None,
                  return_img=False):
    """ Draw horizontal bounding boxes on image

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (list or ndarray): A ndarray of shape (N, 4)
        labels (dict): {"category": idx}
        scores (list or ndarray): A ndarray of shape (N, 1)
    """
    if is_str(img_or_path):
        img = cv2.imread(img_or_path)
        img_origin = img.copy()
    else:
        img = img_or_path
        img_origin = img.copy()

    if len(bboxes) == 0:
        return

    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)

    if bboxes.ndim == 1:
        bboxes = np.array([bboxes])

    if labels is None:
        labels_vis = np.array(['ins'] * bboxes.shape[0])
    else:
        labels_vis = [list(label.keys())[0] for label in labels]
        class_idxs = [list(label.values())[0] for label in labels]
        max_label_idx = max(class_idxs)
        if max_label_idx > 20:
            max_label_idx = max_label_idx % 20
        color_dict = {list(label.keys())[0]:color_val(list(label.values())[0]) for label in labels}

    if scores is None:
        scores_vis = np.array([1.0] * bboxes.shape[0])
    else:
        scores_vis = np.array(scores)
        if scores_vis.ndim == 0:
            scores_vis = np.array([scores_vis])

    for bbox, label, score in zip(bboxes, labels_vis, scores_vis):
        if score < score_threshold:
            continue
        bbox = bbox.astype(np.int32)
        xmin, ymin, xmax, ymax = bbox

        if labels is None:
            current_color = color_val()
        else:
            current_color = color_dict[label]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=current_color, thickness=thickness)
        
        if show_label:
            cv2.putText(img, label, (xmin, ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.0, color = current_color, thickness = 2, lineType = 8)
        if show_score:
            cv2.putText(img, "{:.2f}".format(score), (xmin, ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.0, color = current_color, thickness = 2, lineType = 8)
    if show:
        # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(win_name, 360, 360)
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
                  color_name='red',
                  show_label=False,
                  show_score=False,
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

    if rbboxes == []:
        return

    if isinstance(rbboxes, list):
        rbboxes = np.array(rbboxes)
    
    if rbboxes.shape[1] == 5:
        rbboxes_ = []
        for rbbox in rbboxes:
            # rbboxes_.append(thetaobb2pointobb(rbbox))
            if abs(rbbox[-1]) <= 3.15:
                rbboxes_.append(thetaobb2pointobb(rbbox))
            else:
                rbboxes_.append(hobb2pointobb(rbbox))
        rbboxes = np.array(rbboxes_)
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

    if labels is None:
        colors = dict()
        colors[color_name] = color_val(color_name)
    else:
        max_label = 16
        colors = [color_val(_) for _ in range(max_label + 1)]

        # labels_vis = [list(label.keys())[0] for label in labels]
        # class_idxs = [list(label.values())[0] for label in labels]
        # max_label_idx = max(class_idxs)
        # if max_label_idx > 20:
        #     max_label_idx = max_label_idx % 20
        # color_dict = {list(label.keys())[0]:color_val(list(label.values())[0]) for label in labels}


    for rbbox, label, score in zip(rbboxes, labels_vis, scores_vis):
        if score < score_threshold:
            continue
        # if len(rbbox) == 5:
        #     rbbox = np.array(thetaobb2pointobb(rbbox))
        rbbox = rbbox.astype(np.int32)

        cx = np.mean(rbbox[::2])
        cy = np.mean(rbbox[1::2])

        if labels is None:
            current_color = color_val()
        else:
            current_color = colors[label]

        for idx in range(-1, 3, 1):
            cv2.line(img, (int(rbbox[idx*2]), int(rbbox[idx*2+1])), (int(rbbox[(idx+1)*2]), int(rbbox[(idx+1)*2+1])), current_color, thickness=thickness)

        if show_label:
            cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.0, color = current_color, thickness = 2, lineType = 8)
        if show_score:
            cv2.putText(img, "{:.2f}".format(score), (cx, cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.0, color = current_color, thickness = 2, lineType = 8)

    if show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 360, 360)
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        dir_name = osp.abspath(osp.dirname(out_file))
        mkdir_or_exist(dir_name)
        cv2.imwrite(out_file, img)
    if return_img:
        return img

def imshow_segms(img_or_path,
                  segms=None,
                  labels=None,
                  scores=None,
                  score_threshold=0.0,
                  colors='red',
                  thickness=3,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None,
                  return_img=False,
                  draw_contours=False):
    
    grayscale_image = img_or_path.astype(np.float64)
    max_value = np.max(grayscale_image)
    min_value = np.min(grayscale_image)
    grayscale_image = 255 * (grayscale_image - min_value) / (max_value - min_value)
    grayscale_image = grayscale_image.astype(np.uint8)

    grayscale_image = np.pad(grayscale_image, ((25, 25), (25, 25)), 'constant', constant_values = (0, 0))

    # grayscale_image[grayscale_image < 128] = 0

    ret, grayscale_image = cv2.threshold(grayscale_image, 128, 255, 0)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(grayscale_image.astype(np.uint8), connectivity=4)

    det = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        grayscale_image = cv2.dilate(grayscale_image, kernel)

        # make box
        np_contours = np.roll(np.array(np.where(grayscale_image!=0)),1,axis=0).transpose().reshape(-1,2)
        rect = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(grayscale_image, [box], 0, (0, 0, 255), 3)

        x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
        theta = theta * np.pi / 180.0
        thetaobb = [x, y, w, h, theta]
        print(thetaobb)
        pointobb = thetaobb2pointobb(thetaobb)
        print(pointobb)
        imshow_rbboxes(grayscale_image, pointobb, win_name='demo')

    # images, contours, hierarchy = cv2.findContours(grayscale_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if draw_contours:
    #     cv2.drawContours(grayscale_image, contours, -1, (0, 255, 0), 3)
    # if contours != []:
    #     imax_cnt_area = -1
    #     imax = -1
    #     for idx, cnt in enumerate(contours):
    #         cnt_area = cv2.contourArea(cnt)
    #         if imax_cnt_area < cnt_area:
    #             imax = idx
    #             imax_cnt_area = cnt_area
    #     cnt = contours[imax]
    #     rect = cv2.minAreaRect(cnt)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     cv2.drawContours(grayscale_image, [box], 0, (0, 0, 255), 3)

    #     x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
    #     theta = theta * np.pi / 180.0
    #     thetaobb = [x, y, w, h, theta]
    #     print(thetaobb)
    #     pointobb = thetaobb2pointobb(thetaobb)
    #     print(pointobb)
    #     imshow_rbboxes(grayscale_image, pointobb, win_name='demo')
        

#TODO: show both ground truth and detection results

def show_grayscale_as_heatmap(grayscale_image, 
                            show=True,
                            win_name='',
                            wait_time=0,
                            return_img=False):
    """show grayscale image as rgb image

    Args:
        grayscale_image (np.array): gray image
        show (bool, optional): show flag. Defaults to True.
        win_name (str, optional): windows name. Defaults to ''.
        wait_time (int, optional): wait time. Defaults to 0.
        return_img (bool, optional): return colored image. Defaults to False.

    Returns:
        np.array: colored image
    """
    grayscale_image = grayscale_image.astype(np.float64)
    max_value = np.max(grayscale_image)
    min_value = np.min(grayscale_image)
    grayscale_image = 255 * (grayscale_image - min_value) / (max_value - min_value)
    grayscale_image = grayscale_image.astype(np.uint8)
    heatmap_image = cv2.applyColorMap(grayscale_image, cv2.COLORMAP_JET)

    if show:
        cv2.imshow(win_name, heatmap_image)
        cv2.waitKey(wait_time)
    
    if return_img:
        return heatmap_image

def show_image(img, 
               win_name='',
               win_size=600,
               wait_time=0,
               save_name=None):
    """show image

    Args:
        img (np.array): input image
        win_name (str, optional): windows name. Defaults to ''.
        win_size (int, optional): windows size. Defaults to 800.
        wait_time (int, optional): wait time . Defaults to 0.
        output_file ([type], optional): save the image. Defaults to None.

    """
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, win_size, win_size)
    cv2.imshow(win_name, img)
    cv2.waitKey(wait_time)
    if save_name != None:
        cv2.imwrite(save_name, img)

    return img

def show_image_surface_curve(img, direction=0, show=True):
    """
    direction=0 -> height direction
    direction=1 -> width direction
    direction=2 -> all direction
    """
    if img.ndim == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape

    if direction == 2:
        X = np.arange(0, width, 1)
        Y = np.arange(0, height, 1)
        X, Y = np.meshgrid(X, Y)
        Z = img
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap='rainbow')
        data_x, data_y = None, None
    else:
        x = np.arange(0, img.shape[direction], 1)
        if direction == 0:
            y = img[img.shape[direction] // 2, x, ...]
        else:
            y = img[x, img.shape[direction] // 2, ...]
        plt.plot(x, y)
        data_x, data_y = x, y

    if show:
        plt.show()

    return data_x, data_y
    
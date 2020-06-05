from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import math
import pycocotools.mask as maskUtils

# ================== obb convert =======================

def pointobb2pointobb(pointobb):
    """
    docstring here
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
    """

    return pointobb.tolist()

def pointobb2thetaobb(pointobb):
    """
    docstring here
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    pointobb = np.int0(np.array(pointobb))
    pointobb.resize(4, 2)
    rect = cv2.minAreaRect(pointobb)
    x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
    theta = theta / 180.0 * np.pi
    thetaobb = [x, y, w, h, theta]
    
    return thetaobb

def thetaobb2pointobb(thetaobb):
    """
    docstring here
        :param self: 
        :param thetaobb: list, [x, y, w, h, theta]
    """
    box = cv2.boxPoints(((thetaobb[0], thetaobb[1]), (thetaobb[2], thetaobb[3]), thetaobb[4]*180.0/np.pi))
    box = np.reshape(box, [-1, ]).tolist()
    pointobb = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]

    return pointobb

def pointobb2bbox(pointobb):
    """
    docstring here
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
        return [xmin, ymin, xmax, ymax]
    """
    xmin = min(pointobb[0::2])
    ymin = min(pointobb[1::2])
    xmax = max(pointobb[0::2])
    ymax = max(pointobb[1::2])
    bbox = [xmin, ymin, xmax, ymax]
    
    return bbox


def thetaobb2hobb(thetaobb, pointobb_sort_fun):
    """
    docstring here
        :param self: 
        :param thetaobb: list, [x, y, w, h, theta]
    """
    pointobb = thetaobb2pointobb(thetaobb)
    sorted_pointobb = pointobb_sort_fun(pointobb)
    first_point = [sorted_pointobb[0], sorted_pointobb[1]]
    second_point = [sorted_pointobb[2], sorted_pointobb[3]]

    end_point = [sorted_pointobb[6], sorted_pointobb[7]]
    
    h = np.sqrt((end_point[0] - first_point[0])**2 + (end_point[1] - first_point[1])**2)

    hobb = first_point + second_point + [h]
    
    return hobb


def pointobb_extreme_sort(pointobb):
    """
    Find the "top" point and sort all points as the "top right bottom left" order
        :param self: self
        :param points: unsorted points, (N*8) 
    """   
    points_np = np.array(pointobb)
    points_np.resize(4, 2)
    # sort by Y
    sorted_index = np.argsort(points_np[:, 1])
    points_sorted = points_np[sorted_index, :]
    if points_sorted[0, 1] == points_sorted[1, 1]:
        if points_sorted[0, 0] < points_sorted[1, 0]:
            sorted_top_idx = 0
        else:
            sorted_top_idx = 1
    else:
        sorted_top_idx = 0

    top_idx = sorted_index[sorted_top_idx]
    pointobb = pointobb[2*top_idx:] + pointobb[:2*top_idx]
    
    return pointobb


def pointobb_best_point_sort(pointobb):
    """
    Find the "best" point and sort all points as the order that best point is first point
        :param self: self
        :param points: unsorted points, (N*8) 
    """
    xmin, ymin, xmax, ymax = pointobb2bbox(pointobb)
    w = xmax - xmin
    h = ymax - ymin
    reference_bbox = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
    reference_bbox = np.array(reference_bbox)
    normalize = np.array([1.0, 1.0] * 4)
    combinate = [np.roll(pointobb, 0), np.roll(pointobb, 2), np.roll(pointobb, 4), np.roll(pointobb, 6)]
    distances = np.array([np.sum(((coord - reference_bbox) / normalize)**2) for coord in combinate])
    sorted = distances.argsort()

    return combinate[sorted[0]].tolist()


def hobb2pointobb(hobb):
    """
    docstring here
        :param self: 
        :param hobb: list, [x1, y1, x2, y2, h]
    """
    first_point_x = hobb[0]
    first_point_y = hobb[1]
    second_point_x = hobb[2]
    second_point_y = hobb[3]
    h = hobb[4]

    angle_first_second = np.pi / 2.0 - np.arctan2(second_point_y - first_point_y, second_point_x - first_point_x)
    delta_x = h * np.cos(angle_first_second)
    delta_y = h * np.sin(angle_first_second)

    forth_point_x = first_point_x - delta_x
    forth_point_y = first_point_y + delta_y

    third_point_x = second_point_x - delta_x
    third_point_y = second_point_y + delta_y

    pointobb = [first_point_x, first_point_y, second_point_x, second_point_y, third_point_x, third_point_y, forth_point_x, forth_point_y]

    pointobb = [int(_) for _ in pointobb]
    
    return pointobb


def maskobb2thetaobb(maskobb):
    mask = maskUtils.decode(maskobb).astype(np.bool)
    gray = np.array(mask*255, dtype=np.uint8)
    contours = cv2.findContours(gray.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    if contours != []:
        imax_cnt_area = -1
        imax = -1
        for i, cnt in enumerate(contours):
            cnt_area = cv2.contourArea(cnt)
            if imax_cnt_area < cnt_area:
                imax = i
                imax_cnt_area = cnt_area
        cnt = contours[imax]
        rect = cv2.minAreaRect(cnt)
        x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
        thetaobb = [x, y, w, h, theta * np.pi / 180.0]
    else:
        thetaobb = [0, 0, 0, 0, 0]

    return thetaobb

# ================== flip obb =======================

def thetaobb_flip(thetaobbs, img_shape):
    """
    flip thetaobb
        :param self: 
        :param thetaobbs: np.array, [[x, y, w, h, theta]], (..., 5)
    """
    assert thetaobbs.shape[-1] % 5 == 0
    w = img_shape[1]
    flipped = thetaobbs.copy()
    flipped[..., 0] = w - flipped[..., 0] - 1
    flipped[..., [3, 2]] = flipped[..., [2, 3]]
    flipped[..., 4] = -math.pi/2.0 - flipped[..., 4]
    return flipped

def pointobb_flip(pointobbs, img_shape):
    """
    flip pointobbs
        :param self: 
        :param pointobbs: np.array, [[x1, y1, x2, y2, x3, y3, x4, y4]], (..., 8)
    """
    assert pointobbs.shape[-1] % 8 == 0
    pointobb_extreme_sort = False       # TODO: fix this when use the old sort method
    
    if pointobb_extreme_sort:
        w = img_shape[1]
        flipped = pointobbs.copy()
        flipped[..., 0::2] = w - flipped[..., 0::2] - 1
        flipped[..., [2, 6]] = flipped[..., [6, 2]]
        flipped[..., [3, 7]] = flipped[..., [7, 3]]
    else:
        w = img_shape[1]
        pointobbs_cp = pointobbs.copy()
        pointobbs_cp[..., 0::2] = w - pointobbs_cp[..., 0::2] - 1
        pointobbs_cp[..., [2, 6]] = pointobbs_cp[..., [6, 2]]
        pointobbs_cp[..., [3, 7]] = pointobbs_cp[..., [7, 3]]
        ndim_flag = False

        if pointobbs_cp.ndim == 1:
            ndim_flag = True
            pointobbs_cp = pointobbs_cp[np.newaxis, :]

        flipped = []
        for _ in pointobbs_cp:
            flipped.append(np.array(pointobb_best_point_sort(_.tolist())))

        flipped = np.array(flipped)
        if ndim_flag:
            flipped = flipped.squeeze()

    return flipped


def hobb_flip(hobbs, img_shape):
    """
    flip hobbs
        :param self: 
        :param hobbs: np.array, [[x1, y1, x2, y2, h]], (..., 5)
    """
    if hobbs.ndim == 1:
        hobbs = hobbs[np.newaxis, ...]
    assert hobbs.shape[-1] % 5 == 0
    w = img_shape[1]
    pointobbs = []
    for hobb in hobbs:
        pointobb = hobb2pointobb(hobb)
        pointobbs.append(pointobb)
    pointobbs = np.array(pointobbs)

    pointobb_extreme_sort = False       # TODO: fix this when use the old sort method
    
    if pointobb_extreme_sort:
        flipped = hobbs.copy()
        flipped[..., 4] = np.sqrt((flipped[..., 0] - flipped[..., 2])**2 + (flipped[..., 1] - flipped[..., 3])**2)
        flipped[..., 0] = w - flipped[..., 0] - 1
        flipped[..., 1] = flipped[..., 1]
        flipped[..., 2] = w - pointobbs[..., 6] - 1
        flipped[..., 3] = pointobbs[..., 7]
        flipped = flipped.squeeze()
    else:
        pointobbs = pointobb_flip(pointobbs, img_shape)
        thetaobbs = [pointobb2thetaobb(pointobb) for pointobb in pointobbs]
        hobbs = [thetaobb2hobb(thetaobb, pointobb_best_point_sort) for thetaobb in thetaobbs]
        flipped = np.array(hobbs)
        
    return flipped


# ================== rescale obb =======================

def thetaobb_rescale(thetaobbs, scale_factor, reverse_flag=False):
    """
    rescale thetaobb
        :param self: 
        :param thetaobb: np.array, [[x, y, w, h, theta]], (..., 5)
        :param reverse_flag: bool, if reverse_flag=Ture -> reverse rescale
    """
    if reverse_flag == False:
        thetaobbs[..., :-1] *= scale_factor
    else:
        thetaobbs[..., :-1] /= scale_factor
    return thetaobbs

def pointobb_rescale(pointobbs, scale_factor, reverse_flag=False):
    """
    rescale pointobb
        :param self: 
        :param pointobb: np.array, [[x, y, w, h, theta]], (..., 5)
        :param reverse_flag: bool, if reverse_flag=Ture -> reverse rescale
    """
    if reverse_flag == False:
        pointobbs *= scale_factor
    else:
        pointobbs /= scale_factor
    return pointobbs

def hobb_rescale(hobbs, scale_factor, reverse_flag=False):
    """
    rescale hobb
        :param self: 
        :param hobb: np.array, [[x, y, w, h, theta]], (..., 5)
        :param reverse_flag: bool, if reverse_flag=Ture -> reverse rescale
    """
    if reverse_flag == False:
        hobbs *= scale_factor
    else:
        hobbs /= scale_factor
    return hobbs

# def mask2quadrilateral(cnt):    
#     epsilon = 0.01 * cv2.arcLength(cnt, True)
#     epsilon_high, epsilon_low = cv2.arcLength(cnt, True), 0
#     approx = cv2.approxPolyDP(cnt, epsilon, True)
#     crt = approx.shape[0]
#     # epsilon 是从原始轮廓到近似轮廓的最大距离, 越小越精细, 边数越多
#     idx_high, idx_low = 1, 1
#     idx = 0
#     while crt != 4:
#         idx += 1
#         approx = cv2.approxPolyDP(cnt, epsilon, True)
#         crt = approx.shape[0]
        
#         if crt > 4:
#             idx_low += 1
#             epsilon_low = epsilon

#         elif crt < 4:
#             idx_high += 1
#             epsilon_high = epsilon

#         else:
#             break

#         if idx > 100000:
#             if crt <= 2:
#                 approx = np.zeros((4, 2))
#             if crt == 3:
#                 approx = np.vstack((approx.squeeze(), approx.squeeze()[2, :]))
#             if crt >= 5:
#                 approx = approx[:4, ::]
                    
#             break

#         epsilon = (epsilon_low + epsilon_high) / 2.0
#         #if idx < 20 or idx % 100000 == 0:
#          #   print("area:", cv2.contourArea(cnt))
#           #  print(idx, idx_high, idx_low, epsilon_high, epsilon_low, epsilon, crt)
#     return approx, crt

def mask2quadrilateral(cnt):
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    crt = approx.shape[0]
    eps = epsilon
    flag = 0
    eps_high_old = 0.0
    idx = 0
    eps_weight = 1.1
    idx_first = 0
    while crt > 4:
        idx += 1
        idx_first += 1
        eps_high_old = eps
        eps=eps_weight * eps
        if idx > 5000:
            eps_weight = eps_weight * 10
            idx = 0
        app = cv2.approxPolyDP(cnt, eps, True)
        crt = app.shape[0]
        flag = 0
        if idx_first > 10000000:
            return [], False

    eps_high = eps
    if crt < 4:
        idx = 0
        while eps_high_old <= eps_high:
            idx += 1
            eps = (eps_high_old + eps_high) / 2.0
            app = cv2.approxPolyDP(cnt, eps, True)
            crt = app.shape[0]
            if crt > 4:
                eps_high_old = eps
            if crt < 4:
                eps_high = eps
            else:
                break
    app = cv2.approxPolyDP(cnt, eps, True)
    return app, True

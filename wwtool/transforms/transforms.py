import cv2
import math
import numpy as np
import mmcv

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils


def segm2rbbox(segms):
    mask = maskUtils.decode(segms).astype(np.bool)
    gray = np.array(mask*255, dtype=np.uint8)
    images, contours, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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
        thetaobb = [x, y, w, h, theta]
        theta = theta * np.pi / 180.0
        pointobb = thetaobb2pointobb([x, y, w, h, theta])
            
    else:
        thetaobb = [0, 0, 0, 0, 0]
        pointobb = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    return thetaobb, pointobb


# ================== obb convert =======================
def pointobb2pointobb(pointobb):
    """
    docstring here
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
    """

    return pointobb.tolist()

def pointobb2thetaobb(pointobb):
    """convert pointobb to thetaobb
    Input:
        pointobb (list[1x8]): [x1, y1, x2, y2, x3, y3, x4, y4]
    Output:
        thetaobb (list[1x5])
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

def pointobb2sampleobb(pointobb, rate):
    """
    pointobb to sampleobb
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
        :param rate: 0 < rate < 0.5, rate=0 -> pointobb, rate=0.5 -> center point
        return [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8]
    """
    px1, py1, px2, py2, px3, py3, px4, py4 = pointobb

    sx1, sy1 = (px1 + px2) // 2, (py1 + py2) // 2
    sx2, sy2 = (px2 + px3) // 2, (py2 + py3) // 2
    sx3, sy3 = (px3 + px4) // 2, (py3 + py4) // 2
    sx4, sy4 = (px4 + px1) // 2, (py4 + py1) // 2

    sx5, sy5 = (1 - rate) * px2 + rate * px4, (1 - rate) * py2 + rate * py4
    sx6, sy6 = (1 - rate) * px3 + rate * px1, (1 - rate) * py3 + rate * py1
    sx7, sy7 = (1 - rate) * px4 + rate * px2, (1 - rate) * py4 + rate * py2
    sx8, sy8 = (1 - rate) * px1 + rate * px3, (1 - rate) * py1 + rate * py3

    sampleobb = [sx1, sy1, sx5, sy5, sx2, sy2, sx6, sy6, sx3, sy3, sx7, sy7, sx4, sy4, sx8, sy8]
    sampleobb = [int(point) for point in sampleobb]
    return sampleobb


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
    images, contours, hierarchy = cv2.findContours(gray.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
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

def bbox2pseudomask(mask_height, mask_width, bbox):
    """convert bbox to pseudomask
    
    Arguments:
        mask_height {int} -- height of mask
        mask_width {int} -- width of mask
        bbox {list, [1x4]} -- bbox as [xmin, ymin, xmax, ymax]
    
    Returns:
        numpy.ndarray, [mask_height, mask_width] -- generated pseudo mask
    """
    x_range = np.arange(0, mask_width)
    y_range = np.arange(0, mask_height)
    index_x, index_y = np.meshgrid(x_range, y_range)

    left = index_x - bbox[0]
    left = np.maximum(left, 0)
    right = bbox[2] - index_x
    right = np.maximum(right, 0)
    top = index_y - bbox[1]
    top = np.maximum(top, 0)
    bottom = bbox[3] - index_y
    bottom = np.maximum(bottom, 0)

    bbox_pseudo_mask = np.stack((left, top, right, bottom), -1)

    return bbox_pseudo_mask


def pointobb2pseudomask(mask_height, mask_width, pointobb):
    """convert pointobb to pseudo mask
    
    Arguments:
        mask_height {int} -- the height of mask
        mask_width {int} -- the widht of mask
        pointobb {list, [1x8]} -- [x1, y1, x2, y2, x3, y3, x4, y4]
    
    Returns:
        numpy.ndarry, [mask_height, mask_width] -- generated pseudo mask
    """
    thetaobb = pointobb2thetaobb(pointobb)
    pointobb = thetaobb2pointobb(thetaobb)

    rotation_anchor_x, rotation_anchor_y = thetaobb[0], thetaobb[1]
    theta = thetaobb[4]

    pointobb = rotate_pointobb(pointobb, -theta, [rotation_anchor_x, rotation_anchor_y])

    bbox = pointobb2bbox(pointobb)
    bbox_w, bbox_h, bbox_cx, bbox_cy = bbox[2] - bbox[0], bbox[3] - bbox[1], (bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2
    bbox = [(mask_width - bbox_w) // 2, (mask_height - bbox_h) // 2, (mask_width + bbox_w) // 2, (mask_height + bbox_h) // 2]
    move_x, move_y = bbox_cx - mask_width // 2, bbox_cy - mask_height // 2
    bbox_pseudomask = bbox2pseudomask(mask_height, mask_width, bbox)

    # convert pseudo to centerness
    left = bbox_pseudomask[..., 0]
    top = bbox_pseudomask[..., 1]
    right = bbox_pseudomask[..., 2]
    bottom = bbox_pseudomask[..., 3]
    
    centerness = np.sqrt((np.minimum(left, right) / (np.maximum(left, right) + 1)) * (np.minimum(top, bottom) / (np.maximum(top, bottom) + 1 )))
    centerness = mmcv.imrotate(centerness, theta * 180.0 / np.pi, center=(mask_width // 2, mask_height // 2))
    
    M = np.float32([[1, 0, move_x], [0, 1, move_y]])
    pointobb_pseudo_mask = cv2.warpAffine(centerness, M, (mask_height, mask_width))

    pointobb_pseudo_mask = pointobb_pseudo_mask.astype(np.float16)

    return pointobb_pseudo_mask


# ================== rotate obb ======================= 

def rotate_pointobb(pointobb, theta, anchor=None):
    """rotate pointobb around anchor
    
    Arguments:
        pointobb {list or numpy.ndarray, [1x8]} -- vertices of obb region
        theta {int, rad} -- angle in radian measure
    
    Keyword Arguments:
        anchor {list or tuple} -- fixed position during rotation (default: {None}, use left-top vertice as the anchor)
    
    Returns:
        numpy.ndarray, [1x8] -- rotated pointobb
    """
    if type(pointobb) == list:
        pointobb = np.array(pointobb)
    if type(anchor) == list:
        anchor = np.array(anchor).reshape(2, 1)
    v = pointobb.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:,:1]

    rotate_mat = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    res = np.dot(rotate_mat, v - anchor)
    
    return (res + anchor).T.reshape(-1)

# ================== flip obb =======================

def thetaobb_flip(thetaobbs, img_shape):
    """flip thetaobb
    
    Arguments:
        thetaobbs {numpy.ndarray, [..., 5]} -- theta based obb, can be one or two dimension
        img_shape {numpy.ndarray, [1x3]} -- thetaobb corresponding image shape
    
    Returns:
        numpy.ndarray -- fliped thetaobb
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


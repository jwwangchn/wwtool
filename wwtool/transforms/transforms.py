import cv2
import numpy as np

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

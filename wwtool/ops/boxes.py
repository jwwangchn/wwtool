import numpy as np
import cv2


def nms(boxes, scores, iou_threshold=0.5):
    """non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU)
    
    Arguments:
        boxes {np.array} -- [N * 4]
        scores {np.array} -- [N * 1]
        iou_threshold {float} -- threshold for IoU
    """

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    order = scores.argsort()[::-1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    keep = []
    while order.size > 0:
        best_box = order[0]
        keep.append(best_box)

        inter_x1 = np.maximum(x1[order[1:]], x1[best_box])
        inter_y1 = np.maximum(y1[order[1:]], y1[best_box])
        inter_x2 = np.minimum(x2[order[1:]], x2[best_box])
        inter_y2 = np.minimum(y2[order[1:]], y2[best_box])

        inter_w = np.maximum(inter_x2 - inter_x1 + 1, 0.0)
        inter_h = np.maximum(inter_y2 - inter_y1 + 1, 0.0)

        inter = inter_w * inter_h

        iou = inter / (areas[best_box] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        
        order = order[inds + 1]

    return keep

def rotation_nms(rboxes, scores, iou_threshold=0.5):
    """rotation non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU)
    
    Arguments:
        rboxes {np.array} -- [N * 5] (cx, cy, w, h, theta (rad/s))
        scores {np.array} -- [N * 1]
        iou_threshold {float} -- threshold for IoU
    """
    cx = rboxes[:, 0]
    cy = rboxes[:, 1]
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    theta = rboxes[:, 4] * 180.0 / np.pi

    order = scores.argsort()[::-1]

    areas = w * h
    
    keep = []
    while order.size > 0:
        best_rbox_idx = order[0]
        keep.append(best_rbox_idx)

        best_rbbox = np.array([cx[best_rbox_idx], 
                               cy[best_rbox_idx], 
                               w[best_rbox_idx], 
                               h[best_rbox_idx], 
                               theta[best_rbox_idx]])
        remain_rbboxes = np.hstack((cx[order[1:]].reshape(1, -1).T, 
                                    cy[order[1:]].reshape(1,-1).T, 
                                    w[order[1:]].reshape(1,-1).T, 
                                    h[order[1:]].reshape(1,-1).T, 
                                    theta[order[1:]].reshape(1,-1).T))

        inters = []
        for remain_rbbox in remain_rbboxes:
            rbbox1 = ((best_rbbox[0], best_rbbox[1]), (best_rbbox[2], best_rbbox[3]), best_rbbox[4])
            rbbox2 = ((remain_rbbox[0], remain_rbbox[1]), (remain_rbbox[2], remain_rbbox[3]), remain_rbbox[4])
            inter = cv2.rotatedRectangleIntersection(rbbox1, rbbox2)[1]
            if inter is not None:
                inter_pts = cv2.convexHull(inter, returnPoints=True)
                inter = cv2.contourArea(inter_pts)
                inters.append(inter)
            else:
                inters.append(0)

        inters = np.array(inters)
        iou = inters / (areas[best_rbox_idx] + areas[order[1:]] - inters)

        inds = np.where(iou <= iou_threshold)[0]
        
        order = order[inds + 1]

    return keep

def soft_nms(boxes, scores, iou_threshold=0.5, score_threshold=0.001):
    """soft non-maximum suppression (soft-NMS) on the boxes according to their intersection-over-union (IoU)
    
    Arguments:
        boxes {np.array} -- [N * 4]
        scores {np.array} -- [N * 1]
        iou_threshold {float} -- threshold for IoU
        score_threshold {float} -- threshold for score
    """

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    order = scores.argsort()[::-1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    keep = []
    while order.size > 0:
        best_box = order[0]
        keep.append(best_box)

        inter_x1 = np.maximum(x1[order[1:]], x1[best_box])
        inter_y1 = np.maximum(y1[order[1:]], y1[best_box])
        inter_x2 = np.minimum(x2[order[1:]], x2[best_box])
        inter_y2 = np.minimum(y2[order[1:]], y2[best_box])

        inter_w = np.maximum(inter_x2 - inter_x1 + 1, 0.0)
        inter_h = np.maximum(inter_y2 - inter_y1 + 1, 0.0)

        inter = inter_w * inter_h

        iou = inter / (areas[best_box] + areas[order[1:]] - inter)

        weights = np.ones(iou.shape) - iou

        scores[order[1:]] = weights * scores[order[1:]]

        inds = np.where(scores[order[1:]] > score_threshold)[0]
        
        order = order[inds + 1]

    return keep

def iou(bbox1, bbox2):
    """IoU of bbox1 and bbox2
    
    Arguments:
        bbox1 {np.array} -- [N, 4]
        bbox2 {np.array} -- [N, 4]
    """
    x1 = np.maximum(bbox1[:, 0], bbox2[:, 0])
    y1 = np.maximum(bbox1[:, 1], bbox2[:, 1])
    x2 = np.minimum(bbox1[:, 2], bbox2[:, 2])
    y2 = np.minimum(bbox1[:, 3], bbox2[:, 3])

    inter_w = np.maximum(x2 - x1 + 1, 0)
    inter_h = np.maximum(y2 - y1 + 1, 0)

    return inter_h * inter_w

def riou(rbbox1, rbbox2):
    """Rotation IoU of rbbox1 and rbbox2

    Arguments:
        rbbox1 {numpy.array} -- [N * 5], (cx, cy, w, h, angle (rad))
        rbbox2 {numpy.array} -- [N * 5], (cx, cy, w, h, angle (rad))
    """
    pass


if __name__ == '__main__':
    boxes = np.array([[100, 100, 210, 210],
                      [250, 250, 420, 420],
                      [220, 220, 320, 330],
                      [100, 100, 210, 210],
                      [230, 240, 325, 330],
                      [220, 230, 315, 340]], dtype=np.float32)
    
    rbbox = np.array([[100, 100, 210, 210, 180*np.pi/180],
                      [250, 250, 420, 420, 90*np.pi/180],
                      [220, 220, 320, 330, 45*np.pi/180],
                      [100, 100, 210, 210, 135*np.pi/180],
                      [230, 240, 325, 330, 60*np.pi/180],
                      [220, 230, 315, 340, 120*np.pi/180]], dtype=np.float32)

    scores = np.array([0.72, 0.8, 0.92, 0.72, 0.81, 0.9])

    keep_nms = nms(boxes, scores, iou_threshold=0.7)
    keep_soft_nms = soft_nms(boxes, scores, iou_threshold=0.3, score_threshold=0.01)
    print("nms and soft nms: ", keep_nms, keep_soft_nms)

    keep_nms = rotation_nms(rbbox, scores, iou_threshold=0.7)
    print("rotation nms: ", keep_nms)
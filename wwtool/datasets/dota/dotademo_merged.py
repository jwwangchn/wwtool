import numpy as np
import cv2
import os

def back_forward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)] 
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            theta = theta / 180.0 * np.pi
            # if theta < 0:
            #     theta = theta + np.pi * 2
            boxes.append([x, y, w, h, theta, rect[-1]])

    else:
        for rect in coordinate:
            box = np.int0(rect)
            box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)
            
            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            theta = theta / 180.0 * np.pi
            # if theta < 0:
            #     theta = theta + np.pi * 2
            boxes.append([x, y, w, h, theta])

    return np.array(boxes, dtype=np.float32)


def format_label(txt_list):
    format_data = []
    for i in txt_list[2:]:
        format_data.append([int(float(xy)) for xy in i.split(' ')[:8]]
        )
    return np.array(format_data)


src_path = './data/dota/v0/test/'
det_path = './results/dota_v100/submit/merge_dota_bbox'
vis_path = "./results/visual/display/v210_analysis"

vis = False
with_gt = True

res = dict()
for res_file in os.listdir(det_path):
    with open(os.path.join(det_path, res_file), 'r') as f:
        lines = f.readlines()
    
    splitlines = [x.strip().split(' ')  for x in lines]
    bboxes = []
    for splitline in splitlines:
        img_name = splitline[0]
        bbox = [float(_) for _ in splitline[2:]]
        res.setdefault(img_name, []).append(bbox)

for img_name, bboxes in res.items():
    img = cv2.imread(os.path.join(src_path, "images", img_name + '.png'))
    for bbox in bboxes:  
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(img, "De", (int((bbox[0] + bbox[2])//2), int((bbox[1] + bbox[3])//2)), 1, 2, (0, 255, 0))

    if with_gt:
        with open(os.path.join(src_path, "labelTxt-v1.0", img_name + '.txt'), 'r') as gt:
            lines = gt.readlines()
        gt_rbboxes = format_label(lines)

        for gt_rbbox in gt_rbboxes:
            xmin = round(min(gt_rbbox[::2]), 2)
            ymin = round(min(gt_rbbox[1::2]), 2)
            xmax = round(max(gt_rbbox[::2]), 2)
            ymax = round(max(gt_rbbox[1::2]), 2)
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            cv2.putText(img, "GT", ((xmin+xmax)//2, (ymin+ymax)//2), 1, 2, (0, 0, 255))
    
    if vis:
        cv2.imwrite(os.path.join(vis_path, img_name + '.png'), img)
    else:
        cv2.namedWindow("{}".format(img_name), cv2.WINDOW_NORMAL)
        cv2.imshow("{}".format(img_name), img)
        cv2.waitKey(0)
        cv2.destroyWindow("{}".format(img_name))

cv2.destroyAllWindows()

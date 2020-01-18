from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import math
import os
import wwtool

def draw_grid(img, line_color=(0, 255, 0), thickness=2, type_=cv2.LINE_AA, pxstep=80):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep

if __name__ == '__main__':
    coco_class = {   1: 'airplane', 
                       2: 'bridge', 
                       3: 'storage-tank', 
                       4: 'ship', 
                       5: 'swimming-pool', 
                       6: 'vehicle', 
                       7: 'person', 
                       8: 'wind-mill'}  

    pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    show_maskobb = False
    anno_file_name = ['small', 'trainval_test', 'v1', '1.0']

    imgDir = './data/{}/{}/coco/{}/'.format(anno_file_name[0], anno_file_name[2], anno_file_name[1])
    annFile='./data/{}/{}/coco/annotations/{}.json'.format(anno_file_name[0], anno_file_name[2], "_".join(anno_file_name))

    coco=COCO(annFile)

    catIds = coco.getCatIds(catNms=" ")
    imgIds = coco.getImgIds(catIds=catIds)

    line_start_x_point = np.arange(0, 800, 80)
    line_start__point = np.arange(0, 800, 80)


    for idx, imgId in enumerate(imgIds):
        img = coco.loadImgs(imgIds[idx])[0]

        # if img['file_name'] != '0000001_03499_d_0000006__1120_280.png':
        #     continue

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        # print("idx: {}, image file name: {}".format(idx, img['file_name']))

        if show_maskobb:
            pass
        else:
            im = cv2.imread(imgDir + img['file_name'])
            bboxes = []
            labels = []
            labels_set = set()
            for ann in anns:
                bbox = ann['bbox']
                label = coco_class[ann['category_id']]
                bbox = wwtool.xywh2xyxy(bbox)
                bboxes.append(bbox)
                labels.append({label:ann['category_id']})
                labels_set.add(ann['category_id'])

            if 8 not in labels_set:
                continue
            else:
                print("idx: {}, image file name: {}".format(idx, img['file_name']))

            # draw_grid(im, pxstep=160)
            out_file = os.path.join('/data/small/v1/vis', img['file_name'])
            wwtool.imshow_bboxes(im, bboxes, labels=labels, show_label=False, wait_time=10, out_file=out_file)
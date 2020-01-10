from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import math
import os
import wwtool

if __name__ == '__main__':
    coco_class = {'harbor': 1, 'ship': 2, 'small-vehicle': 3, 'large-vehicle': 4, 'storage-tank': 5, 'plane': 6, 'soccer-ball-field': 7, 'bridge': 8, 'baseball-diamond': 9, 'tennis-court': 10, 'helicopter': 11, 'roundabout': 12, 'swimming-pool': 13, 'ground-track-field': 14, 'basketball-court': 15}

    pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    show_maskobb = False
    anno_file_name = ['dota-v1.5', 'v1', 'trainval']

    img_dir = './data/{}/images'.format('/'.join(anno_file_name))
    anno_dir='./data/{}/labelTxt-v1.5'.format('/'.join(anno_file_name))

    for anno_name in os.listdir(anno_dir):
        file_name = os.path.splitext(os.path.basename(anno_name))[0]
        anno_file = os.path.join(anno_dir, anno_name)
        objects = wwtool.dota_parse(anno_file)

        image_name = os.path.join(img_dir, file_name + '.png')
        
        im = cv2.imread(image_name)
        bboxes = []
        labels = []
        for obj in objects:
            bbox = obj['bbox']
            label = obj['label']
            if label != 'harbor':
                continue
            bboxes.append(bbox)
            labels.append({label:coco_class[label]})
        
        wwtool.imshow_bboxes(im, bboxes, labels=labels, show_label=False, wait_time=100)
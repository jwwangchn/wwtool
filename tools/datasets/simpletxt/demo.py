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
    coco_class = { 'airplane':                     1, 
                    'bridge':                       2,
                    'storage-tank':                 3, 
                    'ship':                         4, 
                    'swimming-pool':                5, 
                    'tennis-court':                 6, 
                    'vehicle':                      7, 
                    'person':                       8, 
                    'harbor':                       9, 
                    'wind-mill':                    10}

    pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    show_maskobb = False
    anno_file_name = ['small', 'dota-v1.5']

    img_dir = './data/{}/{}/images'.format(anno_file_name[0], anno_file_name[1])
    anno_dir='./data/{}/{}/labels'.format(anno_file_name[0], anno_file_name[1])

    for anno_name in os.listdir(anno_dir):
        file_name = os.path.splitext(os.path.basename(anno_name))[0]
        anno_file = os.path.join(anno_dir, anno_name)
        objects = wwtool.simpletxt_parse(anno_file)

        image_name = os.path.join(img_dir, file_name + '.png')
        
        im = cv2.imread(image_name)
        bboxes = []
        labels = []
        for obj in objects:
            bbox = obj['bbox']
            label = obj['label']
            bboxes.append(bbox)
            labels.append({label:coco_class[label]})
        
        wwtool.imshow_bboxes(im, bboxes, labels=labels, show_label=False, wait_time=10000)
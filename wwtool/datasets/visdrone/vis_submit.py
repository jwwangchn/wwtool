from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import math
import os

from wwtool.visualization import imshow_bboxes

imgpath = '/media/jwwangchn/data/visdrone/v1/coco/test/'
anno_path = '/home/jwwangchn/Downloads/WHU'

for label_file in os.listdir(anno_path):
    labels = open(os.path.join(anno_path, label_file), 'r').readlines()
    
    im = cv2.imread(os.path.join(imgpath, label_file.split('.')[0] + '.jpg'))
    
    bboxes = []

    for label in labels:
        xmin, ymin, bbox_w, bbox_h = [float(xy) for xy in label.split(',')[:4]]
        xmax, ymax = xmin + bbox_w, ymin + bbox_h
        bbox = xmin, ymin, xmax, ymax
        bboxes.append(bbox)
        # cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 0), 3)
    
    imshow_bboxes(im, bboxes, colors='red', wait_time=1000)

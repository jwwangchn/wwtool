from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import math
import os

from mmdet.datasets.dota.cocoseg import cocoSegmentationToPng

from mmdet.datasets.dota.transform import pointobb_flip, thetaobb_flip, hobb_flip
from mmdet.datasets.dota.transform import pointobb_rescale, thetaobb_rescale, hobb_rescale

def draw_rectangle_by_points(im, points):
    """
    docstring here
        :param points: [x,y,...] (1*8) 
    """
    for idx in range(-1, 3, 1):
        cv2.line(im, (int(points[idx*2]), int(points[idx*2+1])), (int(points[(idx+1)*2]), int(points[(idx+1)*2+1])), (0, 0, 255), 3)
    return im

def show_bbox(imgDir, img, anns):
    im = cv2.imread(imgDir + img['file_name'])
    for ann in anns:
        bbox = ann['bbox']
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255), 3)

    cv2.imshow('demo', im)
    cv2.waitKey(10000)

def show_maskobb(imgDir, img, anns):
    I = io.imread(imgDir + img['file_name'])
    plt.imshow(I); 
    coco.showAnns(anns)
    plt.show()

def show_pointobb(imgDir, img, anns):
    im = cv2.imread(imgDir + img['file_name'])
    flip = np.random.choice([True, False])
    scale = np.random.uniform(0.8, 1.0, 1)[0]
    im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    print("pointobb flip: {}, pointobb scale: {}".format(flip, scale))
    if flip:
        im = cv2.flip(im, 1)
    for ann in anns:
        pointobb = ann['pointobb']
        pointobb = np.array(pointobb)
        pointobb = pointobb_rescale(pointobb, scale, reverse_flag=False)
        if flip:
            img_shape = im.shape
            pointobb = pointobb_flip(pointobb, img_shape)
        for idx in range(4):
            if idx == 0:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            cv2.circle(im, (int(pointobb[2 * idx]), int(pointobb[2 * idx + 1])), 5, color, -1)
        
    cv2.imshow('demo', im)
    cv2.waitKey(10000)

def show_thetaobb(imgDir, img, anns):
    im = cv2.imread(imgDir + img['file_name'])
    flip = np.random.choice([True, False])
    scale = np.random.uniform(0.5, 1.0, 1)[0]
    im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    print("thetaobb flip: {}, thetaobb scale: {}".format(flip, scale))
    if flip:
        im = cv2.flip(im, 1)
    for ann in anns:
        thetaobb = ann['thetaobb']
        thetaobb = np.array(thetaobb)
        thetaobb = thetaobb_rescale(thetaobb, scale, reverse_flag=False)
        if flip:
            img_shape = im.shape
            thetaobb = thetaobb_flip(thetaobb, img_shape)
        cx, cy, w, h, theta = thetaobb

        rect = ((cx, cy), (w, h), theta/np.pi*180.0)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        cv2.drawContours(im, [rect], -1, (0, 0, 255), 3)

    cv2.imshow('demo', im)
    cv2.waitKey(10000)

def show_hobb(imgDir, img, anns):
    im = cv2.imread(imgDir + img['file_name'])
    flip = np.random.choice([True, False])
    scale = np.random.uniform(0.5, 1.0, 1)[0]
    im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    print("hobb flip: {}, hobb scale: {}".format(flip, scale))
    if flip:
        im = cv2.flip(im, 1)
    for ann in anns:
        hobb = ann['hobb']
        hobb = np.array(hobb)
        hobb = hobb_rescale(hobb, scale, reverse_flag=False)
        if flip:
            img_shape = im.shape
            hobb = hobb_flip(hobb, img_shape)
        first_point_x = hobb[..., 0]
        first_point_y = hobb[..., 1]
        second_point_x = hobb[..., 2]
        second_point_y = hobb[..., 3]
        h = hobb[..., 4]

        angle_first_second = np.pi / 2.0 - np.arctan2(second_point_y - first_point_y, second_point_x - first_point_x)
        delta_x = h * np.cos(angle_first_second)
        delta_y = h * np.sin(angle_first_second)

        forth_point_x = first_point_x - delta_x
        forth_point_y = first_point_y + delta_y

        third_point_x = second_point_x - delta_x
        third_point_y = second_point_y + delta_y

        pointobb = [first_point_x, first_point_y, second_point_x, second_point_y, third_point_x, third_point_y, forth_point_x, forth_point_y]

        for idx in range(4):
            if idx == 0:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            cv2.circle(im, (int(pointobb[2*idx]), int(pointobb[2*idx+1])), 5, color, -1)
        im = draw_rectangle_by_points(im, pointobb)

    cv2.imshow('demo', im)
    cv2.waitKey(10000)


if __name__ == '__main__':
    show_items = {'maskobb': show_maskobb, 
                  'bbox': show_bbox, 
                  'pointobb': show_pointobb, 
                  'thetaobb': show_thetaobb, 
                  'hobb': show_hobb}
    show_flag = 'maskobb'

    pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    release_version = 'v1'
    imageset = 'trainval'
    rate = '1.0'
    pointobb_sort_method = ''

    generate_shuffthingmap = False
    shuffthingmap_save_path = './data/dota/{}/{}/segmentation'.format(release_version, imageset)
    if generate_shuffthingmap:
        if not os.path.isdir(shuffthingmap_save_path):
            os.mkdir(shuffthingmap_save_path)

    imgDir = './data/dota/{}/coco/{}/'.format(release_version, imageset)
    annFile='./data/dota/{}/coco/annotations/dota_{}_{}_{}{}.json'.format(release_version, imageset, release_version, rate, pointobb_sort_method)

    coco=COCO(annFile)

    catIds = coco.getCatIds(catNms=[''])
    imgIds = coco.getImgIds(catIds=catIds)

    for idx, imgId in enumerate(imgIds):
        img = coco.loadImgs(imgIds[idx])[0]

        if img['file_name'] != 'P0002__1.0__1533___0.png':
            continue

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        print("idx: {}, image file name: {}".format(idx, img['file_name']))

        if generate_shuffthingmap:
            labelmap_save_path = os.path.join(shuffthingmap_save_path, img['file_name'])
            cocoSegmentationToPng(coco, imgId, labelmap_save_path)
        else:
            show_items[show_flag](imgDir, img, anns)

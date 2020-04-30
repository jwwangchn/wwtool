from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import math
import os
    

if __name__ == '__main__':
    show_flag = 'maskobb'

    # pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    release_version = 'v1'
    imageset = 'train'

    imgpath = '/data/sn6/{}/coco/train_RGB/'.format(release_version)
    annopath = '/data/sn6/{}/coco/annotations/sn6_{}_{}_PS-RGB.json'.format(release_version, imageset, release_version)
    coco=COCO(annopath)

    catIds = coco.getCatIds(catNms=[''])
    imgIds = coco.getImgIds(catIds=catIds)

    for idx, imgId in enumerate(imgIds):
        img = coco.loadImgs(imgIds[idx])[0]

        # if img['file_name'] != 'P0002__1.0__1533___0.png':
        #     continue

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        #print("idx: {}, image file name: {}".format(idx, img['file_name']))

        # SN6_Train_AOI_11_Rotterdam_PS-RGB_20190804144353_20190804144627_tile_9935
        # SN6_Train_AOI_11_Rotterdam_SAR-Intensity_20190804144353_20190804144627_tile_9935.tif
        if img['file_name'] != 'SN6_Train_AOI_11_Rotterdam_PS-RGB_20190804144353_20190804144627_tile_9935.tif':
            continue
        I = io.imread(imgpath + img['file_name'])
        plt.imshow(I); 
        coco.showAnns(anns)
        plt.show()

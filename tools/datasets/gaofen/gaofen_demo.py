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
    imageset = 'test'

    imgpath = '/data/gaofen/{}/coco/{}/'.format(release_version, imageset)
    annopath = '/data/gaofen/{}/coco/annotations/gaofen_{}_{}_1.0.json'.format(release_version, imageset, release_version)

    coco=COCO(annopath)

    catIds = coco.getCatIds(catNms=[''])
    imgIds = coco.getImgIds(catIds=catIds)

    for idx, imgId in enumerate(imgIds):
        img = coco.loadImgs(imgIds[idx])[0]

        # if img['file_name'] != 'P0002__1.0__1533___0.png':
        #     continue

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        print("idx: {}, image file name: {}".format(idx, img['file_name']))

        I = io.imread(imgpath + img['file_name'])
        plt.imshow(I); 
        coco.showAnns(anns)
        plt.show()

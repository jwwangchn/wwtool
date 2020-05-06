import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab
import cv2

from pycocotools.coco import COCO
import wwtool
import mmcv

matplotlib.use('Agg')

def show_bbox(imgDir, img, anns, save_name):
    im = cv2.imread(imgDir + img['file_name'])
    for ann in anns:
        bbox = ann['bbox']
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255), 3)

    cv2.imwrite(save_name, im)
    # wwtool.show_image(im)

def show_maskobb(imgDir, img, anns, save_name):
    I = cv2.imread(imgDir + img['file_name'])
    plt.imshow(I); 
    coco.showAnns(anns)
    plt.show()
    plt.savefig(save_name)


if __name__ == '__main__':
    show_items = {'maskobb': show_maskobb, 
                  'bbox': show_bbox}
    show_flag = 'maskobb'

    pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    release_version = 'v1'
    imageset = 'train_shanghai'
    core_dataset_name = 'buildchange'

    save_flag = True

    imgDir = './data/{}/{}/coco/{}/'.format(core_dataset_name, release_version, imageset)
    annFile = './data/{}/{}/coco/annotations/{}_{}_{}.json'.format(core_dataset_name, release_version, core_dataset_name, imageset, release_version)
    save_dir = './data/{}/{}/coco/vis_annotation/{}'.format(core_dataset_name, release_version, imageset)
    mmcv.mkdir_or_exist(save_dir)
    coco=COCO(annFile)

    catIds = coco.getCatIds(catNms=[''])
    imgIds = coco.getImgIds(catIds=catIds)

    for idx, imgId in enumerate(imgIds):
        img = coco.loadImgs(imgIds[idx])[0]

        # if img['file_name'] != 'P0002__1.0__1533___0.png':
        #     continue

        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        print("idx: {}, image file name: {}".format(idx, img['file_name']))
        save_name = os.path.join(save_dir, img['file_name'])
        show_items[show_flag](imgDir, img, anns, save_name)

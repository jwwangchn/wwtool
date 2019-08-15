from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import math
import os

import mmcv
from wwtool.transforms import pointobb_flip, thetaobb_flip, hobb_flip
from wwtool.transforms import pointobb_rescale, thetaobb_rescale, hobb_rescale, pointobb2pseudomask
from wwtool.visualization import show_centerness


if __name__ == '__main__':
    release_version = 'v1'
    imageset = 'trainval'
    rate = '1.0'
    pointobb_sort_method = 'best'
    extra_info = 'keypoint'
    show = True
    save = False

    imgDir = './data/dota/{}/coco/{}/'.format(release_version, imageset)
    annFile = './data/dota/{}/coco/annotations/dota_{}_{}_{}_{}_{}.json'.format(release_version, imageset, release_version, rate, pointobb_sort_method, extra_info)
    save_path = './data/dota/{}/{}/pseudomasks'.format(release_version, imageset)

    coco=COCO(annFile)

    catIds = coco.getCatIds(catNms=[''])
    imgIds = coco.getImgIds(catIds=catIds)

    progress_bar = mmcv.ProgressBar(len(imgIds))
    for idx, imgId in enumerate(imgIds):
        # if idx >= 1:
        #     break
        img_info = coco.loadImgs(imgIds[idx])[0]
        image_name = img_info['file_name']
        image_file = os.path.join(imgDir, image_name)
        # if img['file_name'] != 'P2246__1.0__0___84.png':
        #     continue
        img = cv2.imread(image_file)
        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        pseudomasks = []
        height = img_info['height']
        width = img_info['width']
        pseudomasks = np.zeros((height, width), dtype=np.float64)

        for ann in anns:
            pointobb = ann['pointobb']
            pseudomask = pointobb2pseudomask(height, width, pointobb)
            pseudomasks += pseudomask

        progress_bar.update()
        if save:
            pseudomask_file = os.path.join(save_path, image_name.split('.png')[0])
            np.save(pseudomask_file, pseudomasks)

        if show:
            pseudomasks_ = show_centerness(pseudomasks, False, return_img=True)

            alpha = 0.6
            beta = (1.0 - alpha)
            dst = cv2.addWeighted(pseudomasks_, alpha, img, beta, 0.0)
            print(os.path.join(save_path, image_name))
            cv2.imwrite(os.path.join(save_path, image_name), dst)
            # cv2.imshow("demo", dst)
            # cv2.waitKey(0)
        # print("idx: {}, image file name: {}".format(idx, img_info['file_name']))

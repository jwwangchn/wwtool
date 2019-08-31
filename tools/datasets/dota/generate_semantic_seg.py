from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import math
import os
import sys
import concurrent.futures

import mmcv
from wwtool.transforms import pointobb_flip, thetaobb_flip, hobb_flip
from wwtool.transforms import pointobb_rescale, thetaobb_rescale, hobb_rescale, pointobb2pseudomask
from wwtool.visualization import show_grayscale_as_heatmap
from wwtool.datasets import cocoSegmentationToPng

class Core():
    def __init__(self,
                release_version,
                imageset,
                rate,
                pointobb_sort_method,
                extra_info,
                multi_processing=False):
        self.release_version = release_version
        self.imageset = imageset
        self.rate = rate
        self.pointobb_sort_method = pointobb_sort_method
        self.extra_info = extra_info

        self.imgDir = './data/dota/{}/coco/{}/'.format(self.release_version, self.imageset)
        self.annFile = './data/dota/{}/coco/annotations/dota_{}_{}_{}_{}_{}.json'.format(self.release_version, self.imageset, self.release_version, self.rate, self.pointobb_sort_method, self.extra_info)
        self.save_path = './data/dota/{}/{}/segmentation'.format(self.release_version, self.imageset)

        self.coco = COCO(self.annFile)
        self.catIds = self.coco.getCatIds(catNms=[''])
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        self.progress_bar = mmcv.ProgressBar(len(self.imgIds))
        self.multi_processing = multi_processing

    def _core_(self, imgId):
        img_info = self.coco.loadImgs(imgId)[0]
        image_name = img_info['file_name']

        pseudomask_file = os.path.join(self.save_path, image_name)
        cocoSegmentationToPng(self.coco, imgId, pseudomask_file, vis=False, return_flag=False)

    def generate_segmentation(self):
        if self.multi_processing:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for _ in executor.map(self._core_, self.imgIds):
                    self.progress_bar.update()
        else:
            for _, imgId in enumerate(self.imgIds):
                self._core_(imgId)
                self.progress_bar.update()

if __name__ == '__main__':
    release_version = 'v1'
    imageset = 'trainval'
    rate = '1.0'
    pointobb_sort_method = 'best'
    extra_info = 'keypoint'

    core = Core(release_version=release_version, 
                imageset=imageset,
                rate=rate,
                pointobb_sort_method=pointobb_sort_method,
                extra_info=extra_info,
                multi_processing=False)

    core.generate_segmentation()

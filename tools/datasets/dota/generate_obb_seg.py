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
                multi_processing=False,
                binary_mask=False,
                vis=False):
        self.release_version = release_version
        self.imageset = imageset
        self.binary_mask = binary_mask
        self.vis = vis

        self.imgDir = './data/{}/{}/coco/{}/'.format(core_dataset, self.release_version, self.imageset)
        self.annFile = './data/{}/{}/coco/annotations/{}.json'.format(core_dataset, self.release_version, "_".join(ann_file_name))
        if binary_mask == True:
            self.save_path = './data/{}/{}/{}/obb_seg_binary'.format(core_dataset, self.release_version, self.imageset)
            self.stuffEndId = 1
        else:
            self.save_path = './data/{}/{}/{}/obb_seg'.format(core_dataset, self.release_version, self.imageset)
            self.stuffEndId = 15
        mmcv.mkdir_or_exist(self.save_path)

        self.coco = COCO(self.annFile)
        self.catIds = self.coco.getCatIds(catNms=[''])
        self.imgIds = self.coco.getImgIds(catIds=self.catIds).sort()
        self.progress_bar = mmcv.ProgressBar(len(self.imgIds))
        self.multi_processing = multi_processing

    def _core_(self, imgId):
        img_info = self.coco.loadImgs(imgId)[0]
        image_name = img_info['file_name']

        if os.path.exists(os.path.join(self.save_path, image_name)):
            return
        # img_list = ['P0019__1.0__824___824.png', 'P0858__1.0__0___441.png', 'P1399__1.0__3296___3296.png', 'P1466__1.0__2472___2472.png', 'P0867__1.0__1794___1027.png']
        # img_list = ['P2802__1.0__4914___4225.png']
        # if image_name not in img_list:
        #     return
        pseudomask_file = os.path.join(self.save_path, image_name)
        cocoSegmentationToPng(self.coco, imgId, pseudomask_file, vis=self.vis, return_flag=False, stuffStartId=0, stuffEndId=self.stuffEndId, binary_mask=self.binary_mask)

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
    core_dataset = 'dota'
    release_version = 'DJ'
    imageset = 'trainval'

    ann_file_name = [core_dataset, imageset, release_version, 'best']

    binary_mask = False
    vis = False

    core = Core(release_version=release_version, 
                imageset=imageset,
                multi_processing=False,
                binary_mask=binary_mask,
                vis=vis)

    core.generate_segmentation()

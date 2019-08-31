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
from wwtool.image import generate_gaussian_image, generate_centerness_image, generate_ellipse_image

class PseudomaskGenerate():
    def __init__(self,
                release_version,
                imageset,
                rate,
                pointobb_sort_method,
                extra_info,
                save_vis=False,
                show_pseudomask=False,
                encode='centernessmask',
                multi_processing=False):
        self.release_version = release_version
        self.imageset = imageset
        self.rate = rate
        self.pointobb_sort_method = pointobb_sort_method
        self.extra_info = extra_info
        self.encode = encode

        self.imgDir = './data/dota/{}/coco/{}/'.format(self.release_version, self.imageset)
        self.annFile = './data/dota/{}/coco/annotations/dota_{}_{}_{}_{}_{}.json'.format(self.release_version, self.imageset, self.release_version, self.rate, self.pointobb_sort_method, self.extra_info)
        
        self.save_vis = save_vis
        self.show_pseudomask = show_pseudomask

        self.save_path = './data/dota/{}/{}/{}'.format(self.release_version, self.imageset, self.encode)
        self.save_vis_path = './data/dota/{}/{}/pseudomask_vis'.format(self.release_version, self.imageset)

        mmcv.mkdir_or_exist(self.save_path)
        mmcv.mkdir_or_exist(self.save_vis_path)

        self.gaussian_image = generate_gaussian_image(512, 512, 2.5)
        self.centerness_image = generate_centerness_image(512, 512)
        self.ellipse_image = generate_ellipse_image(512, 512)

        self.anchor_image = {'centerness': self.centerness_image,
                            'gaussian': self.gaussian_image,
                            'ellipse': self.ellipse_image}

        self.coco = COCO(self.annFile)
        self.catIds = self.coco.getCatIds(catNms=[''])
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        self.progress_bar = mmcv.ProgressBar(len(self.imgIds))
        self.multi_processing = multi_processing

    def __generate_pseudomask(self, imgId):
        img_info = self.coco.loadImgs(imgId)[0]
        image_name = img_info['file_name']
        annIds = self.coco.getAnnIds(imgIds=img_info['id'], catIds=self.catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        pseudomasks = []
        height = img_info['height']
        width = img_info['width']
        pseudomasks = np.zeros((height, width), dtype=np.int32)

        anchor_image = self.anchor_image[self.encode]

        for ann in anns:
            pointobb = ann['pointobb']
            transformed, gaussianmask_location = pointobb2pseudomask(pointobb, anchor_image, host_height = height, host_width = width)
            transformed = transformed.astype(np.int32)
            pseudomasks[gaussianmask_location[1]:gaussianmask_location[3], gaussianmask_location[0]:gaussianmask_location[2]] += transformed
            
        # save pseudomask
        pseudomask_file = os.path.join(self.save_path, image_name)
        pseudomasks = np.clip(pseudomasks, 0, 255)
        # pseudomasks = pseudomasks * 255.0
        pseudomasks = pseudomasks.astype(np.uint8)
        cv2.imwrite(pseudomask_file, pseudomasks)

        if self.save_vis:
            image_file = os.path.join(self.imgDir, image_name)
            img = cv2.imread(image_file)
            pseudomask_vis_file = os.path.join(self.save_vis_path, image_name)
            pseudomasks_ = show_grayscale_as_heatmap(pseudomasks / 255.0, self.show_pseudomask, return_img=True)
            alpha = 0.6
            beta = (1.0 - alpha)
            pseudomasks = cv2.addWeighted(pseudomasks_, alpha, img, beta, 0.0)
            cv2.imwrite(pseudomask_vis_file, pseudomasks)

    def generate_pseudomask_core(self):
        if self.multi_processing:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for _ in executor.map(self.__generate_pseudomask, self.imgIds):
                    self.progress_bar.update()
        else:
            for _, imgId in enumerate(self.imgIds):
                self.__generate_pseudomask(imgId)
                self.progress_bar.update()

if __name__ == '__main__':
    release_version = 'v1'
    imageset = 'trainval'
    rate = '1.0'
    pointobb_sort_method = 'best'
    extra_info = 'keypoint'

    encode = 'centerness'   # centerness, gaussian, ellipse

    save_vis = False
    show_pseudomask = False

    pseudomask_gen = PseudomaskGenerate(release_version=release_version, 
                imageset=imageset,
                rate=rate,
                pointobb_sort_method=pointobb_sort_method,
                extra_info=extra_info,
                save_vis=save_vis,
                show_pseudomask=show_pseudomask,
                encode=encode,
                multi_processing=False)

    pseudomask_gen.generate_pseudomask_core()

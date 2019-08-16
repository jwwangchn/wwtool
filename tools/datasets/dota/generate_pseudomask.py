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
from wwtool.visualization import show_centerness

class Core():
    def __init__(self,
                release_version,
                imageset,
                rate,
                pointobb_sort_method,
                extra_info,
                show,
                save_np,
                multi_processing=False):
        self.release_version = release_version
        self.imageset = imageset
        self.rate = rate
        self.pointobb_sort_method = pointobb_sort_method
        self.extra_info = extra_info
        self.show = show
        self.save_np = save_np

        self.imgDir = './data/dota/{}/coco/{}/'.format(self.release_version, self.imageset)
        self.annFile = './data/dota/{}/coco/annotations/dota_{}_{}_{}_{}_{}.json'.format(self.release_version, self.imageset, self.release_version, self.rate, self.pointobb_sort_method, self.extra_info)
        self.save_path = './data/dota/{}/{}/pseudomasks'.format(self.release_version, self.imageset)

        self.coco = COCO(self.annFile)
        self.catIds = self.coco.getCatIds(catNms=[''])
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        self.progress_bar = mmcv.ProgressBar(len(self.imgIds))
        self.multi_processing = multi_processing

    def _core_(self, imgId):
        img_info = self.coco.loadImgs(imgId)[0]
        image_name = img_info['file_name']
        print(image_name)
        annIds = self.coco.getAnnIds(imgIds=img_info['id'], catIds=self.catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        pseudomasks = []
        height = img_info['height']
        width = img_info['width']
        pseudomasks = np.zeros((height, width), dtype=np.float64)

        for ann in anns:
            pointobb = ann['pointobb']
            pseudomask = pointobb2pseudomask(height, width, pointobb)
            pseudomasks += pseudomask

        # self.progress_bar.update()
        if save_np:
            pseudomask_file = os.path.join(self.save_path, image_name.split('.png')[0])
            np.save(pseudomask_file, pseudomasks)
        else:
            image_file = os.path.join(self.imgDir, image_name)
            img = cv2.imread(image_file)
            pseudomask_file = os.path.join(self.save_path, image_name)
            pseudomasks_ = show_centerness(pseudomasks, False, return_img=True)

            alpha = 0.6
            beta = (1.0 - alpha)
            pseudomasks = cv2.addWeighted(pseudomasks_, alpha, img, beta, 0.0)
            cv2.imwrite(pseudomask_file, pseudomasks)
        return image_name

    def generate_pseudomask(self):
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
    show = False
    save_np = True

    core = Core(release_version=release_version, 
                imageset=imageset,
                rate=rate,
                pointobb_sort_method=pointobb_sort_method,
                extra_info=extra_info,
                show=show,
                save_np=save_np,
                multi_processing=False)

    core.generate_pseudomask()

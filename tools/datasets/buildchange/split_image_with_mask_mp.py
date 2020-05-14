import os
import numpy as np
import cv2
from PIL import Image

import geojson
import shapely.wkt
import rasterio as rio

import wwtool
import mmcv

from multiprocessing import Pool
from functools import partial

Image.MAX_IMAGE_PIXELS = int(2048 * 2048 * 2048 // 4 // 3)



class SplitWithMask():
    def __init__(self, 
                 core_dataset_name='buildchange',
                 src_version='v0',
                 dst_version='v2',
                 imageset='train_shanghai',
                 subimage_size=1024,
                 gap=512,
                 num_processor=16):
        self.core_dataset_name = core_dataset_name
        self.src_version = src_version
        self.dst_version = dst_version
        self.imageset = imageset
        self.subimage_size = subimage_size
        self.gap = gap
        self.image_path = './data/{}/{}/{}/images'.format(core_dataset_name, src_version, imageset)
        self.mask_path = './data/{}/{}/{}/anno_v2'.format(core_dataset_name, src_version, imageset)

        self.image_save_path = './data/{}/{}/{}/images'.format(core_dataset_name, dst_version, "{}_{}".format(imageset, subimage_size))
        wwtool.mkdir_or_exist(self.image_save_path)
        self.label_save_path = './data/{}/{}/{}/labels'.format(core_dataset_name, dst_version, "{}_{}".format(imageset, subimage_size))
        wwtool.mkdir_or_exist(self.label_save_path)

        self.mask_parser = wwtool.MaskParse()
        self.pool = Pool(num_processor)

    def split_image_with_mask(self, image_fn):
        if not image_fn.endswith('.jpg'):
            return
        image_file = os.path.join(self.image_path, image_fn)
        mask_file = os.path.join(self.mask_path, image_fn.replace('jpg', 'png'))
        file_name = os.path.splitext(os.path.basename(image_file))[0]

        if not os.path.exists(mask_file):
            return

        img = cv2.imread(image_file)

        objects = self.mask_parser(mask_file)

        masks = np.array([obj['segmentation'] for obj in objects])

        subimages = wwtool.split_image(img, subsize=self.subimage_size, gap=self.gap)
        subimage_coordinates = list(subimages.keys())

        if masks.shape[0] == 0:
            return

        mask_centroids = []
        for obj in objects:
            geometry = obj['polygon'].centroid
            geo = geojson.Feature(geometry=geometry, properties={})
            coordinate = geo.geometry["coordinates"]
            coordinate[0], coordinate[1] = abs(coordinate[0]), abs(coordinate[1])
            mask_centroids.append(coordinate)

        mask_centroids = np.array(mask_centroids)
        mask_centroids_ = mask_centroids.copy()
        
        for subimage_coordinate in subimage_coordinates:
            objects = []

            mask_centroids_[:, 0] = mask_centroids[:, 0] - subimage_coordinate[0]
            mask_centroids_[:, 1] = mask_centroids[:, 1] - subimage_coordinate[1]


            cx_bool = np.logical_and(mask_centroids_[:, 0] >= 0, mask_centroids_[:, 0] < subimage_size)
            cy_bool = np.logical_and(mask_centroids_[:, 1] >= 0, mask_centroids_[:, 1] < subimage_size)

            subimage_masks = masks[np.logical_and(cx_bool, cy_bool)]

            subimage_masks_ = []
            for subimage_mask in subimage_masks:
                subimage_mask_np = np.array(subimage_mask)
                subimage_mask_np[0::2] = subimage_mask_np[0::2] - subimage_coordinate[0]
                subimage_mask_np[1::2] = subimage_mask_np[1::2] - subimage_coordinate[1]

                subimage_masks_.append(subimage_mask_np.tolist())
            
            subimage_masks = subimage_masks_
            if len(subimage_masks) == 0:
                continue
            img = subimages[subimage_coordinate]
            if np.mean(img) == 0:
                continue

            label_save_file = os.path.join(self.label_save_path, '{}__{}_{}.txt'.format(file_name, subimage_coordinate[0], subimage_coordinate[1]))
            image_save_file = os.path.join(self.image_save_path, '{}__{}_{}.png'.format(file_name, subimage_coordinate[0], subimage_coordinate[1]))
            cv2.imwrite(image_save_file, img)
            
            for subimage_mask in subimage_masks:
                subimage_objects = dict()
                subimage_objects['mask'] = subimage_mask
                subimage_objects['label'] = 'building'
                objects.append(subimage_objects)
            wwtool.simpletxt_dump(objects, label_save_file, encode='mask')

    def core(self):
        image_fn_list = os.listdir(self.image_path)
        print(image_fn_list)
        worker = partial(self.split_image_with_mask)
        self.pool.map(worker, image_fn_list)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


if __name__ == '__main__':
    core_dataset_name = 'buildchange'
    src_version = 'v0'
    dst_version = 'v2'
    imagesets = ['train_beijing', 'train_chengdu', 'train_haerbin', 'train_jinan', 'train_shanghai', 'val_xian']
    # imagesets = ['train_shanghai', 'val_xian']
    subimage_size = 1024
    gap = subimage_size // 2

    for imageset in imagesets:
        print("Begin processing {} set.".format(imageset))
        split_with_mask = SplitWithMask(core_dataset_name=core_dataset_name,
                                        src_version=src_version,
                                        dst_version=dst_version,
                                        imageset=imageset,
                                        subimage_size=subimage_size,
                                        gap=gap,
                                        num_processor=16)

        split_with_mask.core()
        print("Finish processing {} set.".format(imageset))

        

import os
import numpy as np
import cv2
from PIL import Image
from skimage.io import imread

import wwtool

Image.MAX_IMAGE_PIXELS = int(2048 * 2048 * 2048 // 4 // 3)

if __name__ == '__main__':
    core_dataset_name = 'uavdt'
    label_fold = './data/{}/v0/trainval_test/labels'.format(core_dataset_name)
    uavdt_parse = wwtool.UAVDT_PARSE(label_fold)

    image_sets = ['trainval_test']
    image_format = '.jpg'

    subimage_size = 800
    gap = 200

    for image_set in image_sets:
        seq_path = './data/{}/v0/{}/images'.format(core_dataset_name, image_set)

        image_save_path = './data/{}/v1/{}/images'.format(core_dataset_name, image_set)
        wwtool.mkdir_or_exist(image_save_path)
        label_save_path = './data/{}/v1/{}/labels'.format(core_dataset_name, image_set)
        wwtool.mkdir_or_exist(label_save_path)

        for seq_idx, seq_name in enumerate(os.listdir(seq_path)):
            image_path = os.path.join(seq_path, seq_name)
            image_names = list(sorted(os.listdir(image_path)))
            for image_idx, image_name in enumerate(image_names):
                if image_idx % 10 != 0:
                    continue
                print(seq_idx, seq_name, image_idx, image_name)
                file_name = image_name.split(image_format)[0]
                image_file = os.path.join(image_path, file_name + image_format)
                
                img = imread(image_file)

                objects = uavdt_parse.uavdt_parse(seq_name, image_name.split('.')[0])
                if objects == []:
                    continue
                
                bboxes = np.array([wwtool.xyxy2cxcywh(obj['bbox']) for obj in objects])
                labels = np.array([obj['label'] for obj in objects])

                subimages = wwtool.split_image(img, subsize=subimage_size, gap=gap, expand_boundary=True)
                subimage_coordinates = list(subimages.keys())
                bboxes_ = bboxes.copy()
                labels_ = labels.copy()
                if bboxes_.shape[0] == 0:
                    continue

                for subimage_coordinate in subimage_coordinates:
                    objects = []
                    
                    bboxes_[:, 0] = bboxes[:, 0] - subimage_coordinate[0]
                    bboxes_[:, 1] = bboxes[:, 1] - subimage_coordinate[1]
                    cx_bool = np.logical_and(bboxes_[:, 0] >= 0, bboxes_[:, 0] < subimage_size)
                    cy_bool = np.logical_and(bboxes_[:, 1] >= 0, bboxes_[:, 1] < subimage_size)
                    subimage_bboxes = bboxes_[np.logical_and(cx_bool, cy_bool)]
                    subimage_labels = labels_[np.logical_and(cx_bool, cy_bool)]
                    
                    if len(subimage_bboxes) == 0:
                        continue
                    img = subimages[subimage_coordinate]
                    if np.mean(img) == 0:
                        continue

                    label_save_file = os.path.join(label_save_path, '{}_{}__{}_{}.txt'.format(seq_name, file_name, subimage_coordinate[0], subimage_coordinate[1]))
                    image_save_file = os.path.join(image_save_path, '{}_{}__{}_{}.png'.format(seq_name, file_name, subimage_coordinate[0], subimage_coordinate[1]))
                    cv2.imwrite(image_save_file, img)
                    
                    for subimage_bbox, subimage_label in zip(subimage_bboxes, subimage_labels):
                        subimage_objects = dict()
                        subimage_objects['bbox'] = wwtool.cxcywh2xyxy(subimage_bbox.tolist())
                        subimage_objects['label'] = subimage_label
                        objects.append(subimage_objects)
                    wwtool.simpletxt_dump(objects, label_save_file)

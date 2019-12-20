import os
import numpy as np
import cv2
from PIL import Image
from skimage.io import imread

import wwtool

Image.MAX_IMAGE_PIXELS = int(2048 * 2048 * 2048 // 4 // 3)

if __name__ == '__main__':
    image_sets = ['train', 'val']
    subimage_size = 800
    gap = 200

    for image_set in image_sets:
        image_path = './data/visdrone/v0/{}/images'.format(image_set)
        label_path = './data/visdrone/v0/{}/annotations'.format(image_set)

        image_save_path = '/data/visdrone/v1/{}/images'.format(image_set)
        wwtool.mkdir_or_exist(image_save_path)
        label_save_path = '/data/visdrone/v1/{}/labels'.format(image_set)
        wwtool.mkdir_or_exist(label_save_path)

        # print(os.listdir(label_path))
        for idx, label_file in enumerate(os.listdir(label_path)):
            print(idx, label_file)
            file_name = label_file.split('.txt')[0]
            label_file = os.path.join(label_path, file_name + '.txt')
            image_file = os.path.join(image_path, file_name + '.jpg')
            
            img = imread(image_file)

            objects = wwtool.visdrone_parse(label_file)
            bboxes = np.array([obj['bbox'] for obj in objects])
            labels = np.array([obj['label'] for obj in objects])

            subimages = wwtool.split_image(img, subsize=subimage_size, gap=gap)
            subimage_coordinates = list(subimages.keys())
            bboxes_ = bboxes.copy()
            labels_ = labels.copy()
            
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

                label_save_file = os.path.join(label_save_path, '{}__{}_{}.txt'.format(file_name, subimage_coordinate[0], subimage_coordinate[1]))
                image_save_file = os.path.join(image_save_path, '{}__{}_{}.png'.format(file_name, subimage_coordinate[0], subimage_coordinate[1]))
                cv2.imwrite(image_save_file, img)
                
                for subimage_bbox, subimage_label in zip(subimage_bboxes, subimage_labels):
                    subimage_objects = dict()
                    subimage_objects['bbox'] = subimage_bbox.tolist()
                    subimage_objects['label'] = subimage_label
                    objects.append(subimage_objects)
                wwtool.simpletxt_dump(objects, label_save_file)

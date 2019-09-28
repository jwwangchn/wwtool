import os
import numpy as np
import cv2
from PIL import Image
from skimage.io import imread

import wwtool

Image.MAX_IMAGE_PIXELS = int(2048 * 2048 * 2048 // 4 // 3)

if __name__ == '__main__':
    image_sets = ['trainval']
    for image_set in image_sets:
        image_path = '/data/gaofen/v0/{}/images'.format(image_set)
        label_path = '/data/gaofen/v0/{}/labels'.format(image_set)

        image_save_path = '/data/gaofen/v1/{}/images'.format(image_set)
        label_save_path = '/data/gaofen/v1/{}/labels'.format(image_set)

        for label_file in os.listdir(label_path):
            file_name = label_file.split('.xml')[0]
            label_file = os.path.join(label_path, file_name + '.xml')
            image_file = os.path.join(image_path, file_name + '.png')
            
            img = imread(image_file)

            objects = wwtool.rovoc_parse(label_file)
            label_save_file = os.path.join(label_save_path, file_name + '.txt')
            wwtool.split_image(img, subsize=1024, gap=200)
            
            wwtool.simpletxt_dump(objects, label_save_file)
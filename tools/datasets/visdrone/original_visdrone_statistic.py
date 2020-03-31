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
        image_path = './data/visdrone/v0/{}/images'.format(image_set)
        label_path = './data/visdrone/v0/{}/annotations'.format(image_set)

        image_save_path = '/data/visdrone/v1/{}/images'.format(image_set)
        wwtool.mkdir_or_exist(image_save_path)
        label_save_path = '/data/visdrone/v1/{}/labels'.format(image_set)
        wwtool.mkdir_or_exist(label_save_path)

        object_sizes = []
        image_sizes = []
        for idx, label_file in enumerate(os.listdir(label_path)):
            print(idx, label_file)
            file_name = label_file.split('.txt')[0]
            label_file = os.path.join(label_path, file_name + '.txt')
            image_file = os.path.join(image_path, file_name + '.jpg')
            
            img = imread(image_file)
            w, h, _ = img.shape

            objects = wwtool.visdrone_parse(label_file)
            for obj in objects:
                box = wwtool.xyxy2cxcywh(obj['bbox'])
                object_sizes.append(box[2] * box[3])
                image_sizes.append(w * h)

        object_sizes = np.array(object_sizes)
        image_sizes = np.array(image_sizes)

        object_sizes = np.sqrt(object_sizes)
        image_sizes = np.sqrt(image_sizes)
        
        print("object number: ", object_sizes.shape)
        print("max object size: {}".format(np.max(object_sizes)))
        print("min object size: {}".format(np.min(object_sizes)))
        print("absolute: {}, {}, relative: {}, {}".format(np.mean(object_sizes), np.std(object_sizes, ddof=1), np.mean(object_sizes/image_sizes), np.std(object_sizes/image_sizes), ddof=1))
            
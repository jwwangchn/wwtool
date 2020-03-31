import os
import numpy as np
import cv2
from PIL import Image
from skimage.io import imread

import wwtool

Image.MAX_IMAGE_PIXELS = int(2048 * 2048 * 2048 // 4 // 3)

if __name__ == '__main__':
    json_file = '/data/airbus-ship-detection/v0/train/labels/train_ship_segmentations_v2.csv'
    airbus_ship_parse = wwtool.AirbusShipParse(json_file)

    image_sets = ['train']
    image_format = '.jpg'

    for image_set in image_sets:
        image_path = '/data/airbus-ship-detection/v0/train/images'

        object_sizes = []
        image_sizes = []
        for idx, image_name in enumerate(os.listdir(image_path)):
            print(idx, image_name)
            file_name = image_name.split(image_format)[0]
            image_file = os.path.join(image_path, file_name + image_format)
            
            img = imread(image_file)

            w, h, _ = img.shape

            objects = airbus_ship_parse.airbus_ship_parse(image_name)
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
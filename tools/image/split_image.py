import os
import cv2
import numpy as np
import mmcv
from wwtool import split_image, show_image
from skimage.io import imread

if __name__ == '__main__':
    # img = np.random.rand(5292, 3371)
    img_path = '/home/jwwangchn/data/GF/v0/tif'
    save_path = '/home/jwwangchn/data/GF/v1/vis'

    for image_name in os.listdir(img_path):
        mmcv.mkdir_or_exist(os.path.join(save_path, image_name.split('.png')[0]))
        # img = imread(os.path.join(img_path, image_name))
        img = cv2.imread(os.path.join(img_path, image_name))
        print("{} has the shape {}".format(image_name, img.shape))
        subimages = split_image(img, 1024, 0, mode='keep_all')
        subimage_coordinates = list(subimages.keys())
        print("start point: ", subimage_coordinates)

        for subimage_coordinate in subimage_coordinates:
            print("coordinate: {}".format(subimage_coordinate))
            img = subimages[subimage_coordinate]
            if np.mean(img) == 0:
                continue
            save_name = os.path.join(save_path, image_name.split('.png')[0], "{}__{}_{}.png".format(image_name.split('.png')[0], subimage_coordinate[0], subimage_coordinate[1]))
            cv2.imwrite(save_name, img)
            # show_image(img, wait_time=100, save_name=save_name)
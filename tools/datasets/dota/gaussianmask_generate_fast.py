import cv2
import numpy as np

import mmcv
import wwtool
from wwtool.image import generate_gaussian_image, generate_image
from wwtool.visualization import show_grayscale_as_heatmap, show_image
from wwtool.transforms import pointobb_image_transform, thetaobb2pointobb, pointobb2bbox, pointobb2gaussmask

def simple_test():
    image_size = (1024, 1024)
    img = generate_image(height=image_size[0], width=image_size[1], color=0)
    gaussian_image = generate_gaussian_image(512, 512, 2.5)
    show_grayscale_as_heatmap(gaussian_image, win_name='origin')

    thetaobb = [100, 100, 90, 130, 60]
    pointobb = thetaobb2pointobb(thetaobb)


    bbox = pointobb2bbox(pointobb)
    pointobb = np.array(pointobb).reshape(-1, 2).astype(np.float32)
    pointobb[:, 0] = pointobb[:, 0] - bbox[0]
    pointobb[:, 1] = pointobb[:, 1] - bbox[1]

    transformed = pointobb_image_transform(gaussian_image, pointobb)
    show_grayscale_as_heatmap(transformed, win_name='after convert')

    bbox = [int(_) for _ in bbox]

    img_start_x = max(bbox[0], 0)
    img_start_y = max(bbox[1], 0)
    img_end_x = bbox[0] + transformed.shape[1]
    img_end_y = bbox[1] + transformed.shape[0]

    transformed_start_x = max(bbox[0], 0) - bbox[0]
    transformed_start_y = max(bbox[1], 0) - bbox[1]
    transformed_end_x = min(bbox[0] + transformed.shape[1], transformed.shape[1])
    transformed_end_y = min(bbox[1] + transformed.shape[0], transformed.shape[0])

    img[img_start_y:img_end_y, img_start_x:img_end_x] += transformed[transformed_start_y:transformed_end_y, transformed_start_x:transformed_end_x]
    
    show_grayscale_as_heatmap(img)

if __name__ == '__main__':
    image_size = (1024, 1024)
    img = generate_image(height=image_size[0], width=image_size[1], color=0)
    gaussian_image = generate_gaussian_image(512, 512, 2.5)
    show_grayscale_as_heatmap(gaussian_image, win_name='origin')

    thetaobb = [100, 100, 90, 130, 60]
    pointobb = thetaobb2pointobb(thetaobb)

    transformed, gaussianmask_location = pointobb2gaussmask(pointobb, gaussian_image)
    img[gaussianmask_location[1]:gaussianmask_location[3], gaussianmask_location[0]:gaussianmask_location[2]] += transformed

    show_grayscale_as_heatmap(img)
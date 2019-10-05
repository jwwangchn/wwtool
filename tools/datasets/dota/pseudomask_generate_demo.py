import cv2
import numpy as np

import mmcv
import wwtool
from wwtool.generation import generate_centerness_image, generate_image, generate_gaussian_image, generate_ellipse_image
from wwtool.visualization import show_grayscale_as_heatmap, show_image, show_image_surface_curve
from wwtool.transforms import pointobb_image_transform, thetaobb2pointobb, pointobb2bbox, pointobb2pseudomask

if __name__ == '__main__':
    image_size = (1024, 1024)
    img = generate_image(height=image_size[0], width=image_size[1], color=0)
    encoding = 'centerness'       # centerness, gaussian, ellipse
    if encoding == 'gaussian':
        anchor_image = generate_gaussian_image(image_size[0], image_size[1], scale=2.5, threshold=255 * 0.5)
    elif encoding == 'centerness':
        anchor_image = generate_centerness_image(image_size[0], image_size[1], factor=4, threshold=255 * 0.5)
    elif encoding == 'ellipse':
        anchor_image = generate_ellipse_image(image_size[0], image_size[1])
    anchor_image_heatmap = wwtool.show_grayscale_as_heatmap(anchor_image, win_name='before', return_img=True)
    cv2.imwrite('./heatmap.png', anchor_image_heatmap)

    show_image_surface_curve(anchor_image, direction=2)

    thetaobbs = [[0, 0, 120, 200, 60 * np.pi/180.0],
                [300, 200, 50, 70, 30 * np.pi/180.0],
                [450, 500, 300, 230, 45 * np.pi/180.0]]
    thetaobbs = [[300, 200, 50, 70, 30 * np.pi/180.0]]
    pointobbs = []

    for thetaobb in thetaobbs:
        pointobb = thetaobb2pointobb(thetaobb)
        pointobbs.append(pointobb)

    for pointobb in pointobbs:
        transformed, mask_location = pointobb2pseudomask(pointobb, anchor_image, host_height = image_size[0], host_width = image_size[1])
        # show_grayscale_as_heatmap(transformed)
        img[mask_location[1]:mask_location[3], mask_location[0]:mask_location[2]] += transformed

    show_grayscale_as_heatmap(img)
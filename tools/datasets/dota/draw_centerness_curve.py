import cv2
import numpy as np
from matplotlib import pyplot as plt

import mmcv
import wwtool

from wwtool.image import generate_centerness_image, generate_image, generate_gaussian_image, generate_ellipse_image
from wwtool.visualization import show_grayscale_as_heatmap, show_image, show_image_surface_curve
from wwtool.transforms import pointobb_image_transform, thetaobb2pointobb, pointobb2bbox, pointobb2pseudomask


if __name__ == '__main__':
    data_length = 512
    x = np.arange(0, data_length, 0.1)

    left = x
    left = np.maximum(left, 0)
    right = data_length - x
    right = np.maximum(right, 0)

    factors = [1, 2, 4, 8, 16, 32, 64, 128]

    for factor in factors:
        threshold = 1.0 * 0.5
        y = ((np.minimum(left, right) / (np.maximum(left, right) + 1))) ** (1/factor)
        y = np.clip((y * (1.0 - threshold) + threshold), 0, 1.0)
        plt.plot(x, y, label='factor={}'.format(factor))
        plt.legend(loc='best')

    plt.show()
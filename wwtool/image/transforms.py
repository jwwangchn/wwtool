import numpy as np
import cv2


def convert_16bit_to_8bit(img):
    max_value = np.max(img)
    min_value = np.min(img)
    img = (img - min_value) / (max_value - min_value) * 255
    img = img.astype(np.uint8)

    return img
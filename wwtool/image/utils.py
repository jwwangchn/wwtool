import numpy as np
import cv2

def generate_image(height=512, 
                   width=512, 
                   color=(255, 255, 255)):
    b = np.full((height, width, 1), color[0], dtype=np.uint8)
    g = np.full((height, width, 1), color[1], dtype=np.uint8)
    r = np.full((height, width, 1), color[2], dtype=np.uint8)
    img = np.concatenate((b, g, r), axis=2)

    return img

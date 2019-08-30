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

def generate_gaussian_image(height, width, scale=2.5):
    gaussian = lambda x: np.exp(-(1/2) * (x**2)) / (np.sqrt(2 * np.pi))
    scaled_gaussian = lambda x: np.exp(-(1/2) * (x**2))

    x_range = np.arange(0, width)
    y_range = np.arange(0, height)
    index_x, index_y = np.meshgrid(x_range, y_range)

    distance_from_center = scale * np.sqrt(((index_x - width / 2) ** 2) / ((width / 2) ** 2) + ((index_y - height / 2) ** 2) / ((height / 2) ** 2))
    scaled_gaussian_prob = scaled_gaussian(distance_from_center)
    grayscale_image = np.clip(scaled_gaussian_prob * 255, 0, 255)
    grayscale_image = grayscale_image.astype(np.uint8)

    return grayscale_image
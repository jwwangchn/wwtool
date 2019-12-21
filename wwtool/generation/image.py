import numpy as np
import cv2

def generate_image(height=512, 
                   width=512, 
                   color=(255, 255, 255)):
    if type(color) == tuple:
        b = np.full((height, width, 1), color[0], dtype=np.uint8)
        g = np.full((height, width, 1), color[1], dtype=np.uint8)
        r = np.full((height, width, 1), color[2], dtype=np.uint8)
        img = np.concatenate((b, g, r), axis=2)
    else:
        gray = np.full((height, width), color, dtype=np.uint8)
        img = gray

    return img

def generate_gaussian_image(height, width, scale=2.5, threshold=128):
    gaussian = lambda x: np.exp(-(1/2) * (x**2)) / (np.sqrt(2 * np.pi))
    scaled_gaussian = lambda x: np.exp(-(1/2) * (x**2))

    x_range = np.arange(0, width)
    y_range = np.arange(0, height)
    index_x, index_y = np.meshgrid(x_range, y_range)

    distance_from_center = scale * np.sqrt(((index_x - width / 2) ** 2) / ((width / 2) ** 2) + ((index_y - height / 2) ** 2) / ((height / 2) ** 2))
    scaled_gaussian_prob = scaled_gaussian(distance_from_center)
    gaussian_image = np.clip((scaled_gaussian_prob * (255 - threshold) + threshold), 0, 255).astype(np.uint8)

    return gaussian_image

def generate_centerness_image(height, width, factor=4, threshold=128):
    bbox = [0, 0, width - 1, height - 1]
    x_range = np.arange(0, width)
    y_range = np.arange(0, height)
    index_x, index_y = np.meshgrid(x_range, y_range)

    left = index_x - bbox[0]
    left = np.maximum(left, 0)
    right = bbox[2] - index_x
    right = np.maximum(right, 0)
    top = index_y - bbox[1]
    top = np.maximum(top, 0)
    bottom = bbox[3] - index_y
    bottom = np.maximum(bottom, 0)

    centerness_prob = ((np.minimum(left, right) / (np.maximum(left, right) + 1)) * (np.minimum(top, bottom) / (np.maximum(top, bottom) + 1 ))) ** (1/factor)
    centerness_image = np.clip((centerness_prob * (255 - threshold) + threshold), 0, 255).astype(np.uint8)

    return centerness_image

def generate_ellipse_image(height, width, threshold=128):
    c_x = width // 2
    c_y = height // 2
    a = width / 2
    b = height / 2

    x_range = np.arange(0, width)
    y_range = np.arange(0, height)
    index_x, index_y = np.meshgrid(x_range, y_range)

    ellipse_image = ((index_x - c_x) / a) ** 2 + ((index_y - c_y) / b) ** 2

    ellipse_image[ellipse_image <= 1] = 0
    ellipse_image[ellipse_image > 1] = 1
    ellipse_image = 1 - ellipse_image

    ellipse_image = np.clip((ellipse_image * (255 - threshold) + threshold), 0, 255).astype(np.uint8)

    return ellipse_image
import numpy as np
import cv2
import wwtool

def impad(img, shape, pad_val=0, model='center'):
    pass

def convert_16bit_to_8bit(img):
    max_value = np.max(img)
    min_value = np.min(img)
    img = (img - min_value) / (max_value - min_value) * 255
    img = img.astype(np.uint8)

    return img

def split_image(img, subsize=1024, gap=200, mode='keep_all', expand_boundary=True):
    img_height, img_width = img.shape[0], img.shape[1]

    start_xs = np.arange(0, img_width, subsize - gap)
    if mode == 'keep_all':
        start_xs[-1] = img_width - subsize if img_width - start_xs[-1] <= subsize else start_xs[-1]
    elif mode == 'drop_boundary':
        if img_width - start_xs[-1] < subsize - gap:
            start_xs = np.delete(start_xs, -1)
    start_xs[-1] = np.maximum(start_xs[-1], 0)

    start_ys = np.arange(0, img_height, subsize - gap)
    if mode == 'keep_all':
        start_ys[-1] = img_height - subsize if img_height - start_ys[-1] <= subsize else start_ys[-1]
    elif mode == 'drop_boundary':
        if img_height - start_ys[-1] < subsize - gap:
            start_ys = np.delete(start_ys, -1)
    start_ys[-1] = np.maximum(start_ys[-1], 0)

    subimages = dict()
    
    for start_x in start_xs:
        for start_y in start_ys:
            end_x = np.minimum(start_x + subsize, img_width)
            end_y = np.minimum(start_y + subsize, img_height)
            if expand_boundary:
                subimage = wwtool.generate_image(subsize, subsize, color=(0, 0, 0))
                subimage[0:end_y-start_y, 0:end_x-start_x, ...] = img[start_y:end_y, start_x:end_x, ...]
            else:
                subimage = img[start_y:end_y, start_x:end_x, ...]
            coordinate = (start_x, start_y)
            subimages[coordinate] = subimage

    return subimages
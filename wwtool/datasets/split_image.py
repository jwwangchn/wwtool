import numpy as np
import cv2

def split_image(img, subsize=1024, gap=200):
    img_height, img_width = img.shape[0], img.shape[1]

    start_xs = np.arange(0, img_width, subsize - gap)
    start_xs[-1] = img_width - subsize if img_width - start_xs[-1] <= subsize else start_xs[-1]
    start_xs[-1] = np.maximum(start_xs[-1], 0)

    start_ys = np.arange(0, img_height, subsize - gap)
    start_ys[-1] = img_height - subsize if img_height - start_ys[-1] <= subsize else start_ys[-1]
    start_ys[-1] = np.maximum(start_ys[-1], 0)

    subimages = dict()
    for start_x in start_xs:
        for start_y in start_ys:
            end_x = np.minimum(start_x + subsize, img_width)
            end_y = np.minimum(start_y + subsize, img_height)
            subimage = img[start_y:end_y, start_x:end_x, ...]
            coordinate = (start_x, start_y)
            subimages[coordinate] = subimage
            # print(subimage.shape)

    return subimages

if __name__ == '__main__':
    img = np.random.rand(5292, 3371)
    subimages = split_image(img, 1024, 512)
    print(subimages.keys())
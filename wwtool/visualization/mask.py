import PIL
import numpy as np
import cv2
import shapely
import matplotlib.pyplot as plt

import wwtool


def get_palette(num_classes):
    n = num_classes
    palette = [0]*(n*3)
    for j in range(0, n):
        lab = j
        palette[j*3+0] = 0
        palette[j*3+1] = 0
        palette[j*3+2] = 0
        i = 0
        while (lab > 0):
            palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return palette

def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for _ in range(zero_pad):
        palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    new_mask = new_mask.convert('RGB')
    return new_mask

def show_mask(mask, 
              num_classes=2, 
              wait_time=0,
              out_file=None):
    palette = get_palette(num_classes)
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask_np = np.array(colorized_mask)
    colorized_mask_np = colorized_mask_np[:, :, ::-1].copy()
    wwtool.show_image(colorized_mask_np, 
                      wait_time=wait_time, 
                      win_name='mask',
                      save_name=out_file)

def show_polygons_on_image(masks, 
                           img, 
                           alpha=0.4, 
                           output_file=None):
    """show masks on image

    Args:
        masks (list): list of coordinate
        img (np.array): original image
        alpha (int): compress
        output_file (str): save path
    """
    color_list = list(wwtool.COLORS.keys())
    img_h, img_w, _ = img.shape

    foreground = wwtool.generate_image(img_h, img_w, (0, 0, 0))
    for idx, mask in enumerate(masks):
        mask = np.array(mask).reshape(1, -1, 2)
        cv2.fillPoly(foreground, mask, (wwtool.COLORS[color_list[idx % 20]][2], wwtool.COLORS[color_list[idx % 20]][1], wwtool.COLORS[color_list[idx % 20]][0]))

    heatmap = wwtool.show_grayscale_as_heatmap(foreground / 255.0, show=False, return_img=True)
    beta = (1.0 - alpha)
    fusion = cv2.addWeighted(heatmap, alpha, img, beta, 0.0)

    if output_file is not None:
        cv2.imwrite(output_file, fusion)
    else:
        wwtool.show_image(fusion, save_name=None)

    return fusion


def show_polygon(polygon):
    if type(polygon) == str:
        polygon = shapely.wkt.loads(polygon)

    plt.plot(*polygon.exterior.xy)
    plt.show()

    
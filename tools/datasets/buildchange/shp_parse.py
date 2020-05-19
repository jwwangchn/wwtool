import wwtool
import numpy as np
import rasterio as rio
import pycocotools.mask as maskUtils
import cv2
import pandas as pd
import geopandas as gpd

def poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


png_img_fn = './data/buildchange/v0/shanghai/geo_info/L18_106968_219344.png'
jpg_img_fn = './data/buildchange/v0/shanghai/images/L18_106968_219344.jpg'
# shp_fn = './data/buildchange/v0/shanghai/shp_4326/L18_106968_219320.shp'
shp_fn = './data/buildchange/v0/shanghai/merged_shp/L18_106968_219344.shp'


ori_img = rio.open(png_img_fn)
rgb_img = cv2.imread(jpg_img_fn)

shp_parser = wwtool.ShpParse()

objects = shp_parser(shp_fn, ori_img)

gt_masks = []

id_num = []
floors = []
polygons = []

for idx, obj in enumerate(objects):
    mask = obj['segmentation']
    gt_masks.append([mask])

print("Finish to parse shapefile!")

img = wwtool.generate_image(2048, 2048, (0, 0, 0))

COLORS = {'Blue': (0, 130, 200), 'Red': (230, 25, 75), 'Yellow': (255, 225, 25), 'Green': (60, 180, 75), 'Orange': (245, 130, 48), 'Purple': (145, 30, 180), 'Cyan': (70, 240, 240), 'Magenta': (240, 50, 230), 'Lavender': (230, 190, 255), 'Lime': (210, 245, 60), 'Teal': (0, 128, 128), 'Pink': (250, 190, 190), 'Brown': (170, 110, 40), 'Beige': (255, 250, 200), 'Maroon': (128, 0, 0), 'Mint': (170, 255, 195), 'Olive': (128, 128, 0), 'Apricot': (255, 215, 180), 'Navy': (0, 0, 128), 'Grey': (128, 128, 128), 'White': (255, 255, 255), 'Black': (0, 0, 0)}

color_list = list(COLORS.keys())

masks = wwtool.generate_image(2048, 2048)
for idx, gt_mask in enumerate(gt_masks):
    mask = poly2mask(gt_mask, 2048, 2048) * 1
    masks[:, :, 0] = mask * COLORS[color_list[idx % 20]][2]
    masks[:, :, 1] = mask * COLORS[color_list[idx % 20]][1]
    masks[:, :, 2] = mask * COLORS[color_list[idx % 20]][0]
    img += masks

heatmap = wwtool.show_grayscale_as_heatmap(img / 255.0, show=False, return_img=True)
alpha = 0.5
beta = (1.0 - alpha)
fusion = cv2.addWeighted(heatmap, alpha, rgb_img, beta, 0.0)

wwtool.show_image(fusion)
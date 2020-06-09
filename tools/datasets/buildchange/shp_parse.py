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


png_img_fn = '/data/buildchange/v0/shanghai/arg/geo_info/L18_106968_219360.png'
jpg_img_fn = '/data/buildchange/v0/shanghai/arg/images/L18_106968_219360.jpg'
shp_fn = '/data/buildchange/v0/shanghai/arg/roof_shp_4326/L18_106968_219360.shp'
# shp_fn = './data/buildchange/v0/shanghai/merged_shp/L18_106968_219352.shp'
# pixel_anno = '/data/buildchange/v0/shanghai/anno_v2/L18_106968_219352.png'
# roof_fn = './data/buildchange/v0/shanghai/roof_shp_4326/L18_106968_219352.shp'

ori_img = rio.open(jpg_img_fn)
rgb_img = cv2.imread(jpg_img_fn)

shp_parser = wwtool.ShpParse()

objects = shp_parser(shp_fn, 
                    geom_img=png_img_fn,
                    coord='4326',
                    ignore_file=None, 
                    merge_flag=False, 
                    connection_mode='floor')

gt_masks = [obj['segmentation'] for obj in objects]    
wwtool.show_polygons_on_image(gt_masks, rgb_img, output_file=None)
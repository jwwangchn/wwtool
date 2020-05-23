import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import Polygon

import wwtool


if __name__ == '__main__':
    shp_file = '/data/buildchange/v0/shanghai/merged_shp/L18_106968_219344.shp'
    geo_info = '/data/buildchange/v0/shanghai/geo_info/L18_106968_219344.png'
    pixel_anno = '/data/buildchange/v0/shanghai/anno_v2/L18_106968_219344.png'

    # 1. ignore polygons while parsing shapefile
    shp_parser = wwtool.ShpParse()
    objects = shp_parser(shp_file, geo_info, ignore_file=pixel_anno, show_ignored_polygons=True)



    # 2. direct test
    # shp_parser = wwtool.ShpParse()
    # objects = shp_parser(shp_file, geo_info)
    # foot_polygons = [obj['converted_polygon'] for obj in objects]

    # mask_parser = wwtool.MaskParse()
    # objects = mask_parser(pixel_anno, category=255)
    # ignore_polygons = [obj['polygon'] for obj in objects]

    # wwtool.cleaning_polygon_by_polygon(foot_polygons, ignore_polygons)
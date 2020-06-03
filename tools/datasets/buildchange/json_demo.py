import wwtool
import numpy as np
import rasterio as rio
import cv2
import os
import json
import mmcv
import wwtool
import matplotlib.pyplot as plt

import geopandas
from shapely.geometry import Polygon, MultiPolygon


def parse_json(json_file):
    annotations = mmcv.load(json_file)['annotations']

    roofs, footprints, ignores, offsets = [], [], [], []
    for annotation in annotations:
        roofs.append(wwtool.mask2polygon(annotation['roof']))
        footprints.append(wwtool.mask2polygon(annotation['footprint']))
        # ignore = annotation['ignore']
        # offset = annotation['offset']

    roof_polygons = geopandas.GeoSeries(roofs)
    footprint_polygons = geopandas.GeoSeries(footprints)

    if True:
        fig, ax = plt.subplots(1, 1)
        
        roof_df = geopandas.GeoDataFrame({'geometry': roof_polygons, 'foot_df':range(len(roof_polygons))})
        footprint_df = geopandas.GeoDataFrame({'geometry': footprint_polygons, 'foot_df':range(len(footprint_polygons))})

        roof_df.plot(ax=ax, color='red')
        footprint_df.plot(ax=ax, color='green')
        # plt.axis('off')

        # plt.savefig('./a.png', bbox_inches='tight', dpi=600, pad_inches=0.5)

        plt.show()


if __name__ == '__main__':
    image_dir = '/data/buildchange/v2/shanghai/images'
    label_dir = '/data/buildchange/v2/shanghai/labels_json'
    save_dir = '/data/buildchange/v2/shanghai/vis'
    wwtool.mkdir_or_exist(save_dir)

    img_scale = 1024
    for image_fn in os.listdir(image_dir):
        print(image_fn)
        if 'arg_L18_106968_219352__1024_1024' not in image_fn:
            continue
        image_file = os.path.join(image_dir, image_fn)
        label_file = os.path.join(label_dir, image_fn.replace('png', 'json'))
        save_file = os.path.join(save_dir, image_fn)
        img = cv2.imread(image_file)

        parse_json(label_file)



        # img_mask = wwtool.generate_image(img_scale, img_scale, (0, 0, 0))

        COLORS = {'Blue': (0, 130, 200), 'Red': (230, 25, 75), 'Yellow': (255, 225, 25), 'Green': (60, 180, 75), 'Orange': (245, 130, 48), 'Purple': (145, 30, 180), 'Cyan': (70, 240, 240), 'Magenta': (240, 50, 230), 'Lavender': (230, 190, 255), 'Lime': (210, 245, 60), 'Teal': (0, 128, 128), 'Pink': (250, 190, 190), 'Brown': (170, 110, 40), 'Beige': (255, 250, 200), 'Maroon': (128, 0, 0), 'Mint': (170, 255, 195), 'Olive': (128, 128, 0), 'Apricot': (255, 215, 180), 'Navy': (0, 0, 128), 'Grey': (128, 128, 128), 'White': (255, 255, 255), 'Black': (0, 0, 0)}

        color_list = list(COLORS.keys())
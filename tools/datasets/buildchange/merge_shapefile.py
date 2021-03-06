import os
import numpy as np
import cv2
from PIL import Image

import geojson
import shapely.wkt
import rasterio as rio

import wwtool
import pandas as pd
import geopandas as gpd

from multiprocessing import Pool
from functools import partial

import tqdm

pd.set_option('display.max_rows', None)

class MergeShapefile():
    def __init__(self,
                core_dataset_name,
                src_version,
                imageset,
                multi_processing=False,
                num_processor=16):
        self.image_path = './data/{}/{}/{}/images'.format(core_dataset_name, src_version, imageset)
        self.anno_path = './data/{}/{}/{}/shp_4326'.format(core_dataset_name, src_version, imageset)
        self.geo_path = './data/{}/{}/{}/geo_info'.format(core_dataset_name, src_version, imageset)

        self.bad_shapefile = './data/{}/{}/{}/bad_shapefile.txt'.format(core_dataset_name, src_version, imageset)

        self.merged_shapefile_save_path = './data/{}/{}/{}/merged_shp'.format(core_dataset_name, src_version, imageset)
        wwtool.mkdir_or_exist(self.merged_shapefile_save_path)

        self.core_dataset_name = core_dataset_name
        self.src_version = src_version
        self.imageset = imageset
        self.shp_parser = wwtool.ShpParse()
        self.multi_processing = multi_processing
        self.pool = Pool(num_processor)

    def merge_shapefile(self, image_fn):
        if not image_fn.endswith('.jpg'):
            return

        image_file = os.path.join(self.image_path, image_fn)
        file_name = os.path.splitext(os.path.basename(image_file))[0]

        merged_shapefile = os.path.join(self.merged_shapefile_save_path, file_name + '.shp')

        if os.path.exists(merged_shapefile):
            return

        annot_file = os.path.join(self.anno_path, file_name + '.shp')

        geo_info_file = os.path.join(self.geo_path, file_name + '.png')
        geo_info = rio.open(geo_info_file)
        coord_flag = '4326'

        objects = self.shp_parser(annot_file, 
                                geo_info,
                                coord=coord_flag,
                                merge_flag=True,
                                connection_mode='floor')
        if objects == []:
            with open(self.bad_shapefile, 'a') as f:
                f.write("{} {}\n".format(self.imageset, file_name + '.shp'))
            return

        properties = []
        polygons = []

        for idx, obj in enumerate(objects):
            ori_polygon = obj['ori_polygon']
            ori_property = obj['ori_property']
            property_dict = ori_property.to_dict()
            property_dict['Id'] = idx

            properties.append(property_dict)
            polygons.append(ori_polygon)
            
        df = pd.DataFrame(properties)
        gdf = gpd.GeoDataFrame(df, geometry=polygons, crs='EPSG:4326')
        gdf.to_file(merged_shapefile, encoding='utf-8')

    def core(self):
        image_fn_list = os.listdir(self.image_path)
        num_image = len(image_fn_list)
        if self.multi_processing:
            worker = partial(self.merge_shapefile)
            # self.pool.map(worker, self.merge_shapefile)
            ret = list(tqdm.tqdm(self.pool.imap(worker, image_fn_list), total=num_image))
        else:
            for image_fn in image_fn_list:
                self.merge_shapefile(image_fn)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state) 


if __name__ == '__main__':
    core_dataset_name = 'buildchange'
    src_version = 'v0'
    imagesets = ['shanghai']

    for imageset in imagesets:
        print("Begin processing {} set.".format(imageset))
        merge_shapefile = MergeShapefile(core_dataset_name=core_dataset_name,
                                         src_version=src_version,
                                         imageset=imageset,
                                         multi_processing=True,
                                         num_processor=16)
        merge_shapefile.core()
        print("Finish processing {} set.".format(imageset))
            

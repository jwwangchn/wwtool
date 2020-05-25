import os
import numpy as np
import cv2
from PIL import Image

import geojson
import shapely.wkt
import rasterio as rio
import geopandas
import wwtool
import mmcv

from multiprocessing import Pool
from functools import partial
from shapely.geometry import Polygon

import tqdm

Image.MAX_IMAGE_PIXELS = int(2048 * 2048 * 2048 // 4 // 3)


class SplitImage():
    def __init__(self, 
                 core_dataset_name='buildchange',
                 src_version='v0',
                 dst_version='v1',
                 imageset='shanghai',
                 sub_imageset_fold='arg',
                 subimage_size=1024,
                 gap=512,
                 num_processor=16,
                 show=False):
        self.core_dataset_name = core_dataset_name
        self.src_version = src_version
        self.dst_version = dst_version
        self.imageset = imageset
        self.sub_imageset_fold = sub_imageset_fold
        self.subimage_size = subimage_size
        self.gap = gap
        self.image_path = './data/{}/{}/{}/{}/images'.format(core_dataset_name, src_version, imageset, sub_imageset_fold)
        self.roof_shp_path = './data/{}/{}/{}/{}/roof_shp_4326'.format(core_dataset_name, src_version, imageset, sub_imageset_fold)
        self.geo_path = './data/{}/{}/{}/{}/geo_info'.format(core_dataset_name, src_version, imageset, sub_imageset_fold)

        self.image_save_path = './data/{}/{}/{}/{}/images'.format(core_dataset_name, dst_version, imageset, sub_imageset_fold)
        wwtool.mkdir_or_exist(self.image_save_path)
        self.label_save_path = './data/{}/{}/{}/{}/labels'.format(core_dataset_name, dst_version, imageset, sub_imageset_fold)
        wwtool.mkdir_or_exist(self.label_save_path)

        self.shp_parser = wwtool.ShpParse()
        self.pool = Pool(num_processor)
        self.show = show

    def drop_subimage(self, 
                      subimages, 
                      subimage_coordinate, 
                      subimage_masks,
                      center_area=2, 
                      small_object=64,
                      show=False):
        """judge whether to drop the overlap image

        Arguments:
            subimages {dict} -- dict which contains all subimages (value)
            subimage_coordinate {tuple} -- the coordinate of subimage in original image
            subimage_masks {list} -- list of masks in subimages

        Keyword Arguments:
            center_area {int} -- the area of center line (default: {2})
            show {bool} -- whether to show center line (default: {False})

        Returns:
            drop flag -- True: drop the subimage, False: keep the subimage
        """
        # black image
        if np.mean(subimages[subimage_coordinate]) == 0:
            return True

        # no object
        if len(subimage_masks) == 0:
            return True

        # keep the main subimage, just drop the overlap part
        if abs(subimage_coordinate[0] - subimage_coordinate[1]) in (0, 1024) and (subimage_coordinate[0] != 512 and subimage_coordinate[1] != 512):
            return False

        subimage_mask_area = []
        subimage_mask_polygons = []
        for subimage_mask in subimage_masks:
            subimage_mask_polygon = wwtool.mask2polygon(subimage_mask)
            subimage_mask_polygons.append(subimage_mask_polygon)
            subimage_mask_area.append(subimage_mask_polygon.area)

        # (horizontal, vertical)
        center_lines = [Polygon([(0, 512 - center_area), 
                                (0, 512 + center_area), 
                                (1023, 512 + center_area), 
                                (1023, 512 - center_area), 
                                (0, 512 - center_area)]), 
                        Polygon([(512 - center_area, 0), 
                                (512 + center_area, 0), 
                                (512 + center_area, 1023), 
                                (512 - center_area, 1023), 
                                (512 - center_area, 0)])]

        if subimage_coordinate[0] == 512 and subimage_coordinate[1] != 512:
            center_lines = [center_lines[1]]
        elif subimage_coordinate[0] != 512 and subimage_coordinate[1] == 512:
            center_lines = [center_lines[0]]
        else:
            center_lines = center_lines

        subimage_mask_polygons = wwtool.clean_polygon(subimage_mask_polygons)
        subimage_mask_df = geopandas.GeoDataFrame({'geometry': subimage_mask_polygons, 'submask_df':range(len(subimage_mask_polygons))})
        center_line_df = geopandas.GeoDataFrame({'geometry': center_lines, 'center_df':range(len(center_lines))})

        image_border_polygon = [Polygon([(0, 0), (1024-1, 0), (1024-1, 1024-1), (0, 1024-1), (0, 0)])]
        border_line_df = geopandas.GeoDataFrame({'geometry': image_border_polygon, 'border_df':range(len(image_border_polygon))})

        if show:
            fig, ax = plt.subplots()   

            subimage_mask_df.plot(ax=ax, color='red')
            center_line_df.plot(ax=ax, facecolor='none', edgecolor='g')
            border_line_df.plot(ax=ax, facecolor='none', edgecolor='k')
            ax.set_title('{}_{}'.format(subimage_coordinate[0], subimage_coordinate[1]))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

            # plt.show()
            plt.savefig('./{}_{}_{}.png'.format(self.image_fn.replace('.jpg', ''), subimage_coordinate[0], subimage_coordinate[1]), bbox_inches='tight', dpi=600, pad_inches=0.0)

        res_intersection = geopandas.overlay(subimage_mask_df, center_line_df, how='intersection')
        inter_dict = res_intersection.to_dict()
        ignore_indexes = list(set(inter_dict['submask_df'].values()))

        inter_areas = []
        for ignore_index in ignore_indexes:
            inter_areas.append(subimage_mask_polygons[ignore_index].area)

        if len(inter_areas) == 0 or max(inter_areas) < small_object * small_object:
            return True
        else:
            return False

    def split_image(self, image_fn):
        if not image_fn.endswith('.jpg'):
            return
        image_file = os.path.join(self.image_path, image_fn)
        shp_file = os.path.join(self.roof_shp_path, image_fn.replace('jpg', 'shp'))
        geo_file = os.path.join(self.geo_path, image_fn.replace('jpg', 'png'))
        
        file_name = os.path.splitext(os.path.basename(image_file))[0]

        if not os.path.exists(shp_file):
            return

        img = cv2.imread(image_file)
        geo_info = rio.open(geo_file)

        objects = self.shp_parser(shp_file, geo_info)

        masks = np.array([obj['segmentation'] for obj in objects])

        subimages = wwtool.split_image(img, subsize=self.subimage_size, gap=self.gap)
        subimage_coordinates = list(subimages.keys())

        if masks.shape[0] == 0:
            return

        mask_centroids = []
        for obj in objects:
            geometry = obj['converted_polygon'].centroid
            geo = geojson.Feature(geometry=geometry, properties={})
            coordinate = geo.geometry["coordinates"]
            coordinate[0], coordinate[1] = abs(coordinate[0]), abs(coordinate[1])
            mask_centroids.append(coordinate)

        mask_centroids = np.array(mask_centroids)
        mask_centroids_ = mask_centroids.copy()
        
        for subimage_coordinate in subimage_coordinates:
            objects = []

            mask_centroids_[:, 0] = mask_centroids[:, 0] - subimage_coordinate[0]
            mask_centroids_[:, 1] = mask_centroids[:, 1] - subimage_coordinate[1]


            cx_bool = np.logical_and(mask_centroids_[:, 0] >= 0, mask_centroids_[:, 0] < subimage_size)
            cy_bool = np.logical_and(mask_centroids_[:, 1] >= 0, mask_centroids_[:, 1] < subimage_size)

            subimage_masks = masks[np.logical_and(cx_bool, cy_bool)]

            subimage_masks_ = []
            for subimage_mask in subimage_masks:
                if wwtool.mask2polygon(subimage_mask).area < 5:
                    continue
                subimage_mask_np = np.array(subimage_mask)
                subimage_mask_np[0::2] = subimage_mask_np[0::2] - subimage_coordinate[0]
                subimage_mask_np[1::2] = subimage_mask_np[1::2] - subimage_coordinate[1]

                subimage_masks_.append(subimage_mask_np.tolist())
            

            subimage_masks = subimage_masks_
            # cut the polygons by image boundary
            subimage_masks = wwtool.clip_mask(subimage_masks, image_size=(1024, 1024))

            # judge whether to drop this subimage
            drop_flag = self.drop_subimage(subimages, 
                                           subimage_coordinate, 
                                           subimage_masks,
                                           show=self.show)
            if drop_flag:
                continue

            img = subimages[subimage_coordinate]

            label_save_file = os.path.join(self.label_save_path, '{}_{}__{}_{}.txt'.format(self.sub_imageset_fold, file_name, subimage_coordinate[0], subimage_coordinate[1]))
            image_save_file = os.path.join(self.image_save_path, '{}_{}__{}_{}.png'.format(self.sub_imageset_fold, file_name, subimage_coordinate[0], subimage_coordinate[1]))
            cv2.imwrite(image_save_file, img)
            
            for subimage_mask in subimage_masks:
                subimage_objects = dict()
                subimage_objects['mask'] = subimage_mask
                subimage_objects['label'] = 'building'
                objects.append(subimage_objects)
            wwtool.simpletxt_dump(objects, label_save_file, encode='mask')

    def core(self):
        image_fn_list = os.listdir(self.image_path)
        num_image = len(image_fn_list)
        worker = partial(self.split_image)
        # self.pool.map(worker, image_fn_list)
        ret = list(tqdm.tqdm(self.pool.imap(worker, image_fn_list), total=num_image))

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


if __name__ == '__main__':
    core_dataset_name = 'buildchange'
    src_version = 'v0'
    dst_version = 'v2'
    # imagesets = ['shanghai']
    # sub_imageset_folds = {'shanghai': ['arg']}
    imagesets = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    sub_imageset_folds = {'beijing': ['arg', 'google', 'ms', 'tdt'],
                         'chengdu': ['arg', 'google', 'ms', 'tdt'],
                         'haerbin': ['arg', 'google', 'ms'],
                         'jinan': ['arg', 'google', 'ms', 'tdt'],
                         'shanghai': ['arg', 'google', 'ms', 'tdt', 'PHR2016', 'PHR2017']}
    subimage_size = 1024
    gap = subimage_size // 2

    for imageset in imagesets:
        for sub_imageset_fold in sub_imageset_folds[imageset]:
            print("Begin processing {} set.".format(imageset))
            split_image = SplitImage(core_dataset_name=core_dataset_name,
                                    src_version=src_version,
                                    dst_version=dst_version,
                                    imageset=imageset,
                                    sub_imageset_fold=sub_imageset_fold,
                                    subimage_size=subimage_size,
                                    gap=gap,
                                    num_processor=16)

            split_image.core()
            print("Finish processing {} set.".format(imageset))

        

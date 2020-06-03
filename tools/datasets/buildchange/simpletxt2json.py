import os
import numpy as np
import shapely
import cv2
import json
from shapely import affinity
from multiprocessing import Pool
from functools import partial

import mmcv
import wwtool

import tqdm


class Simpletxt2Json():
    def __init__(self,
                dst_version,
                city,
                sub_imageset_folds,
                multi_processing=False,
                num_processor=16):
        self.splitted_image_dir = './data/buildchange/{}/{}/images'.format(dst_version, city)
        self.splitted_label_dir = './data/buildchange/{}/{}/labels'.format(dst_version, city)
        self.json_dir = '/data/buildchange/v2/{}/labels_json'.format(city)
        self.wrong_shp_file_dict = dict()
        for sub_fold in sub_imageset_folds[city]:
            wrong_file = './data/buildchange/v0/{}/{}/wrongShpFile.txt'.format(city, sub_fold)
            ori_filenames = self.read_wrong_file(wrong_file)
            self.wrong_shp_file_dict[sub_fold] = ori_filenames
        wwtool.mkdir_or_exist(self.json_dir)
        self.city = city
        self.multi_processing = multi_processing
        self.pool = Pool(num_processor)

    def read_wrong_file(self, wrong_file):
        ori_filenames = []
        with open(wrong_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                ori_filename = line.strip('\n').split('/')[-1].split('.csv')[0]
                ori_filenames.append(ori_filename)
        
        return ori_filenames

    def simpletxt_parse(self, label_file):
        """parse simpletxt style dataset label file
        
        Arguments:
            label_file {str} -- label file path
        
        Returns:
            dict, {'bbox': [...], 'label': class_name} -- objects' location and class
        """
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        objects = []
        basic_label_str = " "
        for line in lines:
            object_struct = dict()
            line = line.rstrip().split(' ')
            label = basic_label_str.join(line[-1])
            polygon = [float(_) for _ in line[0:-1]]
            object_struct['polygon'] = polygon
            object_struct['label'] = label
            objects.append(object_struct)
        
        return objects

    def get_footprint(self, mask, coordinate, roof_polygons, roof_properties):
        # print(mask, coordinate, roof_polygon, roof_property)
        transform_matrix = [1, 0, 0, 1, coordinate[0], coordinate[1]]
        roi_mask = affinity.affine_transform(mask, transform_matrix)
        # print("move: ", mask, moved_mask, coordinate)
        for idx, roof_polygon in enumerate(roof_polygons):
            if roof_polygon.equals(roi_mask):
                xoffset = roof_properties[idx].to_dict()['xoffset']
                yoffset = roof_properties[idx].to_dict()['yoffset']
                break
            else:
                xoffset, yoffset = 0, 0

        transform_matrix = [1, 0, 0, 1, -coordinate[0], -coordinate[1]]
        split_mask = affinity.affine_transform(roi_mask, transform_matrix)
        transform_matrix = [1, 0, 0, 1, -xoffset, -yoffset]
        footprint_polygon = affinity.affine_transform(split_mask, transform_matrix)        

        return footprint_polygon, xoffset, yoffset

    def simpletxt2json(self, image_fn):
        # 1. open the ignore file and get the polygons
        base_name = wwtool.get_basename(image_fn)
        sub_fold = base_name.split("__")[0].split('_')[0]
        ori_image_fn = "_".join(base_name.split("__")[0].split('_')[1:])
        if ori_image_fn in self.wrong_shp_file_dict[sub_fold]:
            print("Skip this wrong shape file")
            return
        coord_x, coord_y = base_name.split("__")[1].split('_')    # top left corner
        coord_x, coord_y = int(coord_x), int(coord_y)
        print(f"splitted items: {sub_fold}, {ori_image_fn}, {(coord_x, coord_y)}")

        ignore_file = './data/buildchange/{}/{}/{}/pixel_anno_v2/{}'.format(src_version, self.city, sub_fold, ori_image_fn + '.png')
        # print("ignore file name: ", ignore_file)
        roof_shp_file = './data/buildchange/{}/{}/{}/roof_shp_4326/{}'.format(src_version, self.city, sub_fold, ori_image_fn + '.shp')
        geo_info_file = './data/buildchange/{}/{}/{}/geo_info/{}'.format(src_version, self.city, sub_fold, ori_image_fn + '.png')

        objects = shp_parser(roof_shp_file, geo_info_file)
        roof_polygon_4326 = [obj['converted_polygon'] for obj in objects]
        roof_property = [obj['converted_property'] for obj in objects]

        pixel_anno = cv2.imread(ignore_file)
        objects = mask_parser(pixel_anno[coord_y:coord_y + sub_img_h, coord_x:coord_x + sub_img_w, :], category=255)
        if objects == []:
            return
        ignore_polygons = [obj['polygon'] for obj in objects]
        # print("ignore polygon: ", ignore_polygons)

        # 2. read the simpletxt file and convert to polygons
        objects = self.simpletxt_parse(os.path.join(self.splitted_label_dir, base_name + '.txt'))
        roof_polygons = [wwtool.mask2polygon(obj['polygon']) for obj in objects]
        # print("roof polygon: ", roof_polygons)

        _, ignore_indexes = wwtool.cleaning_polygon_by_polygon(roof_polygons[:], ignore_polygons, show=False)
        ignore_list = len(roof_polygons) * [0]
        for ignore_index in ignore_indexes:
            ignore_list[ignore_index] = 1

        new_anno_objects = []
        for idx, roof_polygon in enumerate(roof_polygons):
            footprint_polygon, xoffset, yoffset = self.get_footprint(roof_polygon, [coord_x, coord_y], roof_polygon_4326, roof_property)
            object_struct = dict()
            ignore_flag = ignore_list[idx]
            object_struct['roof'] = wwtool.polygon2mask(roof_polygon)
            object_struct['footprint'] = wwtool.polygon2mask(footprint_polygon)
            object_struct['offset'] = [xoffset, yoffset]
            object_struct['ignore'] = ignore_flag
            new_anno_objects.append(object_struct)

        image_info = {
                        "ori_filename": ori_image_fn + '.jpg',
                        "subimage_filename": image_fn,
                        "width": 1024,
                        "height": 1024,
                        "city": self.city,
                        "sub_fold": sub_fold,
                        "coordinate": [coord_x, coord_y]
                    }

        json_data = {"image": image_info,
                    "annotations": new_anno_objects
                    }

        json_file = os.path.join(self.json_dir, f'{base_name}.json')
        with open(json_file, "w") as jsonfile:
            json.dump(json_data, jsonfile, indent=4)

    def core(self):
        if self.multi_processing:
            image_fn_list = os.listdir(self.splitted_image_dir)
            num_image = len(image_fn_list)
            worker = partial(self.simpletxt2json)
            # self.pool.map(worker, image_fn_list)
            ret = list(tqdm.tqdm(self.pool.imap(worker, image_fn_list), total=num_image))
        else:
            image_fn_list = os.listdir(self.splitted_image_dir)
            progress_bar = mmcv.ProgressBar(len(image_fn_list))
            for _, image_fn in enumerate(image_fn_list):
                self.simpletxt2json(image_fn)
                progress_bar.update()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


if __name__ == '__main__':
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    sub_imageset_folds = {'beijing': ['arg', 'google', 'ms', 'tdt'],
                    'chengdu': ['arg', 'google', 'ms', 'tdt'],
                    'haerbin': ['arg', 'google', 'ms'],
                    'jinan': ['arg', 'google', 'ms', 'tdt'],
                    'shanghai': ['google', 'ms', 'tdt', 'PHR2016', 'PHR2017']}
    # cities = ['shanghai']
    # sub_imageset_folds = {'shanghai': ['arg']}
    
    core_dataset_name = 'buildchange'
    src_version = 'v0'
    dst_version = 'v2'
    sub_img_w, sub_img_h = 1024, 1024

    mask_parser = wwtool.MaskParse()
    shp_parser = wwtool.ShpParse()

    for city in cities:
        convert = Simpletxt2Json(dst_version=dst_version, 
                                 city=city,
                                 sub_imageset_folds=sub_imageset_folds,
                                 multi_processing=True,
                                 num_processor=32)
        convert.core()

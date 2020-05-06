import argparse

import os
import cv2
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import rasterio as rio

import wwtool
from wwtool.datasets import Convert2COCO

class SHP2COCO(Convert2COCO):
    def __generate_coco_annotation__(self, annotpath, imgpath):
        """
        docstring here
            :param self: 
            :param annotpath: the path of each annotation
            :param return: dict()  
        """
        objects = self.__shp_parse__(annotpath, imgpath)
        
        coco_annotations = []

        for object_struct in objects:
            bbox = object_struct['bbox']
            segmentation = object_struct['segmentation']
            label = object_struct['label']

            width = bbox[2]
            height = bbox[3]
            area = height * width

            if area <= self.small_object_area and self.groundtruth:
                self.small_object_idx += 1
                continue

            coco_annotation = {}
            coco_annotation['bbox'] = bbox
            coco_annotation['segmentation'] = [segmentation]
            coco_annotation['category_id'] = label
            coco_annotation['area'] = np.float(area)

            coco_annotations.append(coco_annotation)
            
        return coco_annotations
    
    def __shp_parse__(self, label_file, image_file):
        """
        (xmin, ymin, xmax, ymax)
        """
        img_fn = os.path.splitext(os.path.basename(image_file))[0]

        if imageset == 'train':
            geo_info_file = os.path.join(geopath, img_fn + '.png')
            geo_info = rio.open(geo_info_file)
        else:
            geo_info = rio.open(image_file)

        objects = []
        masks = shp_parser(label_file, geo_info)
        total_object_num = len(masks)
        for mask in masks:
            object_struct = {}

            xmin, ymin, xmax, ymax = wwtool.pointobb2bbox(mask['segmentation'])
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
            object_struct['segmentation'] = mask['segmentation']
            object_struct['label'] = 1
            
            objects.append(object_struct)
        
        if total_object_num > self.max_object_num_per_image:
            self.max_object_num_per_image = total_object_num

        geo_info.close()
        return objects

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument(
        '--imagesets',
        type=str,
        nargs='+',
        choices=['trainval', 'test'])
    parser.add_argument(
        '--release_version', default='v1', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # basic dataset information
    info = {"year" : 2019,
            "version" : "1.0",
            "description" : "SHP-COCO",
            "contributor" : "Jinwang Wang",
            "date_created" : "2020"
            }
    
    licenses = [{"id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }]

    # dataset's information
    image_format='.jpg'
    anno_format='.shp'

    shp_class = [{'supercategory': 'none', 'id': 1,  'name': 'footprint',                 }]

    core_dataset_name = 'buildchange'
    imagesets = ['val_xian']
    release_version = 'v1'
    keypoint = False

    shp_parser = wwtool.ShpParse()

    anno_name = [core_dataset_name, release_version]
    if keypoint:
        for idx in range(len(shp_class)):
            shp_class[idx]["keypoints"] = ['top', 'right', 'bottom', 'left']
            shp_class[idx]["skeleton"] = [[1,2], [2,3], [3,4], [4,1]]
        anno_name.append('keypoint')
    
    for imageset in imagesets:
        imgpath = './data/{}/{}/{}/images'.format(core_dataset_name, release_version, imageset)
        annopath = './data/{}/{}/{}/shp_4326'.format(core_dataset_name, release_version, imageset)
        geopath = './data/{}/{}/{}/geo_info'.format(core_dataset_name, release_version, imageset)
        save_path = './data/{}/{}/coco/annotations'.format(core_dataset_name, release_version)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if imageset == 'val':
            sub_anno_fold = True
        else:
            sub_anno_fold = False

        shp = SHP2COCO(imgpath=imgpath,
                        annopath=annopath,
                        image_format=image_format,
                        anno_format=anno_format,
                        data_categories=shp_class,
                        data_info=info,
                        data_licenses=licenses,
                        data_type="instances",
                        groundtruth=True,
                        small_object_area=0,
                        sub_anno_fold=sub_anno_fold)

        images, annotations = shp.get_image_annotation_pairs()

        json_data = {"info" : shp.info,
                    "images" : images,
                    "licenses" : shp.licenses,
                    "type" : shp.type,
                    "annotations" : annotations,
                    "categories" : shp.categories}

        anno_name.insert(1, imageset)
        with open(os.path.join(save_path, "_".join(anno_name) + ".json"), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)
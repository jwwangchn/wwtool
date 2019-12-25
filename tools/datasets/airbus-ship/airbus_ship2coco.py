import argparse

import os
import cv2
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET

import wwtool
from wwtool.datasets import Convert2COCO

class AirbusShip2COCO(Convert2COCO):
    def __generate_coco_annotation__(self, annotpath, imgpath):
        """
        docstring here
            :param self: 
            :param annotpath: the path of each annotation
            :param return: dict()  
        """
        objects = self.__airbus_ship_parse__(annotpath, imgpath)
        
        coco_annotations = []

        if generate_small_dataset and len(objects) > 0:
            wwtool.generate_same_dataset(imgpath, 
                                        annotpath,
                                        dst_image_path,
                                        dst_label_path,
                                        src_img_format='.png',
                                        src_anno_format='.txt',
                                        dst_img_format='.png',
                                        dst_anno_format='.txt',
                                        parse_fun=wwtool.voc_parse,
                                        dump_fun=wwtool.simpletxt_dump)

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
    
    def __airbus_ship_parse__(self, label_file, image_file):
        """
        (xmin, ymin, xmax, ymax)
        """
        image_objects = airbus_ship_parse.airbus_ship_parse(os.path.basename(image_file))
        objects = []
        total_object_num = len(image_objects)
        small_object_num = 0
        large_object_num = 0
        total_object_num = 0
        for image_object in image_objects:
            object_struct = {}
            xmin, ymin, xmax, ymax = image_object['bbox']
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            total_object_num += 1
            if bbox_h * bbox_w <= small_size:
                small_object_num += 1
            if bbox_h * bbox_w >= large_object_size:
                large_object_num += 1

            object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
            object_struct['segmentation'] = wwtool.bbox2pointobb([xmin, ymin, xmax, ymax])
            object_struct['label'] = 1
            
            objects.append(object_struct)
        
        if total_object_num > self.max_object_num_per_image:
            self.max_object_num_per_image = total_object_num

        if just_keep_small or generate_small_dataset:
            if small_object_num >= total_object_num * small_object_rate and large_object_num < 1:
                return objects
            else:
                return []
        else:
            return objects
            

if __name__ == "__main__":
    # basic dataset information
    info = {"year" : 2019,
                "version" : "1.0",
                "description" : "Airbus-Ship-COCO",
                "contributor" : "Jinwang Wang",
                "url" : "jwwangchn.cn",
                "date_created" : "2019"
            }
    
    licenses = [{"id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }]

    # dataset's information
    image_format='.jpg'
    anno_format='.csv'

    original_airbus_ship_class = {'ship' : 1}

    converted_airbus_ship_class = [{'supercategory': 'none', 'id': 1,  'name': 'ship'}]

    core_dataset_name = 'airbus-ship'
    imagesets = ['train']
    release_version = 'v1'
    rate = '1.0'
    groundtruth = True
    keypoint = False
    
    just_keep_small = True
    generate_small_dataset = False
    small_size = 16 * 16
    small_object_rate = 0.5
    large_object_size = 128 * 128

    anno_name = [core_dataset_name, release_version, rate]
    if just_keep_small:
        anno_name.append('small_object')

    if generate_small_dataset:
        dst_image_path = '/data/small/{}/images'.format(core_dataset_name)
        dst_label_path = '/data/small/{}/labels'.format(core_dataset_name)
        wwtool.mkdir_or_exist(dst_image_path)
        wwtool.mkdir_or_exist(dst_label_path)

    if keypoint:
        for idx in range(len(converted_airbus_ship_class)):
            converted_airbus_ship_class[idx]["keypoints"] = ['top', 'right', 'bottom', 'left']
            converted_airbus_ship_class[idx]["skeleton"] = [[1,2], [2,3], [3,4], [4,1]]
        anno_name.append('keypoint')
    
    if groundtruth == False:
        anno_name.append('no_ground_truth')

    for imageset in imagesets:
        imgpath = './data/{}/{}/{}/images'.format(core_dataset_name, release_version, imageset)
        annopath = './data/{}/{}/{}/labels'.format(core_dataset_name, release_version, imageset)
        save_path = './data/{}/{}/coco/annotations'.format(core_dataset_name, release_version)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        anno_file = os.path.join(annopath, 'train_ship_segmentations_v2.csv')
        airbus_ship_parse = wwtool.AirbusShipParse(anno_file)

        airbus_ship = AirbusShip2COCO(imgpath=imgpath,
                        annopath=annopath,
                        image_format=image_format,
                        anno_format=anno_format,
                        data_categories=converted_airbus_ship_class,
                        data_info=info,
                        data_licenses=licenses,
                        data_type="instances",
                        groundtruth=groundtruth,
                        small_object_area=0)

        images, annotations = airbus_ship.get_image_annotation_pairs()

        json_data = {"info" : airbus_ship.info,
                    "images" : images,
                    "licenses" : airbus_ship.licenses,
                    "type" : airbus_ship.type,
                    "annotations" : annotations,
                    "categories" : airbus_ship.categories}

        anno_name.insert(1, imageset)
        with open(os.path.join(save_path, "_".join(anno_name) + ".json"), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)
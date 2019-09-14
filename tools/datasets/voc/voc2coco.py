import argparse

import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET

from wwtool.datasets.convert2coco import Convert2COCO
from wwtool.transforms import bbox2pointobb, pointobb_extreme_sort, pointobb_best_point_sort


class VOC2COCO(Convert2COCO):
    def __generate_coco_annotation__(self, annotpath, imgpath):
        """
        docstring here
            :param self: 
            :param annotpath: the path of each annotation
            :param return: dict()  
        """
        objects = self.__voc_parse__(annotpath, imgpath)

        coco_annotations = []
        
        for object_struct in objects:
            bbox = object_struct['bbox']
            label = object_struct['label']
            segmentation = object_struct['segmentation']
            pointobb = object_struct['pointobb']

            width = bbox[2]
            height = bbox[3]
            area = height * width

            if area < self.small_object_area and self.groundtruth:
                self.small_object_idx += 1
                continue

            coco_annotation = {}
            coco_annotation['bbox'] = bbox
            coco_annotation['category_id'] = label
            coco_annotation['area'] = np.float(area)
            coco_annotation['segmentation'] = [segmentation]
            coco_annotation['pointobb'] = pointobb

            coco_annotations.append(coco_annotation)
            
        return coco_annotations
    
    def __voc_parse__(self, label_file, image_file):
        tree = ET.parse(label_file)
        root = tree.getroot()
        objects = []
        for single_object in root.findall('object'):
            bndbox = single_object.find('bndbox')
            object_struct = {}

            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            pointobb = bbox2pointobb([xmin, ymin, xmax, ymax])
            object_struct['segmentation'] = [pointobb]
            object_struct['pointobb'] = pointobb_sort_function[pointobb_sort_method](pointobb)
            object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
            object_struct['label'] = voc_class[single_object.find('name').text]
            
            objects.append(object_struct)
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
            "description" : "VOC-COCO",
            "contributor" : "Jinwang Wang",
            "url" : "jwwangchn.cn",
            "date_created" : "2019"
            }
    
    licenses = [{"id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }]

    image_format='.jpg'
    anno_format='.xml'

    voc_class = {'ship': 1}
    coco_class = [{'supercategory': 'none', 'id': 1,  'name': 'ship',                }]

    imagesets = ['trainval', 'test']
    core_dataset = 'sarship'
    groundtruth = True
    release_version = 'v1'

    pointobb_sort_method = 'best' # or "extreme"
    pointobb_sort_function = {"best": pointobb_best_point_sort,
                            "extreme": pointobb_extreme_sort}

    for imageset in imagesets:
        imgpath = './data/{}/{}/{}/images'.format(core_dataset, release_version, imageset)
        annopath = './data/{}/{}/{}/labels'.format(core_dataset, release_version, imageset)
        save_path = './data/{}/{}/coco/annotations'.format(core_dataset, release_version)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        voc = VOC2COCO(imgpath=imgpath,
                        annopath=annopath,
                        image_format=image_format,
                        anno_format=anno_format,
                        data_categories=coco_class,
                        data_info=info,
                        data_licenses=licenses,
                        data_type="instances",
                        groundtruth=groundtruth,
                        small_object_area=0)

        images, annotations = voc.get_image_annotation_pairs()

        json_data = {"info" : voc.info,
                    "images" : images,
                    "licenses" : voc.licenses,
                    "type" : voc.type,
                    "annotations" : annotations,
                    "categories" : voc.categories}

        with open(os.path.join(save_path, "sarship_" + imageset + ".json"), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)
import argparse

import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET

from wwtool.datasets.convert2coco import Convert2COCO
from wwtool.transforms import bbox2pointobb, pointobb_extreme_sort, pointobb_best_point_sort

import wwtool


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
        objects_handle = root.find('objects')
        for single_object in objects_handle.findall('object'):
            points = single_object.find('points')
            object_struct = {}

            pointobb = []
            for point in points[:-1]:
                coords = [float(coord) for coord in point.text.split(',')]
                pointobb += coords

            bbox = wwtool.pointobb2bbox(pointobb)
            bbox = wwtool.xyxy2xywh(bbox)

            object_struct['segmentation'] = pointobb
            object_struct['pointobb'] = pointobb_sort_function[pointobb_sort_method](pointobb)
            object_struct['bbox'] = bbox
            object_struct['label'] = voc_class[single_object.find('possibleresult').find('name').text]
            
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

    image_format='.tif'
    anno_format='.xml'

    voc_class = {'Boeing737': 1, 'Boeing747': 2, 'Boeing777': 3, 'Boeing787': 4, 'A220': 5, 'A321': 6, 'A330': 7, 'A350': 8, 'ARJ21': 9, 'other': 10}

    coco_class = [{'supercategory': 'plane', 'id': 1,  'name': 'Boeing737',      },
                  {'supercategory': 'plane', 'id': 2,  'name': 'Boeing747',      }, 
                  {'supercategory': 'plane', 'id': 3,  'name': 'Boeing777',      },
                  {'supercategory': 'plane', 'id': 4,  'name': 'Boeing787',      },
                  {'supercategory': 'plane', 'id': 5,  'name': 'A220',     },
                  {'supercategory': 'plane', 'id': 6,  'name': 'A321',     },
                  {'supercategory': 'plane', 'id': 7,  'name': 'A330',     },
                  {'supercategory': 'plane', 'id': 8,  'name': 'A350',     },
                  {'supercategory': 'plane', 'id': 9,  'name': 'ARJ21',     },
                  {'supercategory': 'plane', 'id': 10, 'name': 'other',          }]

    imagesets = ['train']
    core_dataset = 'plane'
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

        with open(os.path.join(save_path, "plane_" + imageset + ".json"), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)
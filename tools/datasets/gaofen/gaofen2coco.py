import argparse

import os
import cv2
import json
import numpy as np

import wwtool
from wwtool.datasets import Convert2COCO
from wwtool.transforms import thetaobb2hobb, pointobb2thetaobb, pointobb2sampleobb, pointobb_extreme_sort, pointobb_best_point_sort

class GaoFen2COCO(Convert2COCO):
    def __generate_coco_annotation__(self, annotpath, imgpath):
        """
        docstring here
            :param self: 
            :param annotpath: the path of each annotation
            :param return: dict()  
        """
        objects = self.__simpletxt_parse__(annotpath)
        
        coco_annotations = []
        
        for object_struct in objects:
            bbox = object_struct['bbox']
            label = object_struct['label']
            segmentation = object_struct['segmentation']
            pointobb = object_struct['pointobb']
            thetaobb = object_struct['thetaobb']
            hobb = object_struct['hobb']
            keypoint = object_struct['keypoints']

            width = bbox[2]
            height = bbox[3]
            area = height * width

            if area <= self.small_object_area and self.groundtruth:
                self.small_object_idx += 1
                continue

            coco_annotation = {}
            coco_annotation['bbox'] = bbox
            coco_annotation['category_id'] = label
            coco_annotation['area'] = np.float(area)
            coco_annotation['segmentation'] = [segmentation]
            coco_annotation['pointobb'] = pointobb
            coco_annotation['thetaobb'] = thetaobb
            coco_annotation['hobb'] = hobb
            coco_annotation['keypoints'] = keypoint
            coco_annotation['num_keypoints'] = 4

            coco_annotations.append(coco_annotation)
            
        return coco_annotations
    
    def __simpletxt_parse__(self, label_file):
        objects = []
        if self.groundtruth:
            lines = open(label_file, 'r').readlines()
            for line in lines:
               
                obj_struct = {}

                thetaobb = [float(xy) for xy in line.rstrip().split(' ')[:-1]]
                pointobb = wwtool.thetaobb2pointobb(thetaobb)
                obj_struct['segmentation'] = pointobb
                obj_struct['pointobb'] = pointobb_sort_function[pointobb_sort_method](pointobb)
                obj_struct['thetaobb'] = pointobb2thetaobb(pointobb)
                obj_struct['hobb'] = thetaobb2hobb(obj_struct['thetaobb'], pointobb_sort_function[pointobb_sort_method])

                obj_struct['keypoints'] = obj_struct['pointobb'][:]
                for idx in [2, 5, 8, 11]:
                    obj_struct['keypoints'].insert(idx, 2)

                xmin = min(pointobb[0::2])
                ymin = min(pointobb[1::2])
                xmax = max(pointobb[0::2])
                ymax = max(pointobb[1::2])
                bbox_w = xmax - xmin
                bbox_h = ymax - ymin
                obj_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
                obj_struct['label'] = original_class[line.rstrip().split(' ')[-1]]

                objects.append(obj_struct)
        else:
            obj_struct = {}
            obj_struct['segmentation'] = [0, 0, 0, 0, 0, 0, 0, 0]
            obj_struct['keypoint'] = [0, 0, 0, 0, 0, 0, 0, 0]
            obj_struct['pointobb'] = [0, 0, 0, 0, 0, 0, 0, 0]
            obj_struct['thetaobb'] = [0, 0, 0, 0, 0]
            obj_struct['hobb'] = [0, 0, 0, 0, 0]
            obj_struct['bbox'] = [0, 0, 0, 0]
            obj_struct['label'] = 0

            objects.append(obj_struct)
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
                "version" : "1.5",
                "description" : "DOTA-COCO",
                "contributor" : "Jinwang Wang",
                "url" : "jwwangchn.cn",
                "date_created" : "2019"
            }
    
    licenses = [{"id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }]

    # DOTA dataset's information
    image_format='.png'
    anno_format='.txt'
    core_dataset = 'gaofen'

    original_class = {'ship': 1}
    converted_class = [{'supercategory': 'none', 'id': 1,  'name': 'ship', "keypoints": ['top', 'right', 'bottom', 'left'], "skeleton": [[1,2], [2,3], [3,4], [4,1]]}]
    imagesets = ['trainval','test']
    release_version = 'v2'
    rate = '1.0'

    # pointobb sort method
    pointobb_sort_method = 'best' # or "extreme"
    pointobb_sort_function = {"best": pointobb_best_point_sort,
                            "extreme": pointobb_extreme_sort}

    for imageset in imagesets:
        imgpath = '/data/{}/{}/{}/images'.format(core_dataset, release_version, imageset)
        annopath = '/data/{}/{}/{}/labels'.format(core_dataset, release_version, imageset)
        save_path = '/data/{}/{}/coco/annotations'.format(core_dataset, release_version)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        gaofen = GaoFen2COCO(imgpath=imgpath,
                        annopath=annopath,
                        image_format=image_format,
                        anno_format=anno_format,
                        data_categories=converted_class,
                        data_info=info,
                        data_licenses=licenses,
                        data_type="instances",
                        groundtruth=True,
                        small_object_area=0)

        images, annotations = gaofen.get_image_annotation_pairs()

        json_data = {"info" : gaofen.info,
                    "images" : images,
                    "licenses" : gaofen.licenses,
                    "type" : gaofen.type,
                    "annotations" : annotations,
                    "categories" : gaofen.categories}

        with open(os.path.join(save_path, core_dataset + "_" + imageset + "_" + release_version + "_" + rate + ".json"), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)
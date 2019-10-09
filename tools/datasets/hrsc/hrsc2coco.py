import argparse

import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET

from wwtool.datasets import Convert2COCO
from wwtool.transforms import thetaobb2hobb, pointobb2thetaobb, pointobb2sampleobb, pointobb_extreme_sort, pointobb_best_point_sort, thetaobb2pointobb

class HRSC2COCO(Convert2COCO):
    def __generate_coco_annotation__(self, annotpath, imgpath):
        """
        docstring here
            :param self: 
            :param annotpath: the path of each annotation
            :param return: dict()  
        """
        objects = self.__hrsc_parse__(annotpath, imgpath)
        
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

            if area < self.small_object_area and self.groundtruth:
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
    
    def __hrsc_parse__(self, hrsc_label_file, hrsc_image_file):
        objects = []
        if self.groundtruth:
            tree = ET.parse(hrsc_label_file)
            root = tree.getroot()
            objects = []
            hrsc_object = root.find('HRSC_Objects')
            for hrsc_sub_object in hrsc_object.findall('HRSC_Object'):
                obj_struct = {}
                xmin = float(hrsc_sub_object.find('box_xmin').text)
                ymin = float(hrsc_sub_object.find('box_ymin').text)
                xmax = float(hrsc_sub_object.find('box_xmax').text)
                ymax = float(hrsc_sub_object.find('box_ymax').text)
                bbox_w = xmax - xmin
                bbox_h = ymax - ymin

                cx = float(hrsc_sub_object.find('mbox_cx').text)
                cy = float(hrsc_sub_object.find('mbox_cy').text)
                rbbox_w = float(hrsc_sub_object.find('mbox_w').text)
                rbbox_h = float(hrsc_sub_object.find('mbox_h').text)
                angle = float(hrsc_sub_object.find('mbox_ang').text)
                # angle = angle * 180.0 / np.pi

                obj_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
                obj_struct['thetaobb'] = [cx, cy, rbbox_w, rbbox_h, angle]
                obj_struct['segmentation'] = thetaobb2pointobb(obj_struct['thetaobb'])
                obj_struct['pointobb'] = pointobb_sort_function[pointobb_sort_method](obj_struct['segmentation'])
                obj_struct['keypoints'] = obj_struct['pointobb'][:]
                for idx in [2, 5, 8, 11]:
                    obj_struct['keypoints'].insert(idx, 2)
                obj_struct['hobb'] = thetaobb2hobb(obj_struct['thetaobb'], pointobb_sort_function[pointobb_sort_method])
                obj_struct['label'] = 1

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
    parser = argparse.ArgumentParser(description='hrsc dataset to coco dataset')
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
            "description" : "HRSC-COCO",
            "contributor" : "Jinwang Wang",
            "url" : "jwwangchn.cn",
            "date_created" : "2019"
            }
    
    licenses = [{"id": 1,
                 "name": "Attribution-NonCommercial",
                 "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }]

    image_format='.bmp'
    anno_format='.xml'

    original_hrsc_class = {'ship': 1}

    converted_hrsc_class = [{'supercategory': 'none', 'id': 1,  'name': 'ship',                },]

    imagesets = ['trainval', 'test']
    release_version = 'v2'
    rate = '1.0'
    groundtruth = True
    keypoint = True

    extra_info = ''
    if keypoint:
        for idx in range(len(converted_hrsc_class)):
            converted_hrsc_class[idx]["keypoints"] = ['top', 'right', 'bottom', 'left']
            converted_hrsc_class[idx]["skeleton"] = [[1,2], [2,3], [3,4], [4,1]]
        extra_info += 'keypoint'

    # pointobb sort method
    pointobb_sort_method = 'best' # or "extreme"
    pointobb_sort_function = {"best": pointobb_best_point_sort,
                            "extreme": pointobb_extreme_sort}

    for imageset in imagesets:
        imgpath = './data/hrsc/{}/{}/images'.format(release_version, imageset)
        annopath = './data/hrsc/{}/{}/annotations'.format(release_version, imageset)
        save_path = './data/hrsc/{}/coco/annotations'.format(release_version)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        hrsc = HRSC2COCO(imgpath=imgpath,
                        annopath=annopath,
                        image_format=image_format,
                        anno_format=anno_format,
                        data_categories=converted_hrsc_class,
                        data_info=info,
                        data_licenses=licenses,
                        data_type="instances",
                        groundtruth=groundtruth,
                        small_object_area=0)

        images, annotations = hrsc.get_image_annotation_pairs()

        json_data = {"info" : hrsc.info,
                    "images" : images,
                    "licenses" : hrsc.licenses,
                    "type" : hrsc.type,
                    "annotations" : annotations,
                    "categories" : hrsc.categories}

        with open(os.path.join(save_path, "hrsc_" + imageset + "_" + release_version + "_" + rate + "_" + pointobb_sort_method + "_" + extra_info + ".json"), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)

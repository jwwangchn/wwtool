import argparse

import os
import cv2
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET

import wwtool
from wwtool.datasets import Convert2COCO

class DIOR2COCO(Convert2COCO):
    def __generate_coco_annotation__(self, annotpath, imgpath):
        """
        docstring here
            :param self: 
            :param annotpath: the path of each annotation
            :param return: dict()  
        """
        objects = self.__dior_parse__(annotpath, imgpath)
        
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
    
    def __dior_parse__(self, label_file, image_file):
        """
        (xmin, ymin, xmax, ymax)
        """
        objects = []
        tree = ET.parse(label_file)
        root = tree.getroot()
        total_object_num = len(root.findall('object'))
        small_object_num = 0
        large_object_num = 0
        total_object_num = 0
        for single_object in root.findall('object'):
            bndbox = single_object.find('bndbox')
            object_struct = {}

            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            total_object_num += 1
            if bbox_h * bbox_w <= small_size:
                small_object_num += 1
            if bbox_h * bbox_w >= large_object_size:
                large_object_num += 1

            object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
            object_struct['segmentation'] = wwtool.bbox2pointobb([xmin, ymin, xmax, ymax])
            object_struct['label'] = original_dior_class[single_object.find('name').text]
            
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
                "description" : "DIOR-COCO",
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
    anno_format='.xml'

    original_dior_class = {'airplane':                      1, 
                            'airport':                      2, 
                            'baseballfield':                3, 
                            'basketballcourt':              4, 
                            'bridge':                       5, 
                            'chimney':                      6, 
                            'dam':                          7, 
                            'Expressway-Service-area':      8, 
                            'Expressway-toll-station':      9, 
                            'golffield':                    10, 
                            'groundtrackfield':             11, 
                            'harbor':                       12, 
                            'overpass':                     13, 
                            'ship':                         14, 
                            'stadium':                      15, 
                            'storagetank':                  16, 
                            'tenniscourt':                  17, 
                            'trainstation':                 18, 
                            'vehicle':                      19, 
                            'windmill':                     20}

    converted_dior_class = [{'supercategory': 'none', 'id': 1,  'name': 'airplane',                 },
                            {'supercategory': 'none', 'id': 2,  'name': 'airport',                  },
                            {'supercategory': 'none', 'id': 3,  'name': 'baseball field',           },
                            {'supercategory': 'none', 'id': 4,  'name': 'basketball court',         },
                            {'supercategory': 'none', 'id': 5,  'name': 'bridge',                   },
                            {'supercategory': 'none', 'id': 6,  'name': 'chimney',                  },
                            {'supercategory': 'none', 'id': 7,  'name': 'dam',                      },
                            {'supercategory': 'none', 'id': 8,  'name': 'expressway service area',  },
                            {'supercategory': 'none', 'id': 9,  'name': 'expressway toll station',  },
                            {'supercategory': 'none', 'id': 10,  'name': 'golf course',             },
                            {'supercategory': 'none', 'id': 11,  'name': 'ground track field',      },
                            {'supercategory': 'none', 'id': 12,  'name': 'harbor',                  },
                            {'supercategory': 'none', 'id': 13,  'name': 'overpass',                },
                            {'supercategory': 'none', 'id': 14,  'name': 'ship',                    },
                            {'supercategory': 'none', 'id': 15,  'name': 'stadium',                 },
                            {'supercategory': 'none', 'id': 16,  'name': 'storage tank',            },
                            {'supercategory': 'none', 'id': 17,  'name': 'tennis court',            },
                            {'supercategory': 'none', 'id': 18,  'name': 'train station',           },
                            {'supercategory': 'none', 'id': 19,  'name': 'vehicle',                 },
                            {'supercategory': 'none', 'id': 20,  'name': 'wind mill',               }]

    core_dataset_name = 'dior'
    imagesets = ['trainval_test']
    release_version = 'v1'
    rate = '1.0'
    groundtruth = True
    keypoint = False
    
    just_keep_small = False
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
        for idx in range(len(converted_dior_class)):
            converted_dior_class[idx]["keypoints"] = ['top', 'right', 'bottom', 'left']
            converted_dior_class[idx]["skeleton"] = [[1,2], [2,3], [3,4], [4,1]]
        anno_name.append('keypoint')
    
    if groundtruth == False:
        anno_name.append('no_ground_truth')

    for imageset in imagesets:
        imgpath = './data/{}/{}/{}/images'.format(core_dataset_name, release_version, imageset)
        annopath = './data/{}/{}/{}/labels'.format(core_dataset_name, release_version, imageset)
        save_path = './data/{}/{}/coco/annotations'.format(core_dataset_name, release_version)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        dior = DIOR2COCO(imgpath=imgpath,
                        annopath=annopath,
                        image_format=image_format,
                        anno_format=anno_format,
                        data_categories=converted_dior_class,
                        data_info=info,
                        data_licenses=licenses,
                        data_type="instances",
                        groundtruth=groundtruth,
                        small_object_area=0)

        images, annotations = dior.get_image_annotation_pairs()

        json_data = {"info" : dior.info,
                    "images" : images,
                    "licenses" : dior.licenses,
                    "type" : dior.type,
                    "annotations" : annotations,
                    "categories" : dior.categories}

        anno_name.insert(1, imageset)
        with open(os.path.join(save_path, "_".join(anno_name) + ".json"), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)
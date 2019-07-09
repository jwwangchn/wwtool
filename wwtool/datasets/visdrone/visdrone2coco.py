import argparse

import os
import cv2
import json
import numpy as np

from wwtool.datasets.convert2coco import Convert2COCO

class VisDrone2COCO(Convert2COCO):
    def __generate_coco_annotation__(self, annotpath, imgpath):
        """
        docstring here
            :param self: 
            :param annotpath: the path of each annotation
            :param return: dict()  
        """
        objects = self.__visdrone_parse__(annotpath, imgpath)

        coco_annotations = []
        
        for object_struct in objects:
            bbox = object_struct['bbox']
            label = object_struct['label']
            segmentation = object_struct['segmentation']

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

            coco_annotations.append(coco_annotation)
            
        return coco_annotations
    
    def __visdrone_parse__(self, label_file, image_file):
        """
        <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>, <object_category>, <truncation>, <occlusion>
        """
        objects = []
        if self.groundtruth:
            labels = open(label_file, 'r').readlines()
            for label in labels:
                obj_struct = {}

                obj_struct['bbox'] = [float(xy) for xy in label.split(',')[:4]]

                xmin, ymin, bbox_w, bbox_h = obj_struct['bbox']

                obj_struct['segmentation'] = [xmin, ymin, xmin + bbox_w, ymin, xmin + bbox_w, ymin + bbox_h, xmin, ymin + bbox_h]
                
                obj_struct['label'] = int(label.split(',')[5].strip('\n'))
                
                if obj_struct['label'] == 0 or obj_struct['label'] == 11:
                    continue

                objects.append(obj_struct)
        else:
            obj_struct = {}
            obj_struct['segmentation'] = [0, 0, 0, 0, 0, 0, 0, 0]
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
        choices=['train', 'val', 'test'])
    parser.add_argument(
        '--release_version', default='v1', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # basic dataset information
    info = {"year" : 2019,
            "version" : "1.0",
            "description" : "visdrone-COCO",
            "contributor" : "Jinwang Wang",
            "url" : "jwwangchn.cn",
            "date_created" : "2019"
            }
    
    licenses = [{"id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }]

    # DOTA dataset's information
    image_format='.jpg'
    anno_format='.txt'

    visdrone_class = {'pedestrian': 1, 'people': 2, 'bicycle': 3, 'car': 4, 'van': 5, 'truck': 6, 'tricycle': 7, 'awning-tricycle': 8, 'bus': 9, 'motor': 10}

    coco_class = [{'supercategory': 'none', 'id': 1,  'name': 'pedestrian',                   },
                {'supercategory': 'none', 'id': 2,  'name': 'people',                       }, 
                {'supercategory': 'none', 'id': 3,  'name': 'bicycle',                      },
                {'supercategory': 'none', 'id': 4,  'name': 'car',                          },
                {'supercategory': 'none', 'id': 5,  'name': 'van',                          },
                {'supercategory': 'none', 'id': 6,  'name': 'truck',                        },
                {'supercategory': 'none', 'id': 7,  'name': 'tricycle',                     },
                {'supercategory': 'none', 'id': 8,  'name': 'awning-tricycle',              },
                {'supercategory': 'none', 'id': 9,  'name': 'bus',                          },
                {'supercategory': 'none', 'id': 10, 'name': 'motor',                        }]

    imagesets = ['train', 'val', 'test']
    release_version = 'v1'
    core_dataset = 'visdrone'
    groundtruth = True

    for imageset in imagesets:
        
        imgpath = '/media/jwwangchn/data/{}/{}/{}/images'.format(core_dataset, release_version, imageset)
        annopath = '/media/jwwangchn/data/{}/{}/{}/annotations'.format(core_dataset, release_version, imageset)
        save_path = '/media/jwwangchn/data/{}/{}/coco/annotations'.format(core_dataset, release_version)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if imageset == 'test':
            groundtruth = False

        visdrone = VisDrone2COCO(imgpath=imgpath,
                        annopath=annopath,
                        image_format=image_format,
                        anno_format=anno_format,
                        data_categories=coco_class,
                        data_info=info,
                        data_licenses=licenses,
                        data_type="instances",
                        groundtruth=groundtruth,
                        small_object_area=0)

        images, annotations = visdrone.get_image_annotation_pairs()

        json_data = {"info" : visdrone.info,
                    "images" : images,
                    "licenses" : visdrone.licenses,
                    "type" : visdrone.type,
                    "annotations" : annotations,
                    "categories" : visdrone.categories}

        with open(os.path.join(save_path, "visdrone_" + imageset + "_" + release_version + ".json"), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)

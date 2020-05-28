import argparse

import os
import cv2
import json
import csv
import shutil
import numpy as np

import wwtool

import os
import cv2
import mmcv

class SIMPLETXT2COCO():
    def __init__(self, 
                imgpath=None,
                annopath=None,
                imageset_file=None,
                image_format='.jpg',
                anno_format='.txt',
                data_categories=None,
                data_info=None,
                data_licenses=None,
                data_type="instances",
                groundtruth=True,
                small_object_area=0,
                sub_anno_fold=False,
                cities=None):
        super(SIMPLETXT2COCO, self).__init__()

        self.imgpath = imgpath
        self.annopath = annopath
        self.image_format = image_format
        self.anno_format = anno_format

        self.categories = data_categories
        self.info = data_info
        self.licenses = data_licenses
        self.type = data_type
        self.small_object_area = small_object_area
        self.small_object_idx = 0
        self.groundtruth = groundtruth
        self.max_object_num_per_image = 0
        self.sub_anno_fold = sub_anno_fold
        self.imageset_file = imageset_file

        self.imgpaths, self.annotpaths = [], []

        for city in cities:
            for image_fn in os.listdir(os.path.join(self.imgpath, city, 'images')):
                basename = wwtool.get_basename(image_fn)
                self.imgpaths.append(os.path.join(self.imgpath, city, 'images', basename + '.png'))
                self.annotpaths.append(os.path.join(self.imgpath, city, 'labels', basename + '.txt'))
                
    def get_image_annotation_pairs(self):
        images = []
        annotations = []
        index = 0
        progress_bar = mmcv.ProgressBar(len(self.imgpaths))
        imId = 0
        for imgfile, annofile in zip(self.imgpaths, self.annotpaths):
            # imgpath = os.path.join(self.imgpath, name + self.image_format)
            # annotpath = os.path.join(self.annopath, name + self.anno_format)
            name = wwtool.get_basename(imgfile)

            annotations_coco = self.__generate_coco_annotation__(annofile, imgfile)

            # if annotation is empty, skip this annotation
            if annotations_coco != [] or self.groundtruth == False:
                img = cv2.imread(imgfile)
                height, width, channels = img.shape
                images.append({"date_captured": "2019",
                                "file_name": name + self.image_format,
                                "id": imId + 1,
                                "license": 1,
                                "url": "http://jwwangchn.cn",
                                "height": height,
                                "width": width})

                for annotation in annotations_coco:
                    index = index + 1
                    annotation["iscrowd"] = 0
                    annotation["image_id"] = imId + 1
                    annotation["id"] = index
                    annotations.append(annotation)

                imId += 1

            if imId % 500 == 0:
                print("\nImage ID: {}, Instance ID: {}, Small Object Counter: {}, Max Object Number: {}".format(imId, index, self.small_object_idx, self.max_object_num_per_image))
            
            progress_bar.update()
            
        return images, annotations

    def __generate_coco_annotation__(self, annotpath, imgpath):
        """
        docstring here
            :param self: 
            :param annotpath: the path of each annotation
            :param return: dict()  
        """
        objects = self.__simpletxt_parse__(annotpath, imgpath)
        
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
    
    def __simpletxt_parse__(self, label_file, image_file):
        """
        (xmin, ymin, xmax, ymax)
        """
        with open(label_file, 'r') as f:
            lines = f.readlines()
    
        objects = []
        total_object_num = 0

        for line in lines:
            object_struct = {}
            line = line.rstrip().split(' ')
            label = " ".join(line[-1])
            mask = [float(_) for _ in line[0:-1]]

            xmin, ymin, xmax, ymax = wwtool.pointobb2bbox(mask)
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            total_object_num += 1

            object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
            object_struct['segmentation'] = mask
            object_struct['label'] = 1
            
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
                "description" : "SIMPLETXT-Building-COCO",
                "contributor" : "Jinwang Wang",
                "url" : "jwwangchn.cn",
                "date_created" : "2019"
            }
    
    licenses = [{"id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }]

    original_simpletxt_class = {'building': 1}

    converted_simpletxt_class = [{'supercategory': 'none', 'id': 1,  'name': 'building',                   }]

    # dataset's information
    image_format='.png'
    anno_format='.txt'

    core_dataset_name = 'buildchange'
    # cities = ['shanghai']
    # sub_city_folds = {'shanghai': ['arg']}
    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']

    release_version = 'v2'
    groundtruth = True

    anno_name = [core_dataset_name, release_version, 'trainval']
    
    imgpath = f'./data/{core_dataset_name}/{release_version}'
    annopath = f'./data/{core_dataset_name}/{release_version}'
    save_path = f'./data/{core_dataset_name}/{release_version}/coco/annotations'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    simpletxt2coco = SIMPLETXT2COCO(imgpath=imgpath,
                                    annopath=annopath,
                                    image_format=image_format,
                                    anno_format=anno_format,
                                    data_categories=converted_simpletxt_class,
                                    data_info=info,
                                    data_licenses=licenses,
                                    data_type="instances",
                                    groundtruth=groundtruth,
                                    small_object_area=0,
                                    cities=cities)

    images, annotations = simpletxt2coco.get_image_annotation_pairs()

    json_data = {"info" : simpletxt2coco.info,
                "images" : images,
                "licenses" : simpletxt2coco.licenses,
                "type" : simpletxt2coco.type,
                "annotations" : annotations,
                "categories" : simpletxt2coco.categories}

    with open(os.path.join(save_path, "_".join(anno_name) + ".json"), "w") as jsonfile:
        json.dump(json_data, jsonfile, sort_keys=True, indent=4)
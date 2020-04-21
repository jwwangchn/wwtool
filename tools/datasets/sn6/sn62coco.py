import argparse

import os
import cv2
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET

import wwtool
from wwtool.datasets import Convert2COCO

class SN62COCO(Convert2COCO):
    def __generate_coco_annotation__(self, annotpath, imgpath):
        """
        docstring here
            :param self: 
            :param annotpath: the path of each annotation
            :param return: dict()  
        """
        objects = self.__sn6_parse__(annotpath, imgpath)
        
        coco_annotations = []

        if generate_small_dataset and len(objects) > 0:
            pass
            # wwtool.generate_same_dataset(imgpath, 
            #                             annotpath,
            #                             dst_image_path,
            #                             dst_label_path,
            #                             src_img_format='.png',
            #                             src_anno_format='.txt',
            #                             dst_img_format='.png',
            #                             dst_anno_format='.txt',
            #                             parse_fun=wwtool.voc_parse,
            #                             dump_fun=wwtool.simpletxt_dump)

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
    
    def __sn6_parse__(self, label_file, image_file):
        """
        (xmin, ymin, xmax, ymax)
        """
        objects = []
        if self.groundtruth:
            image_file_name = os.path.splitext(os.path.basename(image_file))[0]
            save_image = wwtool.generate_image(800, 800, color=(0, 0, 0))
            img = cv2.imread(image_file)
            img_height, img_width, _ = img.shape

            image_name = os.path.basename(image_file).split('SN6_{}_AOI_11_Rotterdam_{}_'.format(imageset_file_name, data_source))[1].split('.tif')[0]
            masks = sn6_parse.sn6_parse(image_name)
            objects_small_save = []
            total_object_num = len(masks)
            small_object_num = 0
            large_object_num = 0
            total_object_num = 0
            for mask in masks:
                object_struct = {}
                object_struct_small = {}
                xmin, ymin, xmax, ymax = wwtool.pointobb2bbox(mask['segmentation'])
                bbox_w = xmax - xmin
                bbox_h = ymax - ymin

                if bbox_w * bbox_h <= self.small_object_area:
                    continue

                total_object_num += 1
                if bbox_h * bbox_w <= small_size:
                    small_object_num += 1
                if bbox_h * bbox_w >= large_object_size:
                    large_object_num += 1

                object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
                object_struct['segmentation'] = mask['segmentation']
                object_struct['label'] = 1

                object_struct_small = object_struct.copy()
                object_struct_small['bbox'] = [xmin, ymin, xmax, ymax]
                object_struct_small['label'] = 'building'
                
                objects_small_save.append(object_struct_small)
                objects.append(object_struct)

            if total_object_num > self.max_object_num_per_image:
                    self.max_object_num_per_image = total_object_num

            if just_keep_small or generate_small_dataset:
                if small_object_num >= total_object_num * small_object_rate and large_object_num < 1 and len(objects_small_save) > 0:
                    # print(os.path.join(dst_image_path, image_file_name + '.png'))
                    save_image[0:img_height, 0:img_width, :] = img
                    cv2.imwrite(os.path.join(dst_image_path, image_file_name + '.png'), save_image)
                    anno_file = os.path.join(dst_label_path, image_file_name + '.txt')
                    wwtool.simpletxt_dump(objects_small_save, anno_file)
                    return objects
                else:
                    return []
            else:
                return objects
        else:
            obj_struct = {}
            obj_struct['segmentation'] = [0, 0, 0, 0, 0, 0, 0, 0]
            obj_struct['bbox'] = [0, 0, 0, 0]
            obj_struct['label'] = 0

            objects.append(obj_struct)

            return objects

if __name__ == "__main__":
    # basic dataset information
    info = {"year" : 2019,
                "version" : "1.0",
                "description" : "SN6-COCO",
                "contributor" : "Jinwang Wang",
                "url" : "jwwangchn.cn",
                "date_created" : "2019"
            }
    
    licenses = [{"id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }]

    # dataset's information
    image_format='.tif'
    anno_format='.csv'

    data_source = 'SAR-Intensity'  # SAR-Intensity or PS-RGB
    if data_source == 'SAR-Intensity':
        image_fold = 'SAR'
    else:
        image_fold = 'RGB'

    original_sn6_class = {'building' : 1}

    converted_sn6_class = [{'supercategory': 'none', 'id': 1,  'name': 'building'}]

    core_dataset_name = 'sn6'
    imagesets = ['train']
    if imagesets[0] == 'train':
        imageset_file_name = 'Train'
    else:
        imageset_file_name = 'Test_Public'

    release_version = 'v1'
    rate = '1.0'
    groundtruth = True
    keypoint = False
    
    just_keep_small = False
    generate_small_dataset = False
    small_size = 16 * 16
    small_object_rate = 0.5
    large_object_size = 64 * 64

    anno_name = [core_dataset_name, release_version, data_source]
    if just_keep_small:
        anno_name.append('small_object')

    if generate_small_dataset:
        dst_image_path = '/data/small/{}/{}'.format(core_dataset_name, image_fold)
        dst_label_path = '/data/small/{}/labels'.format(core_dataset_name)
        wwtool.mkdir_or_exist(dst_image_path)
        wwtool.mkdir_or_exist(dst_label_path)

    if keypoint:
        for idx in range(len(converted_sn6_class)):
            converted_sn6_class[idx]["keypoints"] = ['top', 'right', 'bottom', 'left']
            converted_sn6_class[idx]["skeleton"] = [[1,2], [2,3], [3,4], [4,1]]
        anno_name.append('keypoint')
    
    if groundtruth == False:
        anno_name.append('no_ground_truth')

    for imageset in imagesets:
        imgpath = './data/{}/{}/{}/{}'.format(core_dataset_name, release_version, imageset, image_fold)
        annopath = './data/{}/{}/{}/labels'.format(core_dataset_name, release_version, imageset)
        save_path = './data/{}/{}/coco/annotations'.format(core_dataset_name, release_version)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if imageset == 'train':
            anno_file = os.path.join(annopath, 'SN6_Train_AOI_11_Rotterdam_Buildings.csv')
            sn6_parse = wwtool.SN6Parse(anno_file)

        sn6 = SN62COCO(imgpath=imgpath,
                        annopath=annopath,
                        image_format=image_format,
                        anno_format=anno_format,
                        data_categories=converted_sn6_class,
                        data_info=info,
                        data_licenses=licenses,
                        data_type="instances",
                        groundtruth=groundtruth,
                        small_object_area=9)

        images, annotations = sn6.get_image_annotation_pairs()

        json_data = {"info" : sn6.info,
                    "images" : images,
                    "licenses" : sn6.licenses,
                    "type" : sn6.type,
                    "annotations" : annotations,
                    "categories" : sn6.categories}

        anno_name.insert(1, imageset)
        with open(os.path.join(save_path, "_".join(anno_name) + ".json"), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)

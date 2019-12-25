import os
import numpy as np
import cv2
from PIL import Image
from skimage.io import imread

import wwtool

Image.MAX_IMAGE_PIXELS = int(2048 * 2048 * 2048 // 4 // 3)

if __name__ == '__main__':
    core_dataset_name = 'stanford_campus'
    label_fold = './data/{}/v0/annotations'.format(core_dataset_name)
    video_fold = './data/{}/v0/videos'.format(core_dataset_name)
    stanford_compus_parse = wwtool.StanfordCompusParse(label_fold)

    image_set = 'trainval_test'

    subimage_size = 800
    gap = 200

    scenes = ('bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'little', 'nexus', 'quad')
    
    image_save_path = './data/{}/v1/{}/images'.format(core_dataset_name, image_set)
    wwtool.mkdir_or_exist(image_save_path)
    label_save_path = './data/{}/v1/{}/labels'.format(core_dataset_name, image_set)
    wwtool.mkdir_or_exist(label_save_path)

    for scene_name in scenes:
        for video_name in os.listdir(os.path.join(video_fold, scene_name)):
            videocapture = cv2.VideoCapture(os.path.join(video_fold, scene_name, video_name, 'video.mov'))

            frame_id = 0
            while(videocapture.isOpened()):
                ret, img = videocapture.read()
                if ret == False:
                    break
                image_name = "_".join([scene_name, video_name, "{:0>6d}".format(frame_id)]) + '.png'
                if frame_id % 120 != 0:
                    frame_id += 1
                    continue
                print(scene_name, video_name, frame_id, image_name)

                objects = stanford_compus_parse.stanford_compus_parse(scene_name, video_name, frame_id)
                if objects == []:
                    frame_id += 1
                    continue
                
                bboxes = np.array([wwtool.xyxy2cxcywh(obj['bbox']) for obj in objects])
                labels = np.array([obj['label'] for obj in objects])

                subimages = wwtool.split_image(img, subsize=subimage_size, gap=gap, expand_boundary=True)
                subimage_coordinates = list(subimages.keys())
                bboxes_ = bboxes.copy()
                labels_ = labels.copy()
                if bboxes_.shape[0] == 0:
                    frame_id += 1
                    continue

                for subimage_coordinate in subimage_coordinates:
                    objects = []
                    
                    bboxes_[:, 0] = bboxes[:, 0] - subimage_coordinate[0]
                    bboxes_[:, 1] = bboxes[:, 1] - subimage_coordinate[1]
                    cx_bool = np.logical_and(bboxes_[:, 0] >= 0, bboxes_[:, 0] < subimage_size)
                    cy_bool = np.logical_and(bboxes_[:, 1] >= 0, bboxes_[:, 1] < subimage_size)
                    subimage_bboxes = bboxes_[np.logical_and(cx_bool, cy_bool)]
                    subimage_labels = labels_[np.logical_and(cx_bool, cy_bool)]
                    
                    if len(subimage_bboxes) == 0:
                        frame_id += 1
                        continue
                    img = subimages[subimage_coordinate]
                    if np.mean(img) == 0:
                        frame_id += 1
                        continue

                    label_save_file = os.path.join(label_save_path, '{}_{}_{}__{}_{}.txt'.format(scene_name, video_name, frame_id, subimage_coordinate[0], subimage_coordinate[1]))
                    image_save_file = os.path.join(image_save_path, '{}_{}_{}__{}_{}.png'.format(scene_name, video_name, frame_id, subimage_coordinate[0], subimage_coordinate[1]))
                    cv2.imwrite(image_save_file, img)
                    
                    for subimage_bbox, subimage_label in zip(subimage_bboxes, subimage_labels):
                        subimage_objects = dict()
                        subimage_objects['bbox'] = wwtool.cxcywh2xyxy(subimage_bbox.tolist())
                        subimage_objects['label'] = subimage_label
                        objects.append(subimage_objects)
                    wwtool.simpletxt_dump(objects, label_save_file)
                frame_id += 1

            videocapture.release()
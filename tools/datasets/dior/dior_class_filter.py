import os
import cv2
import numpy as np
import wwtool
import mmcv

convert_classes = { 'airplane':                     'airplane', 
                    'airport':                      None, 
                    'baseballfield':                None, 
                    'basketballcourt':              None, 
                    'bridge':                       'bridge', 
                    'chimney':                      None, 
                    'dam':                          None, 
                    'Expressway-Service-area':      None, 
                    'Expressway-toll-station':      None, 
                    'golffield':                    None, 
                    'groundtrackfield':             None, 
                    'harbor':                       None, 
                    'overpass':                     None, 
                    'ship':                         'ship', 
                    'stadium':                      None, 
                    'storagetank':                  'storage-tank', 
                    'tenniscourt':                  None, 
                    'trainstation':                 None, 
                    'vehicle':                      'vehicle', 
                    'windmill':                     'wind-mill'}

if __name__ == "__main__":
    image_format = '.jpg'

    origin_image_path = './data/dior/v1/trainval_test/images'
    origin_label_path = './data/dior/v1/trainval_test/labels'

    filtered_image_path = './data/dior/v1/trainval_test_filtered/images'
    filtered_label_path = './data/dior/v1/trainval_test_filtered/labels'

    wwtool.mkdir_or_exist(filtered_image_path)
    wwtool.mkdir_or_exist(filtered_label_path)

    filter_count = 1
    progress_bar = mmcv.ProgressBar(len(os.listdir(origin_label_path)))
    for label_name in os.listdir(origin_label_path):
        image_objects = wwtool.voc_parse(os.path.join(origin_label_path, label_name))
        filtered_objects = []
        for image_object in image_objects:
            if convert_classes[image_object['label']] == None:
                filter_count += 1
                continue
            else:
                image_object['label'] = convert_classes[image_object['label']]
                filtered_objects.append(image_object)

        if len(filtered_objects) > 0:
            img = cv2.imread(os.path.join(origin_image_path, os.path.splitext(label_name)[0] + image_format))
            save_image_file = os.path.join(filtered_image_path, os.path.splitext(label_name)[0] + '.png')
            # print("Save image file: ", save_image_file)
            cv2.imwrite(save_image_file, img)
            wwtool.simpletxt_dump(filtered_objects, os.path.join(filtered_label_path, os.path.splitext(label_name)[0] + '.txt'))
        
        progress_bar.update()

    print("Filter object counter: {}".format(filter_count))
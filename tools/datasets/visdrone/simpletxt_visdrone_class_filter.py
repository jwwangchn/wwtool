import os
import cv2
import numpy as np
import wwtool
import mmcv

convert_classes = { 'pedestrian':       'person', 
                    'people':           'person', 
                    'bicycle':          None, 
                    'car':              'vehicle', 
                    'van':              'vehicle', 
                    'truck':            'vehicle', 
                    'tricycle':         'vehicle', 
                    'awning-tricycle':  'vehicle', 
                    'bus':              'vehicle', 
                    'motor':            None}
origin_class = {'1':'pedestrian', '2':'people', '3':'bicycle', '4':'car', '5':'van', '6':'truck', '7':'tricycle', '8':'awning-tricycle', '9':'bus', '10':'motor'}


if __name__ == "__main__":
    image_format = '.png'

    origin_image_path = './data/visdrone/v1/trainval/images'
    origin_label_path = './data/visdrone/v1/trainval/labels'

    filtered_image_path = './data/visdrone/v1/trainval_filtered/images'
    filtered_label_path = './data/visdrone/v1/trainval_filtered/labels'

    wwtool.mkdir_or_exist(filtered_image_path)
    wwtool.mkdir_or_exist(filtered_label_path)

    filter_count = 1
    progress_bar = mmcv.ProgressBar(len(os.listdir(origin_label_path)))
    for label_name in os.listdir(origin_label_path):
        image_objects = wwtool.simpletxt_parse(os.path.join(origin_label_path, label_name))
        filtered_objects = []
        for image_object in image_objects:
            if convert_classes[origin_class[image_object['label']]] == None:
                filter_count += 1
                continue
            else:
                image_object['label'] = convert_classes[origin_class[image_object['label']]]
                filtered_objects.append(image_object)

        if len(filtered_objects) > 0:
            img = cv2.imread(os.path.join(origin_image_path, os.path.splitext(label_name)[0] + image_format))
            save_image_file = os.path.join(filtered_image_path, os.path.splitext(label_name)[0] + '.png')
            # print("Save image file: ", save_image_file)
            cv2.imwrite(save_image_file, img)
            wwtool.simpletxt_dump(filtered_objects, os.path.join(filtered_label_path, os.path.splitext(label_name)[0] + '.txt'))
        
        progress_bar.update()

    print("Filter object counter: {}".format(filter_count))
import os
import cv2
import numpy as np
import wwtool
import mmcv

convert_classes = { 'airplane':             'airplane', 
                    'ship':                 'ship', 
                    'storage-tank':         'storage-tank', 
                    'baseball-diamond':     None, 
                    'tennis-court':         'tennis-court', 
                    'basketball-court':     None, 
                    'ground-track-field':   None, 
                    'harbor':               'harbor', 
                    'bridge':               'bridge', 
                    'vehicle':              'vehicle'}
origin_class = {'1':'airplane', '2':'ship', '3':'storage-tank', '4':'baseball-diamond', '5':'tennis-court', '6':'basketball-court', '7':'ground-track-field', '8':'harbor', '9':'bridge', '10':'vehicle'}


if __name__ == "__main__":
    image_format = '.png'

    dataset = 'nwpu'
    origin_image_path = './data/{}/v1/trainval_test/images'.format(dataset)
    origin_label_path = './data/{}/v1/trainval_test/labels'.format(dataset)

    filtered_image_path = './data/{}/v1/trainval_test_filtered/images'.format(dataset)
    filtered_label_path = './data/{}/v1/trainval_test_filtered/labels'.format(dataset)

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
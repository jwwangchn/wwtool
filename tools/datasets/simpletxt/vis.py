import os
import wwtool

label_fold = './data/stanford_campus/v1/trainval_test/labels'
image_fold = './data/stanford_campus/v1/trainval_test/images'

for label_name in sorted(os.listdir(label_fold)):
    print(label_name)
    label_file = os.path.join(label_fold, label_name)
    image_file = os.path.join(image_fold, label_name.split('.')[0] + '.png')
    objects = wwtool.simpletxt_parse(label_file)
    bbox, label = [single_object['bbox'] for single_object in objects], [single_object['label'] for single_object in objects]
    wwtool.imshow_bboxes(image_file, bbox)
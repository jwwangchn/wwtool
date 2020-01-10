import os
import wwtool

label_fold = '/home/jwwangchn/Documents/100-Work/170-Codes/wwtool/data/small/airbus-ship/labels'
image_fold = '/home/jwwangchn/Documents/100-Work/170-Codes/wwtool/data/small/airbus-ship/images'

for label_name in sorted(os.listdir(label_fold)):
    print(label_name)
    label_file = os.path.join(label_fold, label_name)
    image_file = os.path.join(image_fold, os.path.splitext(label_name)[0] + '.png')
    objects = wwtool.simpletxt_parse(label_file)
    bbox, label = [single_object['bbox'] for single_object in objects], [single_object['label'] for single_object in objects]
    wwtool.imshow_bboxes(image_file, bbox)
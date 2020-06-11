import os
import numpy as np


if __name__ == '__main__':
    txt_dir = '/data/dota-v1.0/v0/trainval/labelTxt-v1.0'

    count_classes = {'harbor':0, 'ship':0, 'storage-tank':0, 'plane':0, 'baseball-diamond':0, 'helicopter':0, 'roundabout': 0}

    total_num = 0
    for dota_label_file in os.listdir(txt_dir):
        with open(os.path.join(txt_dir, dota_label_file), 'r') as f:
            dota_labels = f.readlines()[2:]

        for dota_label in dota_labels:
            label = dota_label.split(' ')[8]
            if label in count_classes.keys():
                count_classes[label] += 1

            total_num += 1
    class_sum = 0
    for key, value in count_classes.items():
        class_sum += value

    print(count_classes, class_sum, total_num, class_sum/total_num)

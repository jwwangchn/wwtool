import os
import numpy as np
import shutil

import wwtool


if __name__ == '__main__':
    # cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    cities = ['shanghai']
    
    label_list = []
    for city in cities:
        for label_fn in os.listdir('./data/buildchange/v2/{}/labels_json'.format(city)):
            label_list.append([city, label_fn])
    
    label_list = sorted(label_list)
    np.random.shuffle(label_list)

    dst_label_dir = '/data/buildchange/v2/sampling/labels_json'
    dst_image_dir = '/data/buildchange/v2/sampling/images'
    wwtool.mkdir_or_exist(dst_label_dir)
    wwtool.mkdir_or_exist(dst_image_dir)

    for label_fn in label_list[0:1000]:
        basename = wwtool.get_basename(label_fn[1])
        src_label_file = './data/buildchange/v2/{}/labels_json/{}'.format(label_fn[0], basename + '.json')
        src_image_file = './data/buildchange/v2/{}/images/{}'.format(label_fn[0], basename + '.png')

        dst_label_file = os.path.join(dst_label_dir, basename + '.json')
        dst_image_file = os.path.join(dst_image_dir, basename + '.png')

        shutil.copy(src_label_file, dst_label_file)
        shutil.copy(src_image_file, dst_image_file)
        
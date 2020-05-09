import os

import wwtool


img_path = '/data/dota-v1.0/v1/trainval/images'
img_list = os.listdir(img_path)
img_list = [os.path.join(img_path, img_name) for img_name in img_list]

dataset_mean, dataset_std = wwtool.img_norm_parameter(img_list)

print(dataset_mean, dataset_std)
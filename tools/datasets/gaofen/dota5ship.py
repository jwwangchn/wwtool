import os
import cv2
import json
import numpy as np
import shutil


label_dirs='/data/dota/v1/trainval/labelTxt-v1.0/'
image_dirs='/data/dota/v1/trainval/images/'
save_label_dirs='/data/gaofen/dota/labeltxt/'
save_image_dirs='/data/gaofen/dota/images/'
label_files=os.listdir(label_dirs)
for label_file in label_files:
    file_name = label_file.split('.txt')[0]
    dota_image_file = os.path.join(image_dirs, file_name + '.png')
    dota_label_file = os.path.join(label_dirs, label_file)
    dota_labels = open(dota_label_file, 'r').readlines()
    num=0
    ship_num=0
    for dota_label in dota_labels:
        num+=1
        if dota_label.split(' ')[8]=='ship':
            ship_num+=1
    if ship_num>0.5*num:
        print(label_file)
        shutil.copy(dota_label_file,save_label_dirs)
        shutil.copy(dota_image_file,save_image_dirs)
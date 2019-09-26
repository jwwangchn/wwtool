import os
import cv2
import shutil
import numpy as np

import mmcv

def copy_image_files(src_path, dst_path, dst_file_format=None):
    img_file_list = os.listdir(src_path)
    progress_bar = mmcv.ProgressBar(len(img_file_list))
    if dst_file_format == None:
        for img_file in img_file_list:
            src_file = os.path.join(src_path, img_file)
            dst_file = os.path.join(dst_path, img_file)
            shutil.copy(src_file, dst_file)
            progress_bar.update()
    else:
        for img_file in img_file_list:
            src_file = os.path.join(src_path, img_file)
            img = cv2.imread(src_file)
            dst_file = os.path.join(dst_path, os.path.splitext(img_file)[0] + '.' + dst_file_format)
            cv2.imwrite(dst_file, img)
            progress_bar.update()
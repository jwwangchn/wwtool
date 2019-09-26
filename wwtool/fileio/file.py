import os
import cv2
import shutil
import numpy as np

import mmcv

def copy_files(src_path, dst_path):
    img_file_list = os.listdir(src_path)
    progress_bar = mmcv.ProgressBar(len(img_file_list))
    for img_file in img_file_list:
        src_file = os.path.join(src_path, img_file)
        dst_file = os.path.join(dst_path, img_file)
        shutil.copy(src_file, dst_file)
        progress_bar.update()
    
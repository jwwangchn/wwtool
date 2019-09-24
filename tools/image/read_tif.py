import numpy as np
import cv2
from wwtool.image import convert_16bit_to_8bit
from wwtool.visualization import show_image


if __name__ == '__main__':
    img_path = '/data/503/GF数据/第一批/GF3_MDJ_FSII_006377_E139.7_N35.4_20171026_L1A_HHHV_L10002710612/GF3_MDJ_FSII_006377_E139.7_N35.4_20171026_L1A_HH_L10002710612.tiff'
    # img = cv2.imread(img_path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    print("image shape: {}".format(img.shape))
    img = convert_16bit_to_8bit(img)
    show_image(img)

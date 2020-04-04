import gdal
import numpy as np
import cv2
import os
import tqdm


src_folder = "./data/sn6/v0/test_public/AOI_11_Rotterdam/SAR-Intensity"
dst_folder = "./data/sn6/v0/test_public/AOI_11_Rotterdam/Processed-SAR-Intensity"
sar_img_list = os.listdir(src_folder)
sar_img_list.sort()

if not os.path.exists(dst_folder):
    os.mkdir(dst_folder)
    
for ii in tqdm.trange(len(sar_img_list)):
    sar_img = sar_img_list[ii]
    img_array = gdal.Open(os.path.join(os.getcwd(), src_folder, sar_img)).ReadAsArray()
    for band in range(4):
        img_array[band] = img_array[band] / img_array[band].max() * 255
    HH = img_array[0]
    HV_VH = 0.5 * (img_array[1] + img_array[2])
    VV = img_array[1]
    out_img = np.dstack((HH, HV_VH, VV))
    out_img = out_img.astype("int")
    cv2.imwrite(os.path.join(os.getcwd(), dst_folder, sar_img), out_img)

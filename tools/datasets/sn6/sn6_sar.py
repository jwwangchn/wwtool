import numpy as np
import cv2
import os
import tqdm
import wwtool
from matplotlib import pyplot as plt
import gdal

src_folder = "./data/sn6/v0/test_public/AOI_11_Rotterdam/SAR-Intensity"
dst_folder = "./data/sn6/v0/test_public/AOI_11_Rotterdam/Processed-SAR-Intensity"
sar_img_list = os.listdir(src_folder)
img_number = len(sar_img_list)
sar_img_list.sort()

if not os.path.exists(dst_folder):
    os.mkdir(dst_folder)

plt.figure()

r = np.zeros((900, 900))
g = np.zeros((900, 900))
b = np.zeros((900, 900))

for ii in tqdm.trange(len(sar_img_list)):
    sar_img = sar_img_list[ii]
    img_array = gdal.Open(os.path.join(os.getcwd(), src_folder, sar_img)).ReadAsArray()
    # img_array = scipy.misc.imread(os.path.join(os.getcwd(), src_folder, sar_img))
    for band in range(4):
        # print(img_array[band].min(), img_array[band].max())
        img_array[band] = img_array[band] / img_array[band].max() * 255
        # wwtool.show_image(img_array[band].astype("uint8"))

        img = img_array[band].astype("uint8")

        img = cv2.equalizeHist(img)
        img_array[band] = img

        hist = cv2.calcHist([img], [0], None , [256], [1, 256])

        plt.plot(hist)
    
    plt.show()

    HH = img_array[0]
    HV_VH = 0.5 * (img_array[1] + img_array[2])
    VV = img_array[1]
    out_img = np.dstack((HH, HV_VH, VV))
    

    r = r + HH
    g = g + HV_VH
    b = b + VV

    out_img = out_img.astype("uint8")

    

    # for idx, test_img in enumerate([HH, HV_VH, VV]):
    #     hist = cv2.calcHist([test_img], [0], None , [256], [1, 256])
    #     print(idx, np.mean(test_img), np.std(test_img))
    #     # plt.plot(hist)
    
    # plt.show()

    # cv2.imwrite(os.path.join(os.getcwd(), dst_folder, sar_img), out_img)
    # wwtool.show_image(out_img)

print(np.mean(r) / img_number, np.std(r) / (img_number))
print(np.mean(g) / img_number, np.std(g) / (img_number))
print(np.mean(b) / img_number, np.std(b) / (img_number))
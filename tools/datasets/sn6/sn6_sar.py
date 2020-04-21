import numpy as np
import cv2
import os
import tqdm
import wwtool
from matplotlib import pyplot as plt
import rasterio

def convert2rgb(sar):
    HH = sar[0]
    HV_VH = 0.5 * (sar[1] + sar[2])
    VV = sar[1]
    ret = np.dstack((HH, HV_VH, VV))

    return ret.astype("uint8")


if __name__ == '__main__':
    src_folder = "./data/sn6/v0/test_public/AOI_11_Rotterdam/SAR-Intensity"
    dst_folder = "./data/sn6/v0/test_public/AOI_11_Rotterdam/Processed-SAR-Intensity"

    sar_img_list = os.listdir(src_folder)
    img_number = len(sar_img_list)
    sar_img_list.sort()

    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    to_rgb = False
    equalize_flag = False
    calculate_mean_std = True

    plt.figure()

    band1 = np.zeros((900, 900), dtype=np.float32)
    band2 = np.zeros((900, 900), dtype=np.float32)
    band3 = np.zeros((900, 900), dtype=np.float32)
    band4 = np.zeros((900, 900), dtype=np.float32)

    for idx in tqdm.trange(len(sar_img_list)):
        sar_img = sar_img_list[idx]
        sar_img_file = os.path.join(os.getcwd(), src_folder, sar_img)

        # input sar image
        with rasterio.open(sar_img_file) as src:
            img_array = src.read()

            for band in range(4):
                img_array[band] = img_array[band] / img_array[band].max() * 255

                if equalize_flag:
                    img = img_array[band].astype("uint8")
                    cv2.equalizeHist(img)
                    img_array[band] = img

                # hist = cv2.calcHist([img], [0], None , [256], [1, 256])
                # plt.plot(hist)

            # plt.show()

        if calculate_mean_std:
            band1 += img_array[0]
            band2 += img_array[1]
            band3 += img_array[2]
            band4 += img_array[3]

        if to_rgb:
            out_img = convert2rgb(img_array)
            cv2.imwrite(os.path.join(os.getcwd(), dst_folder, sar_img), out_img)
            wwtool.show_image(out_img)

    print(np.mean(band1) / img_number, np.std(band1) / (img_number))
    print(np.mean(band2) / img_number, np.std(band2) / (img_number))
    print(np.mean(band3) / img_number, np.std(band3) / (img_number))
    print(np.mean(band4) / img_number, np.std(band4) / (img_number))

import numpy as np
import cv2
import os
import tqdm
import wwtool
from matplotlib import pyplot as plt
import rasterio

def convert2rgb(sar, model=1):
    if model == 1:
        HH = sar[0]
        HV_VH = 0.5 * (sar[1] + sar[2])
        VV = sar[1]

        ret = np.dstack((HH, HV_VH, VV))

    if model == 2:
        R = sar[0]
        G = sar[3]
        B = sar[1]

        ret = np.dstack((B, G, R))

    return ret.astype("uint8")


def rescale(band, minval, maxval):
    band = 255 * (band - minval) / (maxval - minval)
    band = band.astype("int")
    band[band < 0] = 0
    band[band > 255] = 255
    return band

if __name__ == '__main__':
    src_folder = "./data/sn6/v0/MultiSensorSample/SAR-Intensity"
    dst_folder = "./data/sn6/v0/MultiSensorSample/Processed-SAR-Intensity"

    sar_img_list = os.listdir(src_folder)
    img_number = len(sar_img_list)
    sar_img_list.sort()

    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    to_rgb = True
    equalize_flag = False
    calculate_mean_std = True
    show_flag = False

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
                min_num = np.percentile(img_array[band][img_array[band]>0], 5)
                max_num = np.percentile(img_array[band][img_array[band]>0], 95)

                img_array[band] = rescale(img_array[band], min_num, max_num)

                if equalize_flag:
                    img = img_array[band].astype("uint8")
                    cv2.equalizeHist(img)
                    img_array[band] = img

                if show_flag:
                    img = img_array[band].astype("uint8")
                    wwtool.show_image(img)

                # hist = cv2.calcHist([img], [0], None , [256], [1, 256])
                # plt.plot(hist)

            # plt.show()

        if to_rgb:
            out_img = convert2rgb(img_array, 2)
            cv2.imwrite(os.path.join(os.getcwd(), dst_folder, sar_img), out_img)
            # wwtool.show_image(out_img)

        if calculate_mean_std:
            band1 += out_img[:, :, 0]
            band2 += out_img[:, :, 1]
            band3 += out_img[:, :, 2]
            # band4 += img_array[3]

    print(np.mean(band1) / img_number, np.std(band1) / (img_number))
    print(np.mean(band2) / img_number, np.std(band2) / (img_number))
    print(np.mean(band3) / img_number, np.std(band3) / (img_number))
    # print(np.mean(band4) / img_number, np.std(band4) / (img_number))

import os
import wwtool
import numpy as np
import rasterio as rio
import cv2
import pandas
import mmcv


if __name__ == '__main__':

    keywords = ['roof', 'footprint']

    for key in keywords:
        csv_file = './data/buildchange/v2/xian_fine/xian_fine_{}_gt.csv'.format(key)
        first_in = True

        json_dir = './data/buildchange/v2/xian_fine/labels_json'
        rgb_img_dir = './data/buildchange/v2/xian_fine/images'

        for json_fn in os.listdir(json_dir):
            base_name = wwtool.get_basename(json_fn)

            rgb_img_file = os.path.join(rgb_img_dir, base_name + '.png')
            json_file = os.path.join(json_dir, json_fn)

            annotations = mmcv.load(json_file)['annotations']

            masks = [wwtool.mask2polygon(anno[key]) for anno in annotations]

            csv_image = pandas.DataFrame({'ImageId': base_name,
                                            'BuildingId': range(len(masks)),
                                            'PolygonWKT_Pix': masks,
                                            'Confidence': 1})
            if first_in:
                csv_dataset = csv_image
                first_in = False
            else:
                csv_dataset = csv_dataset.append(csv_image)

        csv_dataset.to_csv(csv_file, index=False)
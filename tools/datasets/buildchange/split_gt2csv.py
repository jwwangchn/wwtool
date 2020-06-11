import os
import wwtool
import numpy as np
import rasterio as rio
import cv2
import pandas


if __name__ == '__main__':
    csv_file = './data/buildchange/v2/xian_fine/xian_fine_gt.csv'
    first_in = True


    json_dir = f'./data/buildchange/v2/xian_fine/labels'
    rgb_img_dir = f'./data/buildchange/v2/xian_fine/images'

    for shp_fn in os.listdir(json_dir):
        if not shp_fn.endswith('shp'):
            continue
        print("Processing: ", shp_fn)
        base_name = wwtool.get_basename(shp_fn)

        rgb_img_file = os.path.join(rgb_img_dir, base_name + '.jpg')
        shp_file = os.path.join(shp_dir, shp_fn)

        rgb_img = cv2.imread(rgb_img_file)
        geo_info = rio.open(rgb_img_file)

        shp_parser = wwtool.ShpParse()
        objects = shp_parser(shp_file, 
                            geo_info,
                            coord='pixel',
                            merge_flag=True, 
                            connection_mode='floor')

        gt_polygons = [wwtool.mask2polygon(obj['segmentation']) for obj in objects]

        # wwtool.show_polygons_on_image(gt_masks, rgb_img, output_file=None)

        csv_image = pandas.DataFrame({'ImageId': sub_fold + '_' + base_name,
                                        'BuildingId': range(len(gt_polygons)),
                                        'PolygonWKT_Pix': gt_polygons,
                                        'Confidence': 1})
        if first_in:
            csv_dataset = csv_image
            first_in = False
        else:
            csv_dataset = csv_dataset.append(csv_image)

    csv_dataset.to_csv(csv_file, index=False)
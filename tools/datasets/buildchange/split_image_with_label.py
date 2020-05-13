import os
import numpy as np
import cv2
from PIL import Image

import geojson
import shapely.wkt
import rasterio as rio

import wwtool

Image.MAX_IMAGE_PIXELS = int(2048 * 2048 * 2048 // 4 // 3)

if __name__ == '__main__':
    core_dataset_name = 'buildchange'
    src_version = 'v0'
    dst_version = 'v1'
    imagesets = ['train_shanghai', 'val_xian']
    subimage_size = 1024
    gap = subimage_size // 2

    shp_parser = wwtool.ShpParse()

    for imageset in imagesets:
        image_path = './data/{}/{}/{}/images'.format(core_dataset_name, src_version, imageset)
        anno_path = './data/{}/{}/{}/shp_4326'.format(core_dataset_name, src_version, imageset)
        geo_path = './data/{}/{}/{}/geo_info'.format(core_dataset_name, src_version, imageset)

        image_save_path = './data/{}/{}/{}/images'.format(core_dataset_name, dst_version, "{}_{}".format(imageset, subimage_size))
        wwtool.mkdir_or_exist(image_save_path)
        label_save_path = './data/{}/{}/{}/labels'.format(core_dataset_name, dst_version, "{}_{}".format(imageset, subimage_size))
        wwtool.mkdir_or_exist(label_save_path)

        # print(os.listdir(label_path))
        for idx, image_fn in enumerate(os.listdir(image_path)):
            if not image_fn.endswith('.jpg'):
                continue
            print(idx, image_fn)
            image_file = os.path.join(image_path, image_fn)
            file_name = os.path.splitext(os.path.basename(image_file))[0]
            if 'val' in imageset:
                annot_file = os.path.join(anno_path, file_name, file_name + '.shp')
            else:
                annot_file = os.path.join(anno_path, file_name + '.shp')

            if 'train' in imageset:
                geo_info_file = os.path.join(geo_path, file_name + '.png')
                geo_info = rio.open(geo_info_file)
                coord_flag = '4326'
            else:
                geo_info = rio.open(image_file)
                coord_flag = 'pixel'

            img = cv2.imread(image_file)

            objects = shp_parser(annot_file, geo_info, coord=coord_flag)

            masks = np.array([obj['segmentation'] for obj in objects])

            subimages = wwtool.split_image(img, subsize=subimage_size, gap=gap)
            subimage_coordinates = list(subimages.keys())

            if masks.shape[0] == 0:
                continue

            mask_centroids = []
            for obj in objects:
                wkt = shapely.wkt.loads(shapely.wkt.loads(obj['polygon']).centroid.wkt)
                geo = geojson.Feature(geometry=wkt, properties={})
                coordinate = geo.geometry["coordinates"]
                coordinate[0], coordinate[1] = abs(coordinate[0]), abs(coordinate[1])
                mask_centroids.append(coordinate)

            mask_centroids = np.array(mask_centroids)
            mask_centroids_ = mask_centroids.copy()
            
            for subimage_coordinate in subimage_coordinates:
                objects = []

                mask_centroids_[:, 0] = mask_centroids[:, 0] - subimage_coordinate[0]
                mask_centroids_[:, 1] = mask_centroids[:, 1] - subimage_coordinate[1]


                cx_bool = np.logical_and(mask_centroids_[:, 0] >= 0, mask_centroids_[:, 0] < subimage_size)
                cy_bool = np.logical_and(mask_centroids_[:, 1] >= 0, mask_centroids_[:, 1] < subimage_size)

                subimage_masks = masks[np.logical_and(cx_bool, cy_bool)]
                # print(masks)
                # print(np.logical_and(cx_bool, cy_bool))

                # print(subimage_masks)
                subimage_masks_ = []
                for subimage_mask in subimage_masks:
                    subimage_mask_np = np.array(subimage_mask)
                    subimage_mask_np[0::2] = subimage_mask_np[0::2] - subimage_coordinate[0]
                    subimage_mask_np[1::2] = subimage_mask_np[1::2] - subimage_coordinate[1]

                    subimage_masks_.append(subimage_mask_np.tolist())
                
                subimage_masks = subimage_masks_
                if len(subimage_masks) == 0:
                    continue
                img = subimages[subimage_coordinate]
                if np.mean(img) == 0:
                    continue

                label_save_file = os.path.join(label_save_path, '{}__{}_{}.txt'.format(file_name, subimage_coordinate[0], subimage_coordinate[1]))
                image_save_file = os.path.join(image_save_path, '{}__{}_{}.png'.format(file_name, subimage_coordinate[0], subimage_coordinate[1]))
                cv2.imwrite(image_save_file, img)
                
                for subimage_mask in subimage_masks:
                    subimage_objects = dict()
                    subimage_objects['mask'] = subimage_mask
                    subimage_objects['label'] = 'building'
                    objects.append(subimage_objects)
                wwtool.simpletxt_dump(objects, label_save_file, encode='mask')

import wwtool
import numpy as np
import rasterio as rio
import pycocotools.mask as maskUtils
import cv2
import os

def poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask

if __name__ == '__main__':
    image_dir = '/data/buildchange/v1/shanghai_1024/images'
    label_dir = '/data/buildchange/v1/shanghai_1024/labels'
    save_dir = '/data/buildchange/v1/shanghai_1024/vis'
    wwtool.mkdir_or_exist(save_dir)

    img_scale = 1024
    for image_fn in os.listdir(image_dir):
        image_file = os.path.join(image_dir, image_fn)
        label_file = os.path.join(label_dir, image_fn.replace('png', 'txt'))
        save_file = os.path.join(save_dir, image_fn)
        img = cv2.imread(image_file)

        with open(label_file, 'r') as f:
            lines = f.readlines()

        gt_masks = []
        for line in lines:
            object_struct = {}
            line = line.rstrip().split(' ')
            mask = [float(_) for _ in line[0:-1]]
            gt_masks.append([mask])

        img_mask = wwtool.generate_image(img_scale, img_scale, (0, 0, 0))

        COLORS = {'Blue': (0, 130, 200), 'Red': (230, 25, 75), 'Yellow': (255, 225, 25), 'Green': (60, 180, 75), 'Orange': (245, 130, 48), 'Purple': (145, 30, 180), 'Cyan': (70, 240, 240), 'Magenta': (240, 50, 230), 'Lavender': (230, 190, 255), 'Lime': (210, 245, 60), 'Teal': (0, 128, 128), 'Pink': (250, 190, 190), 'Brown': (170, 110, 40), 'Beige': (255, 250, 200), 'Maroon': (128, 0, 0), 'Mint': (170, 255, 195), 'Olive': (128, 128, 0), 'Apricot': (255, 215, 180), 'Navy': (0, 0, 128), 'Grey': (128, 128, 128), 'White': (255, 255, 255), 'Black': (0, 0, 0)}

        color_list = list(COLORS.keys())

        masks = wwtool.generate_image(img_scale, img_scale)
        for gt_mask in gt_masks:
            mask = poly2mask(gt_mask, img_scale, img_scale) * 1
            masks[:, :, 0] = mask * COLORS[color_list[np.random.randint(0, 20)]][2]
            masks[:, :, 1] = mask * COLORS[color_list[np.random.randint(0, 20)]][1]
            masks[:, :, 2] = mask * COLORS[color_list[np.random.randint(0, 20)]][0]
            img_mask += masks

        heatmap = wwtool.show_grayscale_as_heatmap(img_mask / 255.0, show=False, return_img=True)
        alpha = 0.5
        beta = (1.0 - alpha)

        fusion = cv2.addWeighted(heatmap, alpha, img, beta, 0.0)

        wwtool.show_image(fusion, save_name=save_file, wait_time=50)
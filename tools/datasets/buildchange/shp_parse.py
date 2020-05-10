import wwtool
import rasterio as rio
import pycocotools.mask as maskUtils
import cv2

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


png_img_fn = '/data/buildchange/v0/samples/images/L18_106968_219320.png'
jpg_img_fn = '/data/buildchange/v0/samples/images/L18_106968_219320.jpg'
shp_fn = '/data/buildchange/v0/samples/labels/L18_106968_219320.shp'

ori_img = rio.open(png_img_fn)
rgb_img = cv2.imread(jpg_img_fn)

shp_parser = wwtool.ShpParse()

objects = shp_parser(shp_fn, ori_img)

gt_masks = []
for obj in objects:
    mask = obj['segmentation']
    gt_masks.append([mask])

img = wwtool.generate_image(2048, 2048, 0)

for gt_mask in gt_masks:
    mask = poly2mask(gt_mask, 2048, 2048) * 255
    img += mask

heatmap = wwtool.show_grayscale_as_heatmap(img / 255.0, show=False, return_img=True)
alpha = 0.3
beta = (1.0 - alpha)
fusion = cv2.addWeighted(heatmap, alpha, rgb_img, beta, 0.0)

wwtool.show_image(fusion)
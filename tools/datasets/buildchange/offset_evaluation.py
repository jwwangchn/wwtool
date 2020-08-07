import os
import numpy as np
import pandas
import mmcv
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from skimage import measure
import cv2
import wwtool
from shapely import affinity


image_dir = '/home/jwwangchn/Documents/100-Work/170-Codes/aidet/data/buildchange/v1/xian_fine/images'
models = ['bc_v015_mask_rcnn_r50_v2_roof_trainval']
for model in models:
    print(model)
    anno_file = '/home/jwwangchn/Documents/100-Work/170-Codes/aidet/data/buildchange/v1/coco/annotations/buildchange_v1_val_xian_fine.json'
    results = mmcv.load(f'/home/jwwangchn/Documents/100-Work/170-Codes/aidet/results/buildchange/{model}/coco_results.pkl')

    coco = COCO(anno_file)
    catIds = coco.getCatIds(catNms=[''])
    imgIds = coco.getImgIds(catIds=catIds)
    for obj in ['footprint', 'roof']:
        first_in = True
        
        for idx, imgId in enumerate(imgIds):
            roof_masks = []
            footprint_masks = []
            footprint_polygons = []
            img = coco.loadImgs(imgIds[idx])[0]
            det, seg, offset = results[idx]

            polygons = []
            for label in range(len(det)):
                bboxes = det[label]
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                else:
                    segms = seg[label]

                if isinstance(offset, tuple):
                    offsets = offset[0]
                else:
                    offsets = offset
                
                for i in range(bboxes.shape[0]):
                    score = bboxes[i][4]
                    if score < 0.3:
                        continue
                    offset = offsets[i]

                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    gray = np.array(mask*255, dtype=np.uint8)

                    contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] if len(contours) == 2 else contours[1]
                    
                    if contours != []:
                        cnt = max(contours, key = cv2.contourArea)
                        if cv2.contourArea(cnt) < 5:
                            continue
                        mask = np.array(cnt).reshape(1, -1).tolist()[0]
                        if len(mask) < 8:
                            continue
                    else:
                        continue
                    
                    # TODO: convert to wkt format
                    polygon = wwtool.mask2polygon(mask)
                    roof_masks.append(mask)
                    polygons.append(polygon)
                    bbox = bboxes[i][0:4]
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    transform_matrix = [1, 0, 0, 1,  -1.0 * offset[0], -1.0 * offset[1]]
                    footprint_polygon = affinity.affine_transform(polygon, transform_matrix)
                    
                    footprint_mask = wwtool.polygon2mask(footprint_polygon)
                    footprint_masks.append(footprint_mask)
                    footprint_polygons.append(footprint_polygon)

            print(img['file_name'])
            # wwtool.show_polygons_on_image(footprint_masks, cv2.imread(os.path.join(image_dir, img['file_name'])), alpha=0.3)

            if obj == 'footprint':
                polygons = footprint_polygons
            else:
                polygons = polygons
            csv_image = pandas.DataFrame({'ImageId': img['file_name'].split('.')[0],
                                            'BuildingId': range(len(polygons)),
                                            'PolygonWKT_Pix': polygons,
                                            'Confidence': 1})
            if first_in:
                csv_dataset = csv_image
                first_in = False
            else:
                csv_dataset = csv_dataset.append(csv_image)

        csv_dataset.to_csv(f'/home/jwwangchn/Documents/100-Work/170-Codes/aidet/results/buildchange/{model}/result_{obj}.csv', index=False)
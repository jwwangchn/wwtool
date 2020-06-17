import numpy as np
import pandas
import mmcv
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from skimage import measure
import cv2
import wwtool

# def mask2polygon(mask):
#     contours = measure.find_contours(mask, 0.5, positive_orientation='low')
#     polygons = []

#     for contour in contours:
#         for i in range(len(contour)):
#             row, col = contour[i]
#             contour[i] = (col - 1, row - 1)

#         if contour.shape[0] < 3:
#             continue
        
#         poly = Polygon(contour)
#         if poly.area < 5:
#             continue
#         poly = poly.simplify(1.0, preserve_topology=False)
#         if poly.geom_type == 'MultiPolygon':
#             for poly_ in poly:
#                 polygons.append(poly_)
#                 segmentation = np.array(poly_.exterior.coords).ravel().tolist()
#         else:
#             polygons.append(poly)
#             segmentation = np.array(poly.exterior.coords).ravel().tolist()
#             segmentations.append(segmentation)


models = ['bc_v007_mask_rcnn_r50_v2_roof_trainval']
for model in models:
    print(model)
    anno_file = '/home/jwwangchn/Documents/100-Work/170-Codes/aidet/data/buildchange/v2/coco/annotations/buildchange_v2_val_xian_fine.json'
    results = mmcv.load(f'/home/jwwangchn/Documents/100-Work/170-Codes/aidet/results/buildchange/{model}/coco_results.pkl')

    coco = COCO(anno_file)
    catIds = coco.getCatIds(catNms=[''])
    imgIds = coco.getImgIds(catIds=catIds)

    first_in = True
    for idx, imgId in enumerate(imgIds):
        img = coco.loadImgs(imgIds[idx])[0]
        det, seg = results[idx]

        polygons = []
        for label in range(len(det)):
            bboxes = det[label]
            if isinstance(seg, tuple):
                segms = seg[0][label]
            else:
                segms = seg[label]
            for i in range(bboxes.shape[0]):
                score = bboxes[i][4]
                if score < 0.05:
                    continue
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
                polygons.append(polygon)

        csv_image = pandas.DataFrame({'ImageId': img['file_name'].split('.')[0],
                                        'BuildingId': range(len(polygons)),
                                        'PolygonWKT_Pix': polygons,
                                        'Confidence': 1})
        if first_in:
            csv_dataset = csv_image
            first_in = False
        else:
            csv_dataset = csv_dataset.append(csv_image)

    csv_dataset.to_csv(f'/home/jwwangchn/Documents/100-Work/170-Codes/aidet/results/buildchange/{model}/result.csv', index=False)
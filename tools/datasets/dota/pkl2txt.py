import os
import mmcv
import wwtool

outputs = mmcv.load('./tools/datasets/dota/dota_coco_results.pkl')

print("outputs length: ", len(outputs))

for idx, output in enumerate(outputs):
    det, seg = output
    print(len(det))
    print(len(seg))

    if idx == 0:
        break
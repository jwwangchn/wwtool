import numpy as np
import mmcv
from wwtool import pointobb2bbox, pointobb2pseudomask, thetaobb2pointobb, show_centerness


pointobb = thetaobb2pointobb([50, 50, 20, 50, 60*np.pi/180])
bbox = pointobb2bbox(pointobb)
print(bbox)
proposal = [50, 50, 70, 70]
# proposal = bbox

union_xmin = np.minimum(bbox[0], proposal[0])
union_ymin = np.minimum(bbox[1], proposal[1])
union_xmax = np.maximum(bbox[2], proposal[2])
union_ymax = np.maximum(bbox[3], proposal[3])
union_w = np.maximum(int(union_xmax - union_xmin), 1)
union_h = np.maximum(int(union_ymax - union_ymin), 1)

pointobb[::2] = pointobb[::2] - union_xmin
pointobb[1::2] = pointobb[1::2] - union_ymin

print(union_h, union_w)
union_pseudo_mask = pointobb2pseudomask(union_h, union_w, pointobb)
show_centerness(union_pseudo_mask, show=True)

moved_bbox = [proposal[0] - union_xmin, proposal[1] - union_ymin, proposal[2] - union_xmin, proposal[3] - union_ymin]
moved_bbox = [int(_) for _ in moved_bbox]
x1, y1, x2, y2 = moved_bbox

w = np.maximum(x2 - x1, 1)
h = np.maximum(y2 - y1, 1)

target = mmcv.imresize(union_pseudo_mask[y1:y1 + h, x1:x1 + w],
                        (28, 28), interpolation='bilinear')

show_centerness(union_pseudo_mask[y1:y1 + h, x1:x1 + w], show=True)
show_centerness(target, show=True)
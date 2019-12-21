import numpy as np
from wwtool import iou

bboxes1 = np.array([[1,2,3,4], [5,6,7,8], [9,1,1,1]], dtype=np.float32)
bboxes2 = np.array([[1,2,3,4], [5,6,7,8], [9,1,1,1], [9,1,1,1]], dtype=np.float32)

print(bboxes1.shape)
print(bboxes2.shape)


overlap = iou(bboxes1, bboxes2)

print(overlap)
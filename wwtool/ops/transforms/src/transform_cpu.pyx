import numpy as np
cimport numpy as np


cdef pointobb2pseudomask_cpu(int mask_height, int mask_width, pointobb):
    thetaobb = pointobb2thetaobb(pointobb)
    pointobb = thetaobb2pointobb(thetaobb)

    rotation_anchor_x, rotation_anchor_y = thetaobb[0], thetaobb[1]
    theta = thetaobb[4]

    pointobb = rotate_pointobb(pointobb, -theta, [rotation_anchor_x, rotation_anchor_y])

    bbox = pointobb2bbox(pointobb)
    bbox_pseudomask = bbox2pseudomask(mask_height, mask_width, bbox)

    # convert pseudo to centerness
    left = bbox_pseudomask[..., 0]
    top = bbox_pseudomask[..., 1]
    right = bbox_pseudomask[..., 2]
    bottom = bbox_pseudomask[..., 3]
    centerness = np.sqrt((np.minimum(left, right) / (np.maximum(left, right) + 1)) * (np.minimum(top, bottom) / (np.maximum(top, bottom)  + 1 )))
    centerness = mmcv.imrotate(centerness, theta*180.0/np.pi, center=(rotation_anchor_x + mask_width//2, rotation_anchor_y + mask_height//2))
    
    pointobb_pseudo_mask = centerness[mask_height // 2 : mask_height * 3 // 2, mask_width // 2 : mask_width * 3 // 2]

    return pointobb_pseudo_mask
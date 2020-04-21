import numpy as np
import wwtool
import mmcv


if __name__ == '__main__':
    thetaobbs = [[400, 400, 300, 150, 45*np.pi/180], 
                [600, 600, 300, 200, 135*np.pi/180]]
    pointobbs = [wwtool.thetaobb2pointobb(thetaobb) for thetaobb in thetaobbs]

    img = wwtool.generate_image(1024, 1024)
    img_origin = img.copy()
    wwtool.imshow_rbboxes(img, thetaobbs, win_name='origin')

    rotation_angle = 45
    rotation_anchor = [img.shape[0]//2, img.shape[1]//2]
    
    rotated_img = mmcv.imrotate(img_origin, rotation_angle)

    rotated_pointobbs = [wwtool.rotate_pointobb(pointobb, rotation_angle*np.pi/180, rotation_anchor) for pointobb in pointobbs]

    rotated_thetaobbs = [wwtool.pointobb2thetaobb(rotated_pointobb) for rotated_pointobb in rotated_pointobbs]
    wwtool.imshow_rbboxes(rotated_img, rotated_thetaobbs, win_name='rotated')
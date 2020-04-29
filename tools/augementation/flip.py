import numpy as np
import wwtool
import mmcv


if __name__ == '__main__':
    thetaobbs = [[200, 200, 300, 150, 45*np.pi/180], 
                [700, 800, 300, 200, 135*np.pi/180]]
    pointobbs = [wwtool.thetaobb2pointobb(thetaobb) for thetaobb in thetaobbs]

    img = wwtool.generate_image(1024, 1024)
    img_origin = img.copy()
    wwtool.imshow_rbboxes(img, thetaobbs, win_name='origin')

    pointobbs = np.array(pointobbs)

    pointobbs[..., 1::2] = 1024 - pointobbs[..., 1::2] - 1

    pointobbs = [wwtool.pointobb_best_point_sort(pointobb) for pointobb in pointobbs.tolist()]

    flipped_thetaobbs = [wwtool.pointobb2thetaobb(pointobb) for pointobb in pointobbs]

    wwtool.imshow_rbboxes(img_origin, flipped_thetaobbs, win_name='flipped')
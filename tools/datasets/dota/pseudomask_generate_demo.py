import cv2
import numpy as np

import mmcv
import wwtool
from wwtool.generation import generate_centerness_image, generate_image, generate_gaussian_image, generate_ellipse_image
from wwtool.visualization import show_grayscale_as_heatmap, show_image, show_image_surface_curve
from wwtool.transforms import pointobb_image_transform, thetaobb2pointobb, pointobb2bbox, pointobb2pseudomask

if __name__ == '__main__':
    image_size = (1024, 1024)
    img = generate_image(height=image_size[0], width=image_size[1], color=0)
    
    img = cv2.imread('/home/jwwangchn/Documents/100-Work/110-Projects/2019-DOTA/04-TGRS/TGRS-Paper-DOTA-2019/images/draw/ex_origin.png')
    height, width, _ = img.shape
    encoding = 'centerness'       # centerness, gaussian, ellipse
    if encoding == 'gaussian':
        anchor_image = generate_gaussian_image(image_size[0], image_size[1], scale=2.5, threshold=255 * 0.5)
    elif encoding == 'centerness':
        anchor_image = generate_centerness_image(image_size[0], image_size[1], factor=4, threshold=255 * 0.0)
    elif encoding == 'ellipse':
        anchor_image = generate_ellipse_image(image_size[0], image_size[1])
    # anchor_image_heatmap = wwtool.show_grayscale_as_heatmap(anchor_image, win_name='before', return_img=True)
    # cv2.imwrite('./heatmap.png', anchor_image_heatmap)

    # show_image_surface_curve(anchor_image, direction=2)

    thetaobbs = [[0, 0, 120, 200, 60 * np.pi/180.0],
                [300, 200, 50, 70, 30 * np.pi/180.0],
                [450, 500, 300, 230, 45 * np.pi/180.0]]
    thetaobbs = [[300, 200, 50, 70, 30 * np.pi/180.0]]
    pointobbs = []

    thetaobbs = [[265.041748046875, 461.97314453125, 83.17558288574219, 179.03013610839844, -0.4197568982424701], [158.6978759765625, 213.91476440429688, 82.1655502319336, 182.01783752441406, -0.445131230792636], [163.17056274414062, 505.3885498046875, 82.40795135498047, 178.5282440185547, -0.41368921859386076], [57.678192138671875, 259.0009765625, 83.1702880859375, 179.4698944091797, -0.4243928624890527], [264.6916809082031, 741.2076416015625, 82.70415496826172, 145.08529663085938, -0.42366891543622837], [373.6033630371094, 713.7134399414062, 81.72265625, 179.64755249023438, -0.4245613407175139]]

    for thetaobb in thetaobbs:
        pointobb = thetaobb2pointobb(thetaobb)
        pointobbs.append(pointobb)

    pseudomasks = np.zeros((height, width), dtype=np.int32)
    for pointobb in pointobbs:
        transformed, mask_location = pointobb2pseudomask(pointobb, anchor_image, host_height = height, host_width = width)
        # show_grayscale_as_heatmap(transformed)
        transformed = transformed.astype(np.int32)
        pseudomasks[mask_location[1]:mask_location[3], mask_location[0]:mask_location[2]] = np.where(transformed > pseudomasks[mask_location[1]:mask_location[3], mask_location[0]:mask_location[2]], transformed, pseudomasks[mask_location[1]:mask_location[3], mask_location[0]:mask_location[2]])

    pseudomasks_ = show_grayscale_as_heatmap(pseudomasks / 255.0, False, return_img=True)
    alpha = 0.4
    beta = (1.0 - alpha)
    pseudomasks = cv2.addWeighted(pseudomasks_, alpha, img, beta, 0.0)

    show_image(pseudomasks, save_name='centermap_result.png')
    # show_grayscale_as_heatmap(pseudomasks)
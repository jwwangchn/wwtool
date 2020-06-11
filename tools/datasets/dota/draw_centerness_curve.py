import cv2
import numpy as np
from matplotlib import pyplot as plt

import mmcv
import wwtool

from wwtool import generate_centerness_image, generate_image, generate_gaussian_image, generate_ellipse_image
from wwtool.visualization import show_grayscale_as_heatmap, show_image, show_image_surface_curve
from wwtool.transforms import pointobb_image_transform, thetaobb2pointobb, pointobb2bbox, pointobb2pseudomask

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


if __name__ == '__main__':
    # data_length = 512
    # x = np.arange(0, data_length+1, 0.1)

    # left = x
    # left = np.maximum(left, 0)
    # right = data_length - x
    # right = np.maximum(right, 0)

    # factors = [4]

    # for factor in factors:
    #     threshold = 1.0 * 0.5
    #     y = ((np.minimum(left, right) / (np.maximum(left, right) + 1))) ** (1/factor)
    #     y = np.clip((y * (1.0 - threshold) + threshold), 0, 1.0)
    #     plt.plot(x, y, label='factor={}'.format(factor))
    #     plt.legend(loc='best')

    # plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)

    height, width = 256, 256
    factor = 4 
    threshold = 0.5

    bbox = [0, 0, width, height]
    x_range = np.arange(0, width+1)
    y_range = np.arange(0, height+1)
    index_x, index_y = np.meshgrid(x_range, y_range)
    

    left = index_x - bbox[0]
    left = np.maximum(left, 0)
    right = bbox[2] - index_x
    right = np.maximum(right, 0)
    top = index_y - bbox[1]
    top = np.maximum(top, 0)
    bottom = bbox[3] - index_y
    bottom = np.maximum(bottom, 0)

    centerness_prob = ((np.minimum(left, right) / (np.maximum(left, right) + 1)) * (np.minimum(top, bottom) / (np.maximum(top, bottom) + 1 ))) ** (1/factor)
    centerness_image = np.clip((centerness_prob * (1 - threshold) + threshold), 0, 1)


    x_zeros = np.zeros(width + 1)
    y_zeros = np.ones(height + 1) * (height)
    x_range = np.arange(0, width + 1)
    y_range = np.arange(0, height + 1)
    left = x_range
    left = np.maximum(left, 0)
    right = width - x_range
    right = np.maximum(right, 0)

    z = ((np.minimum(left, right) / (np.maximum(left, right) + 1))) ** (1/factor)

    # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    ax.plot_surface(index_x, index_y, centerness_prob, rstride=1, cstride=1, cmap='rainbow', alpha=0.6)

    ax.plot(x_range, y_zeros, z)
    ax.plot(x_zeros, y_range, z)
    # cset = ax.contour(index_x, index_y, centerness_prob, zdir='x', offset=0, cmap=cm.coolwarm)
    # cset = ax.contour(index_x, index_y, centerness_prob, zdir='y', offset=64, cmap=cm.coolwarm)
    ax.grid(False)
    # plt.xticks([])
    # plt.yticks([])
    # plt.axis('off')
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_zlim(0.0, 1.0)
    # ax.legend()
    plt.savefig('/home/jwwangchn/Documents/100-Work/110-Projects/2019-DOTA/04-TGRS/TGRS-Paper-DOTA-2019/images/centermap_plot.pdf', bbox_inches='tight', dpi=600, pad_inches=0.05)
    plt.show()
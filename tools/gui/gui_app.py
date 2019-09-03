import os, cv2
import numpy as np
import pycocotools.mask as maskUtils
import mmcv
import random
from queue import Queue
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg

from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.core import get_classes
from mmdet.apis import inference_detector, dota_show_result, dota_inference_detector
from mmdet.datasets.dota.transform import maskobb2thetaobb
from mmdet.datasets.dota.utils import show_thetaobb, show_hobb, show_pointobb
from wwtool.gui import MainUI
from processer import ProducerThread

pg.setConfigOptions(imageAxisOrder='row-major')

def update():
    if display.alive:
        display.update()
        app.processEvents()  # force complete redraw for every plot
    else:
        pass

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    DOTA_CLASS_NAMES = ('__background__', 'harbor', 'ship', 'small-vehicle', 'large-vehicle', 'storage-tank', 'plane', 'soccer-ball-field', 'bridge', 'baseball-diamond', 'tennis-court', 'helicopter', 'roundabout', 'swimming-pool', 'ground-track-field', 'basketball-court')

    DOTA_CLASS_NAMES_OFFICIAL = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter')

    COLORS = {'ship': (75, 25, 230), 'harbor': (75, 180, 60), 'small-vehicle': (25, 225, 255), 'large-vehicle': (200, 130, 245), 'plane': (48, 130, 245), 'soccer-ball-field': (128, 0, 0), 'tennis-court': (240, 240, 70), 'baseball-diamond': (230, 50, 240), 'roundabout': (195, 255, 170), 'swimming-pool': (190, 190, 250), 'basketball-court': (128, 128, 0), 'storage-tank': (255, 190, 230), 'ground-track-field': (40, 110, 170), 'helicopter': (200, 130, 0), 'bridge': (200, 250, 255)}

    dataset = 'dota'

    data_version = 'v1'
    imageset = 'test'
    epoch = 12
    rate = '1.0'
    score_thr = 0.5
    save_vis = True

    img_dir = "./data/dota/{}/{}/images".format(data_version, imageset)
    imgs_list = []
    for folder, subs, files in os.walk(img_dir):
        for filename in files:  
            imgs_list.append(os.path.abspath(os.path.join(folder, filename)))
    print(imgs_list[0:5])
    random.shuffle(imgs_list)

    config_version = 'v301'

    config_file = './configs/dota/dota_{}.py'.format(config_version)
    checkout_file = './work_dirs/dota_{}/epoch_12.pth'.format(config_version)

    BUFFER = Queue(maxsize = 32)
    app = QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(border=True)
    display = MainUI(win, BUFFER)

    producer = ProducerThread(imgs_list=imgs_list,
                              config_file=config_file,
                              checkout_file=checkout_file,
                              score_thr=score_thr,
                              buffer=BUFFER)
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000/25)

    # Start the Producer Thread
    producer.start()
    # Start the application
    QtGui.QApplication.instance().exec_()

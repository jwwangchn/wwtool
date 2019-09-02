import sys

import numpy as np
from PyQt5 import QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from matplotlib import cm

class MainUI(object):
    def __init__(self, MainWindow, buffer):

        self.alive = True
        self.exit_on_end = True

        MainWindow.setWindowTitle('Demo for Object Detetion')
        MainWindow.resize(1200, 800)

        # Define the ViewBoxes for each of the images to be displayed
        self.score_box = MainWindow.addViewBox(1, 1, colspan=1)
        self.gt_box = MainWindow.addViewBox(1, 0, colspan=1)
        self.score_box.invertY(True)  # Images usually have their Y-axis pointing downward
        self.gt_box.invertY(True)

        self.score_box.setAspectLocked(True)
        self.gt_box.setAspectLocked(True)


        self.fpsLabel = pg.LabelItem(justify='left')
        MainWindow.addItem(self.fpsLabel, 0, 0)
        self.bufferLabel = MainWindow.addLabel('', 0, 1)

        self.nameLabel = MainWindow.addLabel('', 3, 0, colspan=2)
        font = QtGui.QFont()
        font.setPointSize(4)
        self.nameLabel.setFont(font)

        self.score_img = pg.ImageItem()
        self.gt_img = pg.ImageItem()
        self.ref_img = pg.ImageItem()
        self.score_box.addItem(self.score_img)
        self.gt_box.addItem(self.gt_img)

        # self.view_box.setRange(QtCore.QRectF(0, 0, 512, 512))
        self.bounding_box = QtWidgets.QGraphicsRectItem()
        self.bounding_box.setPen(QtGui.QColor(255, 0, 0))
        self.bounding_box.setParentItem(self.gt_img)
        self.gt_box.addItem(self.bounding_box)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        self.peak = pg.GraphItem(size=30, symbol='+', pxMode=True,
                                 symbolBrush=brush,
                                 symbolPen=None)
        self.peak.setParentItem(self.score_img)
        self.score_box.addItem(self.peak)
        self.peak_pos = None
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255, alpha=0))

        self.prior_radius = pg.GraphItem(size=0, symbol='o', pxMode=True,
                                         symbolBrush=brush, symbolPen='b')
        self.prior_radius.setParentItem(self.score_img)
        self.score_box.addItem(self.prior_radius)

        # Add the Labels to the images
        param_dict = {'color': (255, 255, 255),
                      'anchor': (0, 1)}
        label_score = pg.TextItem(text='Center Map', **param_dict)
        label_gt = pg.TextItem(text='Tracking Box', **param_dict)

        font.setPointSize(16)
        label_score.setFont(font)
        label_gt.setFont(font)

        label_score.setParentItem(self.score_img)
        label_gt.setParentItem(self.gt_img)

        self.score_box.addItem(label_score)
        self.gt_box.addItem(label_gt)


        # The alpha parameter is used to overlay the score map with the image,
        # where alpha=1 corresponds to the score_map alone and alpha=0 is
        # the image alone.


        self.error_plot = MainWindow.addPlot(4, 0, colspan=2, title='Confident Score')
        self.curve = self.error_plot.plot(pen='y')
        # Sets a line indicating the 63 pixel error corresponding to half of the
        # initial reference bounding box, and a possible measure of tracking
        # failure.
        half_ref = pg.InfiniteLine(movable=False, angle=0, pen=(0, 0, 200),
                                   label='ctr_error={value:0.2f}px',
                                   labelOpts={'color': (200,200,200),
                                              'movable': True,
                                              'fill': (0, 0, 200, 100)})
        half_ref.setPos([1, 1])
        self.error_plot.addItem(half_ref)
        self.pscore = []

        self.index = 0

        MainWindow.show()

        self.lastTime = ptime.time()
        self.fps = None
        self.buffer = buffer

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = np.stack([gray, gray, gray], axis=2)
        return gray

    def exit(self):
        sys.exit()

    def update(self):
        # Gets an element from the buffer and free the buffer
        buffer_element = self.buffer.get()
        self.buffer.task_done()
        # When the Producer Thread finishes publishing the data it sends a None
        # through the buffer to sinalize it has finished.
        if buffer_element is not None:
            img = buffer_element.img
            name = buffer_element.file_name
            bbox = buffer_element.bbox
            best_pscore = buffer_element.best_pscore

            print(img, name, bbox, best_pscore)

            # frame
            #img_gray = self.rgb2gray(img)/255
            self.score_img.setImage(img, autoDownsample=False)

            self.gt_img.setImage(img, autoDownsample=False)
            if bbox is not None:
               self.bounding_box.setRect(*bbox)
               self.peak.setData(pos=[(bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2)])
               self.pscore.append(best_pscore)
               self.curve.setData(self.pscore)

            # if bbox is not None:
            #     self.bounding_box.setRect(*bbox)
            #     center_error = np.linalg.norm([bbox[0]+bbox[2]/2-peak[1], bbox[1]+bbox[3]/2-peak[0]])
            #     self.center_errors.append(center_error)
            #     self.curve.setData(self.center_errors)
            # else:
            #     self.bounding_box.setRect(0, 0, 0, 0)

            # Calculate the fps rate.
            now = ptime.time()
            dt = now - self.lastTime
            self.lastTime = now
            if self.fps is None:
                self.fps = 1.0/dt
            else:
                s = np.clip(dt*3., 0, 1)
                self.fps = self.fps * (1-s) + (1.0/dt) * s
            self.fpsLabel.setText('{:.2f} fps'.format(self.fps), color='w')
            self.bufferLabel.setText('{} in Buffer'.format(self.buffer.qsize()))
            self.nameLabel.setText(name, size='10pt', color='w')

            self.index += 1
        else:
            # Set alive attribute to False to indicate the end of the program
            self.alive = False
            if self.exit_on_end:
                self.exit()
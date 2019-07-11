import os
import cv2
import numpy as np

import torch
from torchvision.ops.boxes import nms
from wwtool.visualization import imshow_bboxes


if __name__ == '__main__':
    result1 = '/home/jwwangchn/Documents/Research/Projects/2019-visdrone/09-Results/submit_method1'
    result2 = '/home/jwwangchn/Documents/Research/Projects/2019-visdrone/09-Results/submit_method2'
    result3 = '/home/jwwangchn/Documents/Research/Projects/2019-visdrone/09-Results/submit_method3'

    ensemble = '/home/jwwangchn/Documents/Research/Projects/2019-visdrone/09-Results/ensemble'
    imgpath = '/media/jwwangchn/data/visdrone/v1/coco/test/'

    nms_threshold = 0.5
    show_flag = True
    ensemble_list = [result1, result2]

    for file_name in os.listdir(result1):
        bboxes = []
        scores = []
        labels = []

        datas = list()
        for result_file in ensemble_list:
            with open(os.path.join(result_file, file_name)) as f:
                data = f.readlines()
            datas += data

        for data in datas:
            xmin, ymin, bbox_w, bbox_h = [float(xy) for xy in data.split(',')[:4]]
            score = float(data.split(',')[4])
            label = float(data.split(',')[5])

            xmax, ymax = xmin + bbox_w, ymin + bbox_h
            bbox = [xmin, ymin, xmax, ymax]
            
            scores.append(score)
            bboxes.append(bbox)
            labels.append(label)

        bboxes = torch.tensor(bboxes)
        scores = torch.tensor(scores)
        labels = torch.tensor(labels)

        keeps = nms(bboxes, scores, nms_threshold)

        final_bboxes = bboxes[keeps].numpy()
        final_scores = scores[keeps].numpy()
        final_labels = labels[keeps].numpy()

        ensemble_save_file = os.path.join(ensemble, file_name)
        with open(ensemble_save_file, 'w') as f:
            for bbox, label, score in zip(final_bboxes, final_labels, final_scores):
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                w = xmax - xmin
                h = ymax - ymin

                command_bbox = '{:.0f},{:.0f},{:.0f},{:.0f},{:.4f},{:.0f},{},{}\n'.format(xmin, ymin, w, h, score, label, -1, -1)
                f.write(command_bbox)

        if show_flag:
            im = cv2.imread(os.path.join(imgpath, file_name.split('.')[0] + '.jpg'))
            imshow_bboxes(im, 
                        final_bboxes, 
                        scores = None, 
                        score_threshold = 0.0, 
                        colors = 'red', 
                        wait_time = 2000)
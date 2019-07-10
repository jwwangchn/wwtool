import os
import cv2

from wwtool.visualization import imshow_bboxes


if __name__ == '__main__':
    imgpath = '/media/jwwangchn/data/visdrone/v1/coco/test/'
    anno_path = '/home/jwwangchn/Downloads/submit'

    for label_file in os.listdir(anno_path):
        labels = open(os.path.join(anno_path, label_file), 'r').readlines()
        
        im = cv2.imread(os.path.join(imgpath, label_file.split('.')[0] + '.jpg'))
        
        bboxes = []
        scores = []
        for label in labels:
            xmin, ymin, bbox_w, bbox_h = [float(xy) for xy in label.split(',')[:4]]
            score = float(label.split(',')[4])
            xmax, ymax = xmin + bbox_w, ymin + bbox_h
            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)
            scores.append(score)
        
        imshow_bboxes(im, 
                    bboxes, 
                    scores = None, 
                    score_threshold = 0.0, 
                    colors = 'red', 
                    wait_time = 3000)

import numpy as np
import mmcv
import cv2
from threading import Thread
from collections import namedtuple
import wwtool

from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.core import get_classes
from mmdet.apis import inference_detector, dota_show_result, dota_inference_detector
from mmdet.datasets.dota.transform import maskobb2thetaobb
from mmdet.datasets.dota.utils import show_thetaobb, show_hobb, show_pointobb


class ProducerThread(Thread):
    def __init__(self,
                imgs_list,
                config_file,
                checkout_file,
                score_thr,
                buffer):
        super(ProducerThread, self).__init__(daemon=True)
        self.buffer_element = namedtuple('BufferElement', ['img', 'file_name', 'bbox', 'best_pscore'])
        self.imgs_list = imgs_list
        self.config_file = config_file
        self.checkout_file = checkout_file
        self.score_thr = score_thr
        self.buffer = buffer

        self.cfg = mmcv.Config.fromfile(self.config_file)
        self.cfg.model.pretrained = None
        self.model_type = self.cfg.model.type
        if self.model_type == 'RBBoxRCNN':
            self.encode_method = self.cfg.test_cfg.rbbox.encode
        else:
            self.encode_method = None

        print("Start to load the model")
        self.model = build_detector(self.cfg.model, test_cfg=self.cfg.test_cfg)
        _ = load_checkpoint(self.model, self.checkout_file)
        print("Finish loading the model")
        
    def run(self):
        for idx, img_path in enumerate(self.imgs_list):
            # print("{}, {}".format(idx, img_path))
            # if idx > 100:
            #     break
            # img = cv2.imread(img_path)
            # img_origin = img.copy()
            # image_name = img_path.split('/')[-1]

            # result = dota_inference_detector(self.model, img, self.cfg, device='cuda:0')

            # bbox_result, rbbox_result = result
            # labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]

            # labels = np.concatenate(labels)
            # bboxes = np.vstack(bbox_result)
            # if self.encode_method == None:
            #     rbboxs = mmcv.concat_list(rbbox_result)
            # else:
            #     rbboxs = np.vstack(rbbox_result)

            # scores = bboxes[:, -1]
            # inds = scores > self.score_thr

            # obbs = []
            # for idx in np.where(scores > self.score_thr)[0]:
            #     obb = rbboxs[idx]
            #     if self.model_type == 'MaskOBBRCNN':
            #         obb = maskobb2thetaobb(obb)
            #     obbs.append(obb)
                    
            # hbbs = bboxes[inds, :-1]
            # labels = labels[inds]
            # scores = scores[inds]

            img_origin = wwtool.generate_image()
            image_name = ''
            hbbs = np.array([256, 256, 50 + idx, 80 + idx])
            scores = 1.0
            data = self.buffer_element(img_origin, image_name, hbbs, scores)
            self.buffer.put(data)

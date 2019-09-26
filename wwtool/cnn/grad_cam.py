import os
import cv2
import numpy as np

import torch
from torch.autograd import Variable, Function
from torchvision import models, utils


class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if name == "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        return target_activations, output


class GradCam:
    def __init__(self, model, target_layer_names, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.model = model.to(device)
        
        self.extractor = ModelOutputs(self.model, target_layer_names)
        
    def forward(self, img):
        return self.model(img)
    
    def __call__(self, img, topk=1):
        bboxes_top = list()
        
        for k in range(topk):
            features, output = self.extractor(img.to(self.device))
        
            index = np.argsort(output.cpu().numpy())[-1][-(k+1)]  # top probs to low
            
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            one_hot = torch.from_numpy(one_hot)
            one_hot.requires_grad = True
            
            one_hot = torch.sum(one_hot.to(device) * output)
            
            #self.model.features.zero_grad()
            #self.model.classifier.zero_grad()
            self.model.zero_grad()
            
            one_hot.backward(retain_graph=True)
            
            cam_list = []
            for j in range(len(features)):
                grads_val = self.extractor.get_gradients()[j].cpu().numpy()
                target = features[len(features) - j - 1]
                target = target.cpu().numpy()[0, :]
                
                weights = np.mean(grads_val, axis=(2, 3))[0, :]
                cam = np.zeros(target.shape[1:], dtype=np.float32)
                
                for i, w in enumerate(weights):
                    cam += w * target[i, :, :]
                
                cam = np.maximum(cam, 0)
                cam = cv2.resize(cam, (224, 224))
                cam = cam - np.min(cam)
                cam = cam / np.max(cam)
                
                cam_list.append(cam)
        
            for i in range(len(cam_list) - 1):
                cam_list[i] = cv2.resize(cam_list[i], (cam_list[i + 1].shape[0], cam_list[i + 1].shape[1]))
                #cam_list[i] = cv2.resize(cam_list[i], (224, 224))
                
                #cam_list[i + 1] = cam_list[i + 1] + cam_list[i]
                cam_list[i + 1] = np.array((cam_list[i + 1], cam_list[i])).max(axis=0)
            
            mask = cam_list[-1] 
            
            mask_copy = mask.copy()
            
            shreld = mask.sum() / (mask.shape[0] * mask.shape[1]) * 1.7
            
            mask = np.array(mask >= shreld, dtype='uint8')
            
            ret, binary = cv2.threshold(mask, shreld, 255, cv2.THRESH_BINARY)
            
            _, contours, hierarcy = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            cont_ = sorted(contours, key=cv2.contourArea, reverse=True)
            
            if len(cont_) == 0:
                box = np.zeros((4,2), dtype=int)
            else:
                c = cont_[0]
                # compute the rotated bounding box of the largest contour
                rect = cv2.minAreaRect(c)
                box = np.int0(cv2.boxPoints(rect))
            bboxes_top.append({"classify":index, "bbox":box, "mask":mask_copy})
        return bboxes_top
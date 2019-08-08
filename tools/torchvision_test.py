import torch
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import transforms as T
import cv2
import os

num_classes = 2

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# move model to the right device
model.to(device)

model.load_state_dict(torch.load("./0.pth"))

model.eval()

img_fold = "/media/jwwangchn/data/uav/v3/trainval/images/"

for img_name in os.listdir(img_fold):
    im = cv2.imread(os.path.join(img_fold, img_name))
    
    im = im.transpose((2, 0, 1))/255.0

    im = [torch.FloatTensor(im).to(device)]

    # x = [torch.rand(3, 300, 400).to(device), torch.rand(3, 500, 400).to(device)]
    # print(x[0])
    predictions = model(im)

    boxes = predictions[0]['boxes'].to('cpu')

    print(boxes.detach().numpy())
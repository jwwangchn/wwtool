import torchvision

model = torchvision.models.segmentation.fcn_resnet50(pretrained=True, progress=True, num_classes=2, aux_loss=None)
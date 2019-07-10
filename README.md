## WWTOOL
WWTOOL is a foundational python library for Computer Vision Research.

It will provide the following functionalities.

- Detection annotation visualization
- Toolkit for object detection dataset (dota, visdrone, ...)

### Requirements

- Python 3.7
- Pytorch 1.0+
- CUDA 8.0+
- [mmcv](https://github.com/open-mmlab/mmcv)

### Installation
```
git clone https://github.com/jwwangchn/wwtool.git
cv wwtool
python setup.py develop
```

### Datasets

```
wwtool
├── wwtool
├── tests
├── data (symlink)
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012

```
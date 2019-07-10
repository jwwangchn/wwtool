## WWTOOL
WWTOOL is a foundational python library for Computer Vision Research.

It will provide the following functionalities.

- Detection annotation visualization
- Toolkit for object detection dataset ([dota](https://captain-whu.github.io/DOTA/index.html), [visdrone](http://aiskyeye.com/), [uav-bd](https://jwwangchn.cn/UAV-BD), ...)

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

### Future works
- [ ] Dataset statistic
- [ ] Results generation tools for dota and visdrone dataset
- [ ] Data augmentation

### Datasets

```
wwtool
├── wwtool
├── tests
├── data (symlink)
│   ├── dota
│   │   ├── coco
│   │   │   ├── annotations
│   │   ├── train
│   │   ├── val
│   │   ├── test

```
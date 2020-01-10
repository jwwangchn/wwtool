import cvxpy as cp
import numpy as np
import json
from pycocotools.coco import COCO

def split_coco(src_ann_file, image_sets={"trainval" : 0.8, "train":0.8, "val":0.2, "test" : 0.2}):
    pass


info = {"year" : 2019,
            "version" : "1.0",
            "description" : "SMALL-COCO",
            "contributor" : "Jinwang Wang",
            "url" : "jwwangchn.cn",
            "date_created" : "2019"
        }

licenses = [{"id": 1,
                "name": "Attribution-NonCommercial",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }]

categories = [{'supercategory': 'none', 'id': 1,  'name': 'airplane',                 },
                    {'supercategory': 'none', 'id': 2,  'name': 'bridge',                   },
                    {'supercategory': 'none', 'id': 3,  'name': 'storage-tank',             },
                    {'supercategory': 'none', 'id': 4,  'name': 'ship',                     },
                    {'supercategory': 'none', 'id': 5,  'name': 'swimming-pool',            },
                    {'supercategory': 'none', 'id': 6,  'name': 'tennis-court',             },
                    {'supercategory': 'none', 'id': 7,  'name': 'vehicle',                  },
                    {'supercategory': 'none', 'id': 8,  'name': 'person',                  },
                    {'supercategory': 'none', 'id': 9,  'name': 'harbor',                  },
                    {'supercategory': 'none', 'id': 10,  'name': 'wind-mill',               }]

src_ann_file_name = ['small', 'trainval_test_trainval_filtered', 'v1', '1.0']
src_ann_file = '/data/{}/v1/coco/annotations/{}.json'.format(src_ann_file_name[0], '_'.join(src_ann_file_name))

dst_trainval_ann_file_name = ['small', 'trainval', 'v1', '1.0']
dst_trainval_ann_file = '/data/{}/v1/coco/annotations/{}.json'.format(dst_trainval_ann_file_name[0], '_'.join(dst_trainval_ann_file_name))

dst_train_ann_file_name = ['small', 'train', 'v1', '1.0']
dst_train_ann_file = '/data/{}/v1/coco/annotations/{}.json'.format(dst_train_ann_file_name[0], '_'.join(dst_train_ann_file_name))

dst_val_ann_file_name = ['small', 'val', 'v1', '1.0']
dst_val_ann_file = '/data/{}/v1/coco/annotations/{}.json'.format(dst_val_ann_file_name[0], '_'.join(dst_val_ann_file_name))

dst_test_ann_file_name = ['small', 'test', 'v1', '1.0']
dst_test_ann_file = '/data/{}/v1/coco/annotations/{}.json'.format(dst_test_ann_file_name[0], '_'.join(dst_test_ann_file_name))

trainval_rate = 0.5
train_rate = 0.8

coco = COCO(src_ann_file)
catIds = coco.getCatIds(catNms=[''])
imgIds = coco.getImgIds(catIds=catIds)

np.random.seed(0)
np.random.shuffle(imgIds)
trainval_image_ids = imgIds[0 : int(len(imgIds) * trainval_rate)]

train_image_ids = trainval_image_ids[0 : int(len(trainval_image_ids) * train_rate)]

val_image_ids = trainval_image_ids[int(len(trainval_image_ids) * train_rate):]

test_image_ids = imgIds[int(len(imgIds) * trainval_rate):]

for imgIds, dst_ann_file in zip([trainval_image_ids, train_image_ids, val_image_ids, test_image_ids], [dst_trainval_ann_file, dst_train_ann_file, dst_val_ann_file, dst_test_ann_file]):
    images, annotations = [], []
    for idx, _ in enumerate(imgIds):
        img = coco.loadImgs(imgIds[idx])[0]
        annIds = coco.getAnnIds(imgIds = img['id'], catIds = catIds, iscrowd = None)
        anns = coco.loadAnns(annIds)

        images.append(img)
        annotations = annotations + anns

    json_data = {"info" : info,
                "images" : images,
                "licenses" : licenses,
                "type" : "instances",
                "annotations" : annotations,
                "categories" : categories}

    with open(dst_ann_file, "w") as jsonfile:
        json.dump(json_data, jsonfile, sort_keys=True, indent=4)
import cvxpy as cp
import numpy as np
import json
from pycocotools.coco import COCO

def mergecoco(src_ann_files, info, licenses, categories, dst_ann_file):
    images, annotations = [], []
    for src_ann_file in src_ann_files:
        coco = COCO(src_ann_file)
        catIds = coco.getCatIds(catNms=[''])
        imgIds = coco.getImgIds(catIds=catIds)
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

if __name__ == '__main__':
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

    categories = [  {'supercategory': 'none', 'id': 1,  'name': 'airplane',                 },
                    {'supercategory': 'none', 'id': 2,  'name': 'bridge',                   },
                    {'supercategory': 'none', 'id': 3,  'name': 'storage-tank',             },
                    {'supercategory': 'none', 'id': 4,  'name': 'ship',                     },
                    {'supercategory': 'none', 'id': 5,  'name': 'swimming-pool',            },
                    {'supercategory': 'none', 'id': 6,  'name': 'tennis-court',             },
                    {'supercategory': 'none', 'id': 7,  'name': 'vehicle',                  },
                    {'supercategory': 'none', 'id': 8,  'name': 'person',                   },
                    {'supercategory': 'none', 'id': 9,  'name': 'harbor',                   },
                    {'supercategory': 'none', 'id': 10, 'name': 'wind-mill',                }]

    src_ann_file_names = [['small', 'trainval_test', 'v1', '1.0'],
                        ['visdrone', 'trainval_filtered', 'v1', '1.0', 'small_object']]
    src_ann_files = ['/data/{}/v1/coco/annotations/{}.json'.format(src_ann_file_name[0], '_'.join(src_ann_file_name)) for src_ann_file_name in src_ann_file_names]

    new_imageset_name = "_".join([src_ann_file_name[1] for src_ann_file_name in src_ann_file_names])
    print(new_imageset_name)

    dst_ann_file_name = ['small', new_imageset_name, 'v1', '1.0']
    dst_ann_file = '/data/{}/v1/coco/annotations/{}.json'.format(dst_ann_file_name[0], '_'.join(dst_ann_file_name))
    
    use_origin_info = False
    if use_origin_info:
        coco = COCO(src_ann_files[0])
        info = coco.dataset['info']
        licenses = coco.dataset['licenses']
        categories = coco.dataset['categories']

    mergecoco(src_ann_files, info, licenses, categories, dst_ann_file)
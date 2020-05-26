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

    cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    sub_city_folds = {'beijing': ['arg', 'google', 'ms', 'tdt'],
                  'chengdu': ['arg', 'google', 'ms', 'tdt'],
                  'haerbin': ['arg', 'google', 'ms'],
                  'jinan': ['arg', 'google', 'ms', 'tdt'],
                  'shanghai': ['arg', 'google', 'ms', 'tdt', 'PHR2016', 'PHR2017']}
    
    release_version = 'v2'
    imageset = 'val'
    
    for city in cities:
        src_ann_file_names = []
        for sub_city_fold in sub_city_folds:
            src_ann_file_names.append(['buildchange', release_version, imageset, city, sub_city_fold])

        src_ann_files = ['./data/{}/v1/coco/annotations/{}.json'.format(src_ann_file_name[0], '_'.join(src_ann_file_name)) for src_ann_file_name in src_ann_file_names]

        new_imageset_name = "_".join(['buildchange', release_version, imageset, city])
        print(new_imageset_name)

        dst_ann_file_name = ['buildchange', new_imageset_name]
        dst_ann_file = './data/{}/v1/coco/annotations/{}.json'.format(dst_ann_file_name[0], '_'.join(dst_ann_file_name))
        
        use_origin_info = True
        if use_origin_info:
            coco = COCO(src_ann_files[0])
            # info = coco.dataset['info']
            # licenses = coco.dataset['licenses']
            categories = coco.dataset['categories']

        mergecoco(src_ann_files, info, licenses, categories, dst_ann_file)
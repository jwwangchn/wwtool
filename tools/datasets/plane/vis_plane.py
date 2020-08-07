import os

import bstool


if __name__ == '__main__':
    image_dir = './data/buildchange/v1/shanghai/images'
    anno_file = './data/buildchange/v1/coco/annotations/buildchange_v1_train_shanghai_oriented_line.json'
    # anno_file = '/data/hrsc/v1/coco/annotations/hrsc_v1_trainval.json'
    # image_dir = '/data/hrsc/v1/trainval/images'

    coco_parser = bstool.COCOParse(anno_file)

    for image_name in os.listdir(image_dir):
        anns = coco_parser(image_name)

        image_file = os.path.join(image_dir, image_name)
        bstool.show_coco_mask(coco_parser.coco, image_file, anns)
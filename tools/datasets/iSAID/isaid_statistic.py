import wwtool


coco_dota_class = {}

ann_file_name = ['dota', 'trainval', 'v1', '1.0', 'best_keypoint']
# ann_file_name.append('small_object')
ann_file = '/data/iSAID/iSAID_val.json'

size_measure_by_ratio = False
if size_measure_by_ratio == False:
    size_set = [4*4, 8*8, 16*16, 32*32, 64*64, 64*64]
    label_set = ["4*4", "8*8", "16*16", "32*32", "64*64", "64*64-inf"]
else:
    size_set = [0.12/100, 1.08/100, 9.72/100]
    label_set = ["0.12/100", "1.08/100", "9.72/100"]

statistic = wwtool.COCO_Statistic(ann_file, 
                                  size_set=size_set, 
                                  label_set=label_set, 
                                  size_measure_by_ratio=size_measure_by_ratio, 
                                  class_instance=None,
                                  min_area=36,
                                  max_small_length=8)

statistic.mask_ratio()
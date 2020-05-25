import wwtool

# coco_small_class = {    1: 'airplane', 
#                        2: 'bridge', 
#                        3: 'storage-tank', 
#                        4: 'ship', 
#                        5: 'swimming-pool', 
#                        6: 'tennis-court', 
#                        7: 'vehicle', 
#                        8: 'person', 
#                        9: 'harbor', 
#                        10: 'wind-mill'}

coco_dota_class = {}

ann_file_name = ['dota', 'trainval', 'v1', '1.0', 'best_keypoint']
# ann_file_name.append('small_object')
ann_file = './data/{}/v1/coco/annotations/{}.json'.format(ann_file_name[0], '_'.join(ann_file_name))

size_measure_by_ratio = False
if size_measure_by_ratio == False:
    size_set = [4*4, 8*8, 16*16, 32*32, 64*64, 64*64]
    label_set = ["4*4", "8*8", "16*16", "32*32", "64*64", "64*64-inf"]
else:
    size_set = [0.12/100, 1.08/100, 9.72/100]
    label_set = ["0.12/100", "1.08/100", "9.72/100"]

class_instance = wwtool.Small()

statistic = wwtool.COCO_Statistic(ann_file, 
                                  size_set=size_set, 
                                  label_set=label_set, 
                                  size_measure_by_ratio=size_measure_by_ratio, 
                                  class_instance=wwtool.DOTA(),
                                  min_area=36,
                                  max_small_length=8)

for pie_flag in [False, True]:
    statistic.total_size_distribution(plot_pie=pie_flag, save_file_name=ann_file_name[:])

for number_flag in [False, True]:
    statistic.class_size_distribution(coco_class=None, save_file_name=ann_file_name[:], number=number_flag)
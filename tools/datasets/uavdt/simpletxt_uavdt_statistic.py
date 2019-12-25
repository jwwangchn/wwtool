import wwtool

coco_uavdt_class = {1: 'ped', 
                       2: 'person_on_vhcl', 
                       3: 'car', 
                       4: 'bicycle', 
                       5: 'mbike', 
                       6: 'non_mot_vhcl', 
                       7: 'static_person', 
                       8: 'distractor', 
                       9: 'occluder', 
                       10: 'occluder_on_grnd', 
                       11: 'occluder_full', 
                       12: 'occluder_full', 
                       13: 'reflection'}

ann_file_name = ['uavdt', 'trainval_test', 'v1', '1.0']
ann_file_name.append('small_object')
ann_file = './data/{}/v1/coco/annotations/{}.json'.format(ann_file_name[0], '_'.join(ann_file_name))

size_measure_by_ratio = False
if size_measure_by_ratio == False:
    size_set = [4*4, 8*8, 16*16, 32*32, 64*64, 128*128, 256*256]
    label_set = ["4*4", "8*8", "16*16", "32*32", "64*64", "128*128", "256*256"]
else:
    size_set = [0.12/100, 1.08/100, 9.72/100]
    label_set = ["0.12/100", "1.08/100", "9.72/100"]

dior_statistic = wwtool.COCO_Statistic(ann_file, size_set=size_set, label_set=label_set, size_measure_by_ratio=size_measure_by_ratio)

for pie_flag in [False, True]:
    dior_statistic.total_size_distribution(plot_pie=pie_flag, save_file_name=ann_file_name[:])
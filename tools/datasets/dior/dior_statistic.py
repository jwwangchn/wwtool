import wwtool

coco_dior_class = {1: 'airplane', 
                       2: 'airport', 
                       3: 'baseballfield', 
                       4: 'basketballcourt', 
                       5: 'bridge', 
                       6: 'chimney', 
                       7: 'dam', 
                       8: 'Expressway-Service-area', 
                       9: 'Expressway-toll-station', 
                       10: 'golffield', 
                       11: 'groundtrackfield', 
                       12: 'harbor', 
                       13: 'overpass', 
                       14: 'ship', 
                       15: 'stadium', 
                       16: 'storagetank', 
                       17: 'tenniscourt', 
                       18: 'trainstation', 
                       19: 'vehicle', 
                       20: 'windmill'}

ann_file_name = ['dior', 'trainval_test', 'v1', '1.0']
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

# for pie_flag in [False, True]:
#     dior_statistic.total_size_distribution(plot_pie=pie_flag, save_file_name=ann_file_name[:])

dior_statistic.class_size_distribution(coco_class=coco_dior_class, save_file_name=ann_file_name[:])
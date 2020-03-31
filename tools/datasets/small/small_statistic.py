import wwtool

coco_small_class = {   1: 'airplane', 
                       2: 'bridge', 
                       3: 'storage-tank', 
                       4: 'ship', 
                       5: 'swimming-pool', 
                       6: 'vehicle', 
                       7: 'person', 
                       8: 'wind-mill'}

ann_file_name = ['small', 'val', 'v1', '1.0']
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

statistic = wwtool.COCO_Statistic(ann_file, size_set=size_set, label_set=label_set, size_measure_by_ratio=size_measure_by_ratio, class_instance=class_instance, show_title=False)

for pie_flag in [False, True]:
    statistic.total_size_distribution(plot_pie=pie_flag, save_file_name=ann_file_name[:])

for number_flag in [False, True]:
    statistic.class_size_distribution(coco_class=coco_small_class, save_file_name=ann_file_name[:], number=number_flag)

statistic.image_object_num_distribution(save_file_name=ann_file_name[:])

statistic.object_aspect_ratio_distribution(save_file_name=ann_file_name[:])

# statistic.class_num_per_image(coco_class=coco_dior_class, save_file_name=ann_file_name[:])
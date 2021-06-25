import wwtool
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

coco_small_class = {   1: 'building'}

ann_file_name = ['shanghai', 'trian', 'v1', '1.0']
# ann_file_name.append('small_object')
ann_file = '/data/buildchange/public/20201028/coco/annotations/buildchange_public_20201028_val_shanghai_xian_minarea_500.json'

class_instance = wwtool.Small()

statistic = wwtool.COCO_Statistic(ann_file, show_title=False)

for pie_flag in [False, True]:
    statistic.total_size_distribution(plot_pie=pie_flag, save_file_name=ann_file_name[:])

statistic.image_object_num_distribution(save_file_name=ann_file_name[:])

statistic.object_aspect_ratio_distribution(save_file_name=ann_file_name[:])

# statistic.class_num_per_image(coco_class=coco_dior_class, save_file_name=ann_file_name[:])
import os
import shutil

dst_image_path = '/media/jwwangchn/Jwwang134/100-Work/190-Datasets/UCAS-AOD/v1/merge/images'
dst_label_path = '/media/jwwangchn/Jwwang134/100-Work/190-Datasets/UCAS-AOD/v1/merge/labels'

src_paths = ['/media/jwwangchn/Jwwang134/100-Work/190-Datasets/UCAS-AOD/v0/CAR', '/media/jwwangchn/Jwwang134/100-Work/190-Datasets/UCAS-AOD/v0/PLANE']

idx = 1

for src_path in src_paths:
    label = src_path.split('/')[-1].lower()
    src_image_path = os.path.join(src_path, 'images')
    src_label_path = os.path.join(src_path, 'labels')

    for file_name in os.listdir(src_image_path):

        src_label_file = os.path.join(src_label_path, file_name.split('.')[0] + '.txt')
        src_image_file = os.path.join(src_image_path, file_name.split('.')[0] + '.png')
        
        dst_label_file_name = str(label) + "_{:0>4d}".format(idx) + '.txt'
        dst_label_file = os.path.join(dst_label_path, dst_label_file_name)
        dst_image_file_name = str(label) + "_{:0>4d}".format(idx) + '.png'
        dst_image_file = os.path.join(dst_image_path, dst_image_file_name)

        shutil.copy(src_label_file, dst_label_file)
        shutil.copy(src_image_file, dst_image_file)

        idx += 1

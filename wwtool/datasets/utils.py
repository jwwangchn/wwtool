import os
import numpy as np
import shutil

import wwtool


def shuffle_dataset(origin_dataset_dir, trainval_dir, test_dir, trainval_rate=0.8, image_format='.png', label_format='.txt', seed=0):
    """Generate trainval and test sets from origin set by copying files randomly.
    
    Arguments:
        origin_dataset_dir {str} -- path of origin dataset, contains `images` and `labels` folds
        trainval_dir {str} -- path of trainval set, contains `images` and `labels` folds
        test_dir {str} -- path of test set, contains `images` and `labels` folds
        seed {int} -- seed of random function
    
    Returns:
        None
    """
    np.random.seed(seed)
    src_label_path = os.path.join(origin_dataset_dir, 'labels')
    src_image_path = os.path.join(origin_dataset_dir, 'images')

    trainval_dst_label_path = os.path.join(trainval_dir, 'labels')
    wwtool.mkdir_or_exist(trainval_dst_label_path)
    trainval_dst_image_path = os.path.join(trainval_dir, 'images')
    wwtool.mkdir_or_exist(trainval_dst_image_path)

    test_dst_label_path = os.path.join(test_dir, 'labels')
    wwtool.mkdir_or_exist(test_dst_label_path)
    test_dst_image_path = os.path.join(test_dir, 'images')
    wwtool.mkdir_or_exist(test_dst_image_path)

    file_names = [label_file.split('.txt')[0] for label_file in os.listdir(src_label_path)]
    file_names = sorted(file_names)
    np.random.shuffle(file_names)

    trainval_file_names = file_names[0 : int(len(file_names) * trainval_rate)]
    test_file_names = file_names[int(len(file_names) * trainval_rate):]

    for trainval_file_name in trainval_file_names:
        print("From {} to {}.".format(os.path.join(src_label_path, trainval_file_name), os.path.join(trainval_dst_label_path, trainval_file_name)))
        shutil.copy(os.path.join(src_label_path, trainval_file_name + label_format), os.path.join(trainval_dst_label_path, trainval_file_name + label_format))
        shutil.copy(os.path.join(src_image_path, trainval_file_name + image_format), os.path.join(trainval_dst_image_path, trainval_file_name + image_format))

    for test_file_name in test_file_names:
        print("From {} to {}.".format(os.path.join(src_label_path, test_file_name), os.path.join(test_dst_label_path, test_file_name)))
        shutil.copy(os.path.join(src_label_path, test_file_name + label_format), os.path.join(test_dst_label_path, test_file_name + label_format))
        shutil.copy(os.path.join(src_image_path, test_file_name + image_format), os.path.join(test_dst_image_path, test_file_name + image_format))

import os
import numpy as np
from collections import defaultdict

import wwtool


if __name__ == '__main__':
    core_dataset_name = 'buildchange'
    src_version = 'v0'
    # imagesets = ['shanghai']
    # sub_imageset_folds = {'shanghai': ['arg']}
    imagesets = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    sub_imageset_folds = {'beijing': ['arg', 'google', 'ms', 'tdt'],
                            'chengdu': ['arg', 'google', 'ms', 'tdt'],
                            'haerbin': ['arg', 'google', 'ms'],
                            'jinan': ['arg', 'google', 'ms', 'tdt'],
                            'shanghai': ['arg', 'google', 'ms', 'tdt', 'PHR2016', 'PHR2017']}

    seed = 0
    train_rate = 0.8
    imagename_sets = defaultdict(set)

    np.random.seed(seed)

    for imageset in imagesets:
        for sub_imageset_fold in sub_imageset_folds[imageset]:
            print('Processing {} {}'.format(imageset, sub_imageset_fold))
            image_path = './data/{}/{}/{}/{}/images'.format(core_dataset_name, src_version, imageset, sub_imageset_fold)
            if image_path.endswith('.jpg') or image_path.endswith('.png'):
                pass
            else:
                continue
            for image_fn in os.listdir(image_path):
                image_basename = wwtool.get_basename(image_fn)
                imagename_sets[imageset].add(image_basename)

        file_names = list(dict(imagename_sets)[imageset])
        file_names = sorted(file_names)
        np.random.shuffle(file_names)
        train_file_names = file_names[0 : int(len(file_names) * train_rate)]
        val_file_names = file_names[int(len(file_names) * train_rate):]

        save_trainset_fn = './data/{}/{}/{}/trainset.txt'.format(core_dataset_name, src_version, imageset)
        save_valset_fn = './data/{}/{}/{}/valset.txt'.format(core_dataset_name, src_version, imageset)

        for save_fn, file_names in zip([save_trainset_fn, save_valset_fn], [train_file_names, val_file_names]):
            with open(save_fn, 'w') as f:
                for file_name in file_names:
                    f.write('{}\n'.format(file_name))
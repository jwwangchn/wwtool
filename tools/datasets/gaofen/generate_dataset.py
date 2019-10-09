import wwtool

origin_dataset_dir = '/data/gaofen/v1/origin'
trainval_dir = '/data/gaofen/v1/trainval'
test_dir = '/data/gaofen/v1/test'
wwtool.shuffle_dataset(origin_dataset_dir, trainval_dir, test_dir) 
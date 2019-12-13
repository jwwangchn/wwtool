import wwtool

origin_dataset_dir = './data/UCAS-AOD/v1/merge'
trainval_dir = './data/UCAS-AOD/v1/trainval'
test_dir = './data/UCAS-AOD/v1/test'
wwtool.shuffle_dataset(origin_dataset_dir, trainval_dir, test_dir, trainval_rate=0.7351)

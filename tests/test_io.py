import wwtool

result_file = './tests/data/coco_results_file_test.pkl.bbox.json'
anno_file = './tests/data/annotation_file_test.json'

det_data = wwtool.load_detection_results(result_file, anno_file)

print(det_data)
import solaris

roof_gt = '/data/buildchange/v2/xian_fine/xian_fine_roof_gt.csv'
footprint_gt = '/data/buildchange/v2/xian_fine/xian_fine_footprint_gt.csv'

roof_pred = '/home/jwwangchn/Documents/100-Work/170-Codes/aidet/results/buildchange/bc_v014_mask_rcnn_hrnetv2p_w32_v2_roof_trainval/result_roof.csv'
footprint_pred = '/home/jwwangchn/Documents/100-Work/170-Codes/aidet/results/buildchange/bc_v014_mask_rcnn_hrnetv2p_w32_v2_roof_trainval/result_footprint.csv'

a, b = solaris.eval.challenges.spacenet_buildings_2(roof_pred, roof_gt)
print("F1: {}, Precision: {} Recall: {}".format(b['F1Score'].mean(), b['Precision'].mean(), b['Recall'].mean()))

a, b = solaris.eval.challenges.spacenet_buildings_2(footprint_pred, footprint_gt)
print("F1: {}, Precision: {} Recall: {}".format(b['F1Score'].mean(), b['Precision'].mean(), b['Recall'].mean()))
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':

    valset_file = 'tools/datasets/sn6/val.csv'

    valset_data = pd.read_csv(valset_file)
    valset_data = valset_data.dropna(axis=0)
    valset = []

    for idx in range(valset_data.shape[0]):
        valset.append(valset_data.iloc[idx, 0])


    trainval_label_file = './data/sn6/v1/train/labels/SN6_Train_AOI_11_Rotterdam_Buildings.csv'
    anno_data = pd.read_csv(trainval_label_file)
    anno_data = anno_data.dropna(axis=0)
    annos = anno_data

    trainset_firstfile = True
    trainset_csv_file = '/home/jwwangchn/Documents/100-Work/170-Codes/wwdetection/data/sn6/v1/train/labels/trainset_ground_truth.csv'
    valset_firstfile = True
    valset_csv_file = '/home/jwwangchn/Documents/100-Work/170-Codes/wwdetection/data/sn6/v1/train/labels/valset_ground_truth.csv'
    for idx in tqdm(range(annos.shape[0])):
        data_frame = pd.DataFrame({'ImageId': annos.iloc[idx, 0],
                                    'TileBuildingId': annos.iloc[idx, 1],
                                    'PolygonWKT_Pix': annos.iloc[idx, 2],
                                    'Mean_Building_Height': annos.iloc[idx, 3],
                                    'Median_Building_Height': annos.iloc[idx, 4],
                                    'StdDev_Building_Height': annos.iloc[idx, 5]}, index=[0])

        image_full_name = 'SN6_Train_AOI_11_Rotterdam_SAR-Intensity_' + annos.iloc[idx, 0] + '.tif'
        # print(image_full_name)
        if image_full_name in valset:
            if valset_firstfile:
                valset_csv = data_frame
                valset_firstfile = False
            else:
                valset_csv = valset_csv.append(data_frame)
        else:
            if trainset_firstfile:
                trainset_csv = data_frame
                trainset_firstfile = False
            else:
                trainset_csv = trainset_csv.append(data_frame)
    
    valset_csv.to_csv(valset_csv_file, index=False)
    trainset_csv.to_csv(trainset_csv_file, index=False)
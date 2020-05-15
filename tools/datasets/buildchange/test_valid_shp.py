import os
import geopandas as gpd

trainset = 'train_shanghai'

shp_path = './data/buildchange/v0/{}/shp_4326'.format(trainset)

for shp_fn in os.listdir(shp_path):
    if not shp_fn.endswith('shp'):
        continue

    shp_fn = os.path.join(shp_path, shp_fn)
    try:
        shp = gpd.read_file(shp_fn, encoding='utf-8')
    except:
        print("\nCan't open this shp file: {}".format(shp_fn))
        continue

    ori_polygon_list = []
    ori_floor_list = []

    for idx, row_data in shp.iterrows():
        polygon = row_data.geometry
        try:
            floor = row_data.Floor
        except:
            print("\nThis file does not floor key: {}".format(shp_fn))
            continue
        
    

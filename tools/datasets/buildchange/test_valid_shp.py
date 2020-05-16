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
        print("Can't open this shp file: {}".format(shp_fn))
        continue        
    

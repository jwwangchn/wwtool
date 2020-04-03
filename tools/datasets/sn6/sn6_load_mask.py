import solaris as sol
import os
import skimage
from matplotlib import pyplot as plt

image = skimage.io.imread('/data/sn6/v1/samples/images/SN6_Train_AOI_11_Rotterdam_PS-RGB_20190804111224_20190804111453_tile_8691.tif')
f, axarr = plt.subplots(figsize=(10, 10))
plt.imshow(image, cmap='gray')

fp_mask = sol.vector.mask.footprint_mask(df='/data/sn6/v0/MultiSensorSample/geojson_buildings/SN6_Train_AOI_11_Rotterdam_Buildings_20190804111224_20190804111453_tile_8691.geojson',
                                      reference_im='/data/sn6/v1/samples/images/SN6_Train_AOI_11_Rotterdam_PS-RGB_20190804111224_20190804111453_tile_8691.tif')
f, ax = plt.subplots(figsize=(10, 10))
plt.imshow(fp_mask, cmap='gray')

plt.show()
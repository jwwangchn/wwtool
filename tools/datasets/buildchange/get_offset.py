import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import Polygon

import wwtool


class GetRoofOffset():
    def __init__(self,
                rgb_file,
                foot_shp_file,
                geo_info,
                pred_segmentation_file,
                side_shp_file,
                pixel_anno):
        self.rgb_file = rgb_file
        self.foot_shp_file = foot_shp_file
        self.geo_info = geo_info
        self.pred_segmentation_file = pred_segmentation_file
        self.side_shp_file = side_shp_file
        self.pixel_anno = pixel_anno
        mask_parser = wwtool.MaskParse()
        objects = mask_parser(pixel_anno, category=255)
        self.ignore_polygons = [obj['polygon'] for obj in objects]

        self.rgb_image = cv2.imread(rgb_file)
        shp_parser = wwtool.ShpParse()
        objects = shp_parser(foot_shp_file, geo_info)
        self.foot_mask = cv2.imread(geo_info, 0)        # annotated by floor heigh
        self.foot_polygons = [obj['converted_polygon'] for obj in objects]
        self.floors = [obj['converted_property']['Floor'] for obj in objects]
        self.pred_segmentation = cv2.imread(pred_segmentation_file, 0)
        try:
            side_shp = geopandas.read_file(side_shp_file, encoding='utf-8')
            self.side_annotations = [side_coor for _, side_coor in side_shp.iterrows()]
        except:
            print("Can't open this side shp file: {}".format(side_shp_file))
        self.offset_and_floorheigh_list = []

    def get_mean_offset(self):
        ret = cv2.findContours(self.foot_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        foot_contours = ret[0] if len(ret) == 2 else ret[1]     # opencv3 or opencv4

        offsets = []
        for side_anno in self.side_annotations:
            foot_coordinate = int(side_anno[1].coords[0][0]), int(abs(side_anno[1].coords[0][1]))
            foot_root_vector0 = int(side_anno[1].coords[1][0]), int(abs(side_anno[1].coords[1][1]))
            foot_root_vector1 = int(side_anno[1].coords[2][0]), int(abs(side_anno[1].coords[2][1]))

            if self.foot_mask[foot_coordinate[0], foot_coordinate[1]] > 0:
                floor_height = self.foot_mask[foot_coordinate[0], foot_coordinate[1]]
            else:
                distances = []
                for cnt in foot_contours:
                    dist = cv2.pointPolygonTest(cnt, foot_coordinate, True)
                    distances.append(dist)
                floor_height = self.foot_mask[
                    foot_contours[np.argmax(np.array(distances))][0][0][1], foot_contours[np.argmax(np.array(distances))][0][0][0]]
                
            offset_vector = (foot_root_vector1[0] - foot_root_vector0[0], foot_root_vector1[1] - foot_root_vector0[1])
            floor_height = floor_height if floor_height > 0 else 0.0001
            offset_one_floor = offset_vector / floor_height
            
            self.offset_and_floorheigh_list.append([offset_vector, floor_height])    
            offsets.append(offset_one_floor)
        
        offsets = list(filter(lambda x: abs(x[0]) < 5 and abs(x[1]) < 5, offsets))
        mean_offset = np.mean(np.array(offsets), axis=0)

        print("The mean offset of {} is {}".format(self.rgb_file, mean_offset))

        return mean_offset

    def cleaning_data_by_pred(self):
        # TODO:
        _, binary = cv2.threshold(self.pred_segmentation, 0, 250, cv2.THRESH_BINARY)
        num_labels, labels = cv2.connectedComponents(binary, 
                                                     connectivity=4, 
                                                     ltype=cv2.CV_32S)
        build_connection_coord_list = []
        for index_predbuild in range(0, num_labels):
            coors = np.where(labels == index_predbuild)[1], np.where(labels == index_predbuild)[0]
            coors = np.array(coors).T + [0, 0]
            coors_set = set([tuple(_) for _ in coors])
            build_connection_coord_list.append(coors_set)
        
        for floor_heigh in self.foot_mask:
            if floor_heigh != 0:
                pass
    
    def __call__(self, show=False):
        # mean_offset = self.get_mean_offset()
        # if not (str(mean_offset) == 'nan'):
        #     if -10 < mean_offset[0] < 10 and -10 < mean_offset[1] < 10:
        #         pass
        pass
                


if __name__ == '__main__':
    rgb_file = '/data/buildchange/v0/shanghai/images/L18_106968_219320.jpg'
    foot_shp_file = '/data/buildchange/v0/shanghai/merged_shp/L18_106968_219320.shp'
    geo_info = '/data/buildchange/v0/shanghai/geo_info/L18_106968_219320.png'
    pred_segmentation_file = '/data/buildchange/v0/shanghai/pred/L18_106968_219320.png'
    side_shp_file = '/data/buildchange/v0/shanghai/side_shp/L18_106968_219320.shp'
    pixel_anno = '/data/buildchange/v0/shanghai/anno_v2/L18_106968_219320.png'

    get_roof_offset = GetRoofOffset(rgb_file=rgb_file,
                                    foot_shp_file=foot_shp_file,
                                    geo_info=geo_info,
                                    pred_segmentation_file=pred_segmentation_file,
                                    side_shp_file=side_shp_file,
                                    pixel_anno=pixel_anno)

    get_roof_offset(show=True)
import os
import json
import csv
import re
import cv2
import geojson
import shapely.wkt
import numpy as np
import wwtool
from tqdm import tqdm
import pandas as pd
import networkx as nx

from collections import defaultdict
import xml.etree.ElementTree as ET
from pycocotools import mask
from PIL import Image

import rasterio as rio
import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union, nearest_points
import geopandas as gpd

from skimage import measure                        # (pip install scikit-image)

def voc_parse(label_file):
    """parse VOC style dataset label file
    
    Arguments:
        label_file {str} -- label file path
    
    Returns:
        dict, {'bbox': [xmin, ymin, xmax, ymax], 'label': class_name} -- objects' location and class
    """
    tree = ET.parse(label_file)
    root = tree.getroot()
    objects = []
    for single_object in root.findall('object'):
        bndbox = single_object.find('bndbox')
        object_struct = {}

        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        object_struct['bbox'] = [xmin, ymin, xmax, ymax]
        object_struct['label'] = single_object.find('name').text
        
        objects.append(object_struct)
    return objects


def rovoc_parse(label_file):
    """parse rotation VOC style dataset label file
    
    Arguments:
        label_file {str} -- label file path
    
    Returns:
        dict, {'bbox': [cx, cy, w, h, theta (rad/s)], 'label': class_name} -- objects' location and class
    """
    tree = ET.parse(label_file)
    root = tree.getroot()
    objects = []
    for single_object in root.findall('object'):
        robndbox = single_object.find('robndbox')
        object_struct = {}

        cx = float(robndbox.find('cx').text)
        cy = float(robndbox.find('cy').text)
        w = float(robndbox.find('w').text)
        h = float(robndbox.find('h').text)
        theta = float(robndbox.find('angle').text)

        object_struct['bbox'] = [cx, cy, w, h, theta]
        object_struct['label'] = single_object.find('name').text
        
        objects.append(object_struct)
    return objects

def simpletxt_parse(label_file):
    """parse simpletxt style dataset label file
    
    Arguments:
        label_file {str} -- label file path
    
    Returns:
        dict, {'bbox': [...], 'label': class_name} -- objects' location and class
    """
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    objects = []
    basic_label_str = " "
    for line in lines:
        object_struct = dict()
        line = line.rstrip().split(' ')
        label = basic_label_str.join(line[4:])
        bbox = [float(_) for _ in line[0:4]]
        object_struct['bbox'] = bbox
        object_struct['label'] = label
        objects.append(object_struct)
    
    return objects

def dota_parse(label_file):
    """parse dota style dataset label file
    
    Arguments:
        label_file {str} -- label file path
    
    Returns:
        dict, {'bbox': [...], 'label': class_name} -- objects' location and class
    """
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    objects = []
    for line in lines:
        object_struct = dict()
        line = line.rstrip().split(' ')
        label = line[8]
        pointobb = [float(xy) for xy in line[:8]]
        bbox = wwtool.pointobb2bbox(pointobb)
        object_struct['bbox'] = bbox
        object_struct['label'] = label
        object_struct['pointobb'] = pointobb
        objects.append(object_struct)
    
    return objects

class XVIEW_PARSE():
    def __init__(self, json_file, xview_class_labels_file):
        with open(json_file) as f:
            data = json.load(f)

        self.coords = np.zeros((len(data['features']), 4))
        self.image_names = np.zeros((len(data['features'])), dtype="object")
        self.classes = np.zeros((len(data['features'])))

        for i in tqdm(range(len(data['features']))):
            if data['features'][i]['properties']['bounds_imcoords'] != []:
                b_id = data['features'][i]['properties']['image_id']
                val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
                self.image_names[i] = b_id
                self.classes[i] = data['features'][i]['properties']['type_id']
                if val.shape[0] != 4:
                    print("Issues at %d!" % i)
                else:
                    self.coords[i] = val
            else:
                self.image_names[i] = 'None'

        self.labels = {}
        with open(xview_class_labels_file) as f:
            for row in csv.reader(f):
                self.labels[int(row[0].split(":")[0])] = row[0].split(":")[1]

        print("Finish to load xView json file!")
    
    def xview_parse(self, image_name):
        """bbox -> [xmin, ymin, xmax, ymax]
        
        Arguments:
            image_name {str} -- image file name
        
        Returns:
            objects {bboxes, labels} -- object dict
        """
        bboxes = self.coords[self.image_names == image_name]
        labels = self.classes[self.image_names == image_name].astype(np.int64)

        objects = []
        for bbox, label in zip(bboxes, labels):
            object_struct = dict()
            object_struct['bbox'] = bbox
            object_struct['label'] = self.labels[label]
            objects.append(object_struct)
        
        return objects

def visdrone_parse(label_file):
    """parse visdrone style dataset label file
    
    Arguments:
        label_file {str} -- label file path 
        (<bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>, <object_category>, <truncation>, <occlusion>)
    
    Returns:
        dict, {'bbox': [xmin, ymin, xmax, ymax], 'label': class_name} -- objects' location and class
    """
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    objects = []
    for line in lines:
        object_struct = dict()
        line = line.rstrip().split(',')
        label = line[5]
        bbox = [float(_) for _ in line[0:4]]
        object_struct['bbox'] = wwtool.xywh2xyxy(bbox)
        object_struct['label'] = label
        if object_struct['label'] == '0' or object_struct['label'] == '11':
            continue
        objects.append(object_struct)
    
    return objects

def nwpu_parse(label_file):
    """parse nwpu style dataset label file
    
    Arguments:
        label_file {str} -- label file path 
        (<bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>, <object_category>, <truncation>, <occlusion>)
    
    Returns:
        dict, {'bbox': [xmin, ymin, xmax, ymax], 'label': class_name} -- objects' location and class
    """
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    objects = []
    for line in lines:
        object_struct = dict()
        line = line.rstrip().split(',')
        if line[0] == '':
            break
        label = re.sub("\D", "", line[4])
        bbox = [float(re.sub("\D", "", _)) for _ in line[0:4]]
        object_struct['bbox'] = wwtool.xywh2xyxy(bbox)
        object_struct['label'] = label
        if object_struct['label'] == '0' or object_struct['label'] == '11':
            continue
        objects.append(object_struct)
    
    return objects

class UAVDT_PARSE():
    """
    <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<out-of-view>,<occlusion>,<object_category>
    """
    def __init__(self, label_fold):
        self.uavdt_objects = dict()
        for label_file in os.listdir(label_fold):
            seq_name = label_file.split('_')[0]
            with open(os.path.join(label_fold, label_file), 'r') as f:
                lines = f.readlines()

            single_seq_objects = dict()
            for line in lines:
                image_idx, object_id, xmin, ymin, w, h, oov, occlusion, category = [int(_) for _ in line.rstrip().split(',')]
                xmax = xmin + w
                ymax = ymin + h
                if "img{:0>6d}".format(image_idx) not in single_seq_objects:
                    single_seq_objects["img{:0>6d}".format(image_idx)] = []
                else:
                    single_seq_objects["img{:0>6d}".format(image_idx)].append([xmin, ymin, xmax, ymax, category])
            self.uavdt_objects[seq_name] = single_seq_objects

        print("Finish to load UAVDT txt file!")
    
    def uavdt_parse(self, seq_name, image_name):
        try:
            seq_objects = self.uavdt_objects[seq_name]
            image_index_objects = seq_objects[image_name]
        except KeyError:
            print("Skip this image.")
            return []

        objects = []
        for image_index_object in image_index_objects:
            object_struct = dict()
            object_struct['bbox'] = image_index_object[0:4]
            object_struct['label'] = str(image_index_object[4])
            objects.append(object_struct)
        
        return objects

class StanfordCompusParse():
    """
    <target_id>, <xmin>, <ymin>, <xmax>, <ymax>, <frame_id>, <lost>, <occluded>, <generated>, <label>
    """
    def __init__(self, label_fold):
        self.scenes = ('bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'little', 'nexus', 'quad')
        self.stanford_compus_objects = dict()
        
        for scene_name in self.scenes:
            for video_name in os.listdir(os.path.join(label_fold, scene_name)):
                label_file = os.path.join(label_fold, scene_name, video_name, 'annotations.txt')

                with open(label_file, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    target_id, xmin, ymin, xmax, ymax, frame_id, lost, occluded, generated = [int(_) for _ in line.rstrip().split(' ')[0:-1]]
                    label = line.rstrip().split(' ')[-1].split('"')[1]

                    if lost == 1 or occluded == 1:
                        continue
                    
                    if (scene_name, video_name, frame_id) not in self.stanford_compus_objects:
                        self.stanford_compus_objects[(scene_name, video_name, frame_id)] = []
                    else:
                        self.stanford_compus_objects[(scene_name, video_name, frame_id)].append([xmin, ymin, xmax, ymax, label])
        
        print("Finish to load Stanford Compus txt file!")
    
    def stanford_compus_parse(self, scene_name, video_name, frame_id):
        # print(self.stanford_compus_objects.keys())
        try:
            frame_index_objects = self.stanford_compus_objects[(scene_name, video_name, frame_id)]
        except KeyError:
            print("Skip this image.")
            return []

        objects = []
        for frame_index_object in frame_index_objects:
            object_struct = dict()
            object_struct['bbox'] = frame_index_object[0:4]
            object_struct['label'] = str(frame_index_object[4])
            objects.append(object_struct)
        
        return objects

class AirbusShipParse():
    """
    <target_id>, <xmin>, <ymin>, <xmax>, <ymax>, <frame_id>, <lost>, <occluded>, <generated>, <label>
    """
    def __init__(self, label_file):
        self.airbus_ship_objects = defaultdict(list)

        self.anno_data = pd.read_csv(label_file)
        self.anno_data = self.anno_data.dropna(axis=0)

        for idx in range(self.anno_data.shape[0]):
            image_name = self.anno_data.iloc[idx, 0]
            image_objects = self.anno_data.iloc[idx, 1]

            binary_mask = self.rle_decode(image_objects)
            binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
            bounding_box = mask.toBbox(binary_mask_encoded)
            xmin, ymin, xmax, ymax = wwtool.xywh2xyxy(bounding_box)

            self.airbus_ship_objects[image_name].append([xmin, ymin, xmax, ymax])

    def rle_decode(self, mask_rle, shape=(768, 768)):
        s = mask_rle.split()
        starts =  np.asarray(s[0::2], dtype=int)
        lengths = np.asarray(s[1::2], dtype=int)

        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T

    def airbus_ship_parse(self, image_name):
        try:
            image_objects = self.airbus_ship_objects[image_name]
        except KeyError:
            print("Skip this image.")
            return []

        objects = []
        for image_object in image_objects:
            object_struct = dict()
            object_struct['bbox'] = image_object[0:4]
            object_struct['label'] = "1"
            objects.append(object_struct)
        
        return objects


class SN6Parse():
    """
    <target_id>, <xmin>, <ymin>, <xmax>, <ymax>, <frame_id>, <lost>, <occluded>, <generated>, <label>
    """
    def __init__(self, label_file):
        self.sn6_objects = defaultdict(list)

        self.anno_data = pd.read_csv(label_file)
        self.anno_data = self.anno_data.dropna(axis=0)

        for idx in range(self.anno_data.shape[0]):
            image_name = self.anno_data.iloc[idx, 0]
            wkt = self.anno_data.iloc[idx, 2]
            if wkt == 'POLYGON EMPTY':
                continue
            mask = self.wkt2coord(wkt)

            self.sn6_objects[image_name].append(mask)

    def wkt2coord(self, wkt):
        wkt = shapely.wkt.loads(wkt)
        geo = geojson.Feature(geometry=wkt, properties={})
        coordinate = geo.geometry["coordinates"][0]     # drop the polygon of hole
        mask = []
        for idx, point in enumerate(coordinate):
            if idx == len(coordinate) - 1:
                break
            x, y = point
            mask.append(int(x))
            mask.append(int(y))
        return mask

    def sn6_parse(self, image_name):
        try:
            masks = self.sn6_objects[image_name]
        except KeyError:
            print("Skip this image.")
            return []

        objects = []
        for mask in masks:
            object_struct = dict()
            object_struct['segmentation'] = mask
            object_struct['label'] = "1"
            objects.append(object_struct)
        
        return objects


class ShpParse():
    def _wkt2coord(self, wkt):
        wkt = shapely.wkt.loads(wkt)
        geo = geojson.Feature(geometry=wkt, properties={})
        if geo.geometry == None:
            return []
        coordinate = geo.geometry["coordinates"][0]     # drop the polygon of hole
        mask = []
        for idx, point in enumerate(coordinate):
            if idx == len(coordinate) - 1:
                break
            x, y = point
            mask.append(int(x))
            mask.append(int(y))
        return mask

    def _merge_polygon(self, 
                       polygons, 
                       floors=None, 
                       merge_mode=2,
                       connection_mode='floor',
                       merge_boundary=False):
        if merge_mode == 1:
            merged_polygons = unary_union(polygons)
            merged_floors = []  #TODO: fix the length of merged_floor
        if merge_mode == 2:
            merged_polygons = []
            merged_floors = []

            num_polygon = len(polygons)
            node_list = range(num_polygon)
            link_list = []
            for i in range(num_polygon):
                for j in range(i + 1, num_polygon):
                    polygon1, polygon2 = polygons[i], polygons[j]
                    floor1, floor2 = floors[i], floors[j]
                    if polygon1.is_valid and polygon2.is_valid:
                        inter = polygon1.intersection(polygon2)

                        if connection_mode == 'floor':
                            if (inter.area > 0.0 or inter.geom_type in ["LineString", "MULTILINESTRING"]) and floor1 == floor2:
                                link_list.append([i, j])
                        elif connection_mode == 'line':
                            if inter.geom_type in ["LineString", "MULTILINESTRING"]:
                                link_list.append([i, j])
                        else:
                            raise(RuntimeError("Wrong connection_mode flag: {}".format(connection_mode)))

                        if merge_boundary:
                            if polygon1.area > polygon2.area:
                                difference = polygon1.difference(polygon2)
                                polygons[i] = difference
                            else:
                                difference = polygon2.difference(polygon1)
                                polygons[j] = difference
                    else:
                        continue
            
            G = nx.Graph()
            for node in node_list:
                G.add_node(node, floor=floors[node])

            for link in link_list:
                G.add_edge(link[0], link[1])

            for c in nx.connected_components(G):
                nodeSet = G.subgraph(c).nodes()
                nodeSet = list(nodeSet)

                if len(nodeSet) == 1:
                    single_polygon = polygons[nodeSet[0]]
                    merged_polygons.append(single_polygon)
                    merged_floors.append(floors[nodeSet[0]])
                else:
                    pre_merge = [polygons[node] for node in nodeSet]
                    post_merge = unary_union(pre_merge)
                    if post_merge.geom_type == 'MultiPolygon':
                        for sub_polygon in post_merge:
                            merged_polygons.append(sub_polygon)
                            merged_floors.append(floors[nodeSet[0]])
                    else:
                        merged_polygons.append(post_merge)
                        merged_floors.append(floors[nodeSet[0]])
        
        return merged_polygons, merged_floors

    def __call__(self, 
                shp_fn, 
                geom_img, 
                coord='4326', 
                merge_mode=2,
                connection_mode='floor'):
        """Parse shapefile of building change

        Arguments:
            shp_fn {str} -- file path of shapefile
            geom_img {rasterio type} -- png image contain coordinate information

        Keyword Arguments:
            coord {str} -- coordinate system (default: {'4326'})
            merge_mode {int} -- 1: merge by intersection, 2: merge by rule (default: {2})
            connection_mode {str} -- "line": merge by line intersection, "floor": merge by floor (default: {'floor'})

        Returns:
            [dict] -- dict which contains core information
        """
        try:
            shp = gpd.read_file(shp_fn, encoding='utf-8')
        except:
            print("\nCan't open this shp file: {}".format(shp_fn))
            return []
        ori_polygon_list = []
        ori_floor_list = []
        
        for idx, row_data in shp.iterrows():
            polygon = row_data.geometry
            floor = row_data.Floor

            if polygon == None:
                continue

            ori_polygon_list.append(polygon)
            ori_floor_list.append(floor)

        # merge the splitted building when annotation
        merged_polygon_list, merged_floor_list = self._merge_polygon(ori_polygon_list, 
                                                        ori_floor_list, 
                                                        merge_mode=merge_mode,
                                                        connection_mode=connection_mode)
        # converted coordinate
        converted_polygons = []
        converted_floors = []

        # original coordinate, add these to handle multipolygons after merging
        merged_ori_polygons = []
        merged_ori_floor = []

        for polygon, floor in zip(merged_polygon_list, merged_floor_list):
            if polygon.geom_type == 'Polygon':
                if coord == '4326':
                    polygon_pixel = [(geom_img.index(c[0], c[1])[1], -geom_img.index(c[0], c[1])[0]) for c in polygon.exterior.coords]
                    polygon_pixel = Polygon(polygon_pixel)
                    converted_polygons.append(polygon_pixel)
                    converted_floors.append(floor)
                else:
                    converted_polygons.append(polygon)
                
                merged_ori_polygons.append(polygon)
                merged_ori_floor.append(floor)
            elif polygon.geom_type == 'MultiPolygon':
                for sub_polygon in polygon:
                    if coord == '4326':
                        polygon_pixel = [(geom_img.index(c[0], c[1])[1], -geom_img.index(c[0], c[1])[0]) for c in sub_polygon.exterior.coords]
                        polygon_pixel = Polygon(polygon_pixel)
                        converted_polygons.append(polygon_pixel)
                        converted_floors.append(floor)
                    else:
                        converted_polygons.append(sub_polygon)
                    
                    merged_ori_polygons.append(sub_polygon)
                    merged_ori_floor.append(floor)
            else:
                raise(RuntimeError("type(polygon) = {}".format(type(polygon))))

        if isinstance(converted_polygons, (list, shapely.geometry.multipolygon.MultiPolygon)):
            pass
        else:
            converted_polygons = [converted_polygons]

        masks = []
        final_polygons = []
        final_floors = converted_floors
        for polygon_pixel in converted_polygons:
            final_polygons.append(polygon_pixel)
            wkt  = str(polygon_pixel)
            mask = self._wkt2coord(wkt)
            masks.append(mask)

        objects = []
        for mask, converted_polygon, converted_floor, ori_polygon, ori_floor in zip(masks, final_polygons, final_floors, merged_ori_polygons, merged_ori_floor):
            if mask == []:
                continue
            object_struct = dict()
            mask = [abs(_) for _ in mask]

            xmin, ymin, xmax, ymax = wwtool.pointobb2bbox(mask)
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            object_struct['segmentation'] = mask
            object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
            object_struct['converted_polygon'] = converted_polygon
            object_struct['converted_floor'] = converted_floor
            object_struct['ori_polygon'] = ori_polygon
            object_struct['ori_floor'] = ori_floor
            object_struct['label'] = "1"
            objects.append(object_struct)
        
        return objects


class MaskParse():
    def create_sub_masks(self, mask_image):
        height, width, _ = mask_image.shape
        ret_image = wwtool.generate_image(height, width, color=0)
        gray_mask_image = mask_image[:, :, 0]

        keep_bool = np.logical_or(gray_mask_image == 1, gray_mask_image == 3)

        ret_image[keep_bool] = 1

        return ret_image

    def create_sub_mask_annotation(self, sub_mask):
        contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

        segmentations = []
        polygons = []

        for contour in contours:
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            if contour.shape[0] < 3:
                continue
            
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)
            if poly.geom_type == 'MultiPolygon':
                for poly_ in poly:
                    polygons.append(poly_)
                    segmentation = np.array(poly_.exterior.coords).ravel().tolist()
                    segmentations.append(segmentation)
            else:
                polygons.append(poly)
                segmentation = np.array(poly.exterior.coords).ravel().tolist()
                segmentations.append(segmentation)

        return segmentations, polygons

    def __call__(self, mask_file):
        mask_image = cv2.imread(mask_file)
        ret_image = self.create_sub_masks(mask_image)
        masks, polygons = self.create_sub_mask_annotation(ret_image)

        objects = []
        for mask, polygon in zip(masks, polygons):
            if mask == []:
                continue
            object_struct = dict()
            mask = [abs(_) for _ in mask]

            xmin, ymin, xmax, ymax = wwtool.pointobb2bbox(mask)
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            object_struct['segmentation'] = mask
            object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
            object_struct['polygon'] = polygon
            object_struct['label'] = "1"
            objects.append(object_struct)
        
        return objects

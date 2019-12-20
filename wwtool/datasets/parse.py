import os
import json
import csv
import numpy as np
import wwtool
from tqdm import tqdm
import xml.etree.ElementTree as ET


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
    
    Returns:
        dict, {'bbox': [...], 'label': class_name} -- objects' location and class
    """
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    objects = []
    for line in lines:
        object_struct = dict()
        line = line.rstrip().split(',')
        label = line[5]
        bbox = [float(_) for _ in line[0:4]]
        object_struct['bbox'] = bbox
        object_struct['label'] = label
        if object_struct['label'] == 0 or object_struct['label'] == 11:
            continue
        objects.append(object_struct)
    
    return objects
import os
import numpy as np
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
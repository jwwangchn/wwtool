import os
import numpy as np
import lxml.etree as ET

from wwtool.utils import mkdir_or_exist


def simpletxt_dump(objects, anno_file, encode='bbox'):
    """dump object information to simple txt label files
    
    Arguments:
        objects {dict} -- object information
        label_file {str} -- label file path
    
    Returns:
        None
    """
    with open(anno_file, 'w') as f:
        for obj in objects:
            bbox = obj[encode]
            label = obj['label']
            bbox = [round(_, 3) for _ in map(float, bbox)]
            bbox = ["{:.4f}".format(_) for _ in bbox]
            content = " ".join(map(str, bbox))
            content = content + ' ' + label + '\n'
            f.write(content)

def simple_obb_xml_dump(objects, img_name, save_dir):
    bboxes, rbboxes, pointobbs, labels, scores, areas, number = [], [], [], [], [], [], 0
    for obj in objects:
        bboxes.append(obj['bbox'])
        rbboxes.append(obj['rbbox'])
        pointobbs.append(obj['pointobbs'])
        labels.append(obj['label'])
        scores.append(obj['score'])
        areas.append(obj['rbbox'][2] * obj['rbbox'][3])
        number += 1

    root=ET.Element("annotations")
    ET.SubElement(root, "filename").text = img_name
    ET.SubElement(root,"number").text = str(number)
    
    for idx in range(number):
        object=ET.SubElement(root, "object")
        ET.SubElement(object,"type").text = "robndbox"
        ET.SubElement(object,"name").text = labels[idx]
        ET.SubElement(object,"S").text = str(areas[idx])
        ET.SubElement(object,"score").text = str(scores[idx])

        robndbox=ET.SubElement(object,"robndbox")
        ET.SubElement(robndbox,"cx").text = str(rbboxes[idx][0])
        ET.SubElement(robndbox,"cy").text = str(rbboxes[idx][1])
        ET.SubElement(robndbox,"w").text = str(rbboxes[idx][2])
        ET.SubElement(robndbox,"h").text = str(rbboxes[idx][3])
        ET.SubElement(robndbox,"angle").text = str(rbboxes[idx][4])

        pobndbox=ET.SubElement(object,"pobndbox")
        ET.SubElement(pobndbox,"x1").text=str(pointobbs[idx][0])
        ET.SubElement(pobndbox,"y1").text=str(pointobbs[idx][1])
        ET.SubElement(pobndbox,"x2").text=str(pointobbs[idx][2])
        ET.SubElement(pobndbox,"y2").text=str(pointobbs[idx][3])
        ET.SubElement(pobndbox,"x3").text=str(pointobbs[idx][4])
        ET.SubElement(pobndbox,"y3").text=str(pointobbs[idx][5])
        ET.SubElement(pobndbox,"x4").text=str(pointobbs[idx][6])
        ET.SubElement(pobndbox,"y4").text=str(pointobbs[idx][7])
        
    tree = ET.ElementTree(root)
    tree.write("{}/{}.xml".format(save_dir, img_name), pretty_print=True, xml_declaration=True, encoding='utf-8')
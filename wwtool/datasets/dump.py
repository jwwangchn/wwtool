import os
import numpy as np
import xml.etree.ElementTree as ET

from wwtool.utils import mkdir_or_exist


def simpletxt_dump(objects, file):
    """dump object information to simple txt label files
    
    Arguments:
        objects {dict} -- object information
        label_file {str} -- label file path
    
    Returns:
        None
    """
    with open(file, 'w') as f:
        for obj in objects:
            bbox = obj['bbox']
            label = obj['label']
            bbox = [round(_, 3) for _ in map(float, bbox)]
            bbox = ["{:.4f}".format(_) for _ in bbox]
            content = " ".join(map(str, bbox))
            content = content + ' ' + label + '\n'
            f.write(content)
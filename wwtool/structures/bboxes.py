import numpy as np
from typing import Union, List

class Bboxes:
    """This structure stores a list of boxes as Nx4 (np.ndarray)
    It supports some common methods about bounding boxes

    Default:
        boxes: [xmin, ymin, xmax, ymax] (input and output formats)
        labels: str (input and output)
    """
    def __init__(self):
        self.boxes = np.empty((0, 4))
        self.labels = []

    def append(self, bbox: Union[list, np.ndarray], label: str) -> None:
        """append new bbox to self.boxes
        
        Arguments:
            bbox {list or np.ndarray} -- one bounding box with foramt [xmin, ymin, xmax, ymax]
            label {string} -- label of the new bbox
        """ 
        if isinstance(bbox, list):
            bbox = np.array(bbox)
        
        self.boxes = np.vstack((self.boxes, bbox))
        self.labels.append(label)

    def area(self) -> np.ndarray:
        """Computes the area of all the boxes.
        
        Returns:
            np.ndarray -- a vector with areas of each box.
        """
        area = (self.boxes[:, 2] - self.boxes[:, 0]) * (self.boxes[:, 3] - self.boxes[:, 1])
        return area

    def __len__(self) -> int:
        return self.boxes.shape[0]
import os
import cv2
import mmcv

class Convert2COCO():
    def __init__(self, 
                imgpath=None,
                annopath=None,
                imageset_file=None,
                image_format='.jpg',
                anno_format='.txt',
                data_categories=None,
                data_info=None,
                data_licenses=None,
                data_type="instances",
                groundtruth=True,
                small_object_area=0,
                sub_anno_fold=False):
        super(Convert2COCO, self).__init__()

        self.imgpath = imgpath
        self.annopath = annopath
        self.image_format = image_format
        self.anno_format = anno_format

        self.categories = data_categories
        self.info = data_info
        self.licenses = data_licenses
        self.type = data_type
        self.small_object_area = small_object_area
        self.small_object_idx = 0
        self.groundtruth = groundtruth
        self.max_object_num_per_image = 0
        self.sub_anno_fold = sub_anno_fold
        self.imageset_file = imageset_file

        self.imlist = []
        if self.imageset_file:
            with open(self.imageset_file, 'r') as f:
                lines = f.readlines()
            for img_name in lines:
                img_name = img_name.strip('\n')
                self.imlist.append(img_name)
            print("Loading image names from imageset file, image number: {}".format(len(self.imlist)))
        else:
            for img_name in os.listdir(self.imgpath):
                if img_name.endswith(self.image_format):
                    img_name = img_name.split(self.image_format)[0]
                    self.imlist.append(img_name)
                else:
                    continue
                
    def get_image_annotation_pairs(self):
        images = []
        annotations = []
        index = 0
        progress_bar = mmcv.ProgressBar(len(self.imlist))
        imId = 0
        for name in self.imlist:
            imgpath = os.path.join(self.imgpath, name + self.image_format)
            if self.sub_anno_fold:
                annotpath = os.path.join(self.annopath, name, name + self.anno_format)
            else:
                annotpath = os.path.join(self.annopath, name + self.anno_format)

            annotations_coco = self.__generate_coco_annotation__(annotpath, imgpath)

            # if annotation is empty, skip this annotation
            if annotations_coco != [] or self.groundtruth == False:
                img = cv2.imread(imgpath)
                height, width, channels = img.shape
                images.append({"date_captured": "2019",
                                "file_name": name + self.image_format,
                                "id": imId + 1,
                                "license": 1,
                                "url": "http://jwwangchn.cn",
                                "height": height,
                                "width": width})

                for annotation in annotations_coco:
                    index = index + 1
                    annotation["iscrowd"] = 0
                    annotation["image_id"] = imId + 1
                    annotation["id"] = index
                    annotations.append(annotation)

                imId += 1

            if imId % (len(self.imlist) // 50) == 0:
                print("\nImage ID: {}, Instance ID: {}, Small Object Counter: {}, Max Object Number: {}".format(imId, index, self.small_object_idx, self.max_object_num_per_image))
            
            progress_bar.update()
            

        return images, annotations

    def __generate_coco_annotation__(self, annotpath, imgpath):
        """
        docstring here
            :param self: 
            :param annotpath: the path of each annotation
            :param return: dict()  
        """   
        raise NotImplementedError
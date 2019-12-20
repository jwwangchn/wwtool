import numpy as np
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
from matplotlib import cm


class COCO_STATISTIC():
    def __init__(self, 
                ann_file, 
                size_set=[16*16, 32*32, 96*96], 
                label_set=[], 
                size_measure_by_ratio=False):
        self.ann_file = ann_file
        self.coco = COCO(self.ann_file)
        self.catIds = self.coco.getCatIds(catNms=[''])
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        self.size_set = size_set
        self.label_set = label_set
        self.size_measure_by_ratio = size_measure_by_ratio

    def total_size_distribution(self, plot_pie, save_file_name=[]):
        object_sizes = []
        size_nums = []
        for idx, _ in enumerate(self.imgIds):
            img = self.coco.loadImgs(self.imgIds[idx])[0]
            img_size = img['height'] * img['width']
            annIds = self.coco.getAnnIds(imgIds = img['id'], catIds = self.catIds, iscrowd = None)
            anns = self.coco.loadAnns(annIds)       # per image
            print("idx: {}, image file name: {}".format(idx, img['file_name']))
            for ann in anns:
                bbox = ann['bbox']
                object_size = bbox[2] * bbox[3]
                if self.size_measure_by_ratio:
                    object_sizes.append(object_size/img_size)
                else:
                    object_sizes.append(object_size)
        object_sizes = np.array(object_sizes)
        
        if plot_pie:
            for idx, size_value in enumerate(self.size_set):
                if idx == 0:
                    size_nums.append(np.where(object_sizes <= size_value)[0].shape[0])
                else:
                    size_nums.append(np.where(object_sizes <= size_value)[0].shape[0] - sum(size_nums[:]))

            print("size_nums: {}".format(size_nums))

            # plot
            if len(self.label_set) == 0:
                self.label_set = [str(_) for _ in self.size_set]
            
            fracs = size_nums[:]
            explode = [0] * len(self.size_set)
            explode[0] = 0.1
            plt.axes(aspect=1)
            colors = cm.Set1(np.arange(len(self.size_set)) / float(len(self.size_set)))
            plt.pie(x=fracs, labels=self.label_set, explode=explode, colors=colors, autopct='%3.1f %%', startangle = 0, pctdistance = 0.6)
            plt.title('Size ratio of {}\n{}'.format(save_file_name[0], size_nums))
            save_file_name.append('pie')
        else:
            plt.hist(object_sizes, bins=np.arange(0, 32*32, 32*32//30), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
            print("Total objects: {}".format(object_sizes.shape[0]))
            plt.title('Size Distribution of {}'.format(save_file_name[0]))
            plt.xlabel('size')
            save_file_name.append('hist')
        
        plt.savefig('{}.png'.format("_".join(save_file_name)), bbox_inches='tight', dpi=600, pad_inches=0.1)
        plt.show()

    def class_size_distribution(self):
        pass

    def total_ratio_distribution(self):
        pass
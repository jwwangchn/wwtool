import numpy as np
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
from matplotlib import cm
from collections import defaultdict
import wwtool

plt.rcParams.update({'font.size': 14})
# plt.rcParams["font.family"] = "Times New Roman"

class COCO_Statistic():
    def __init__(self, 
                ann_file, 
                size_set=[16*16, 32*32, 96*96], 
                label_set=[], 
                size_measure_by_ratio=False,
                class_instance=None,
                show_title=False):
        self.ann_file = ann_file
        self.coco = COCO(self.ann_file)
        self.catIds = self.coco.getCatIds(catNms=[''])
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        self.image_num = len(self.imgIds)
        self.size_set = size_set
        self.label_set = label_set
        self.size_measure_by_ratio = size_measure_by_ratio
        self.class_instance = class_instance
        self.show_title = show_title

        categories = self.coco.dataset['categories']
        self.coco_class = dict()
        for category in categories:
            self.coco_class[category['id']] = category['name']

    def total_size_distribution(self, plot_pie, save_file_name=[]):
        plt.clf()
        plt.figure(figsize=(6, 4.5),)
        object_sizes = []
        image_sizes = []
        size_nums = []
        for idx, _ in enumerate(self.imgIds):
            img = self.coco.loadImgs(self.imgIds[idx])[0]
            img_size = img['height'] * img['width']
            annIds = self.coco.getAnnIds(imgIds = img['id'], catIds = self.catIds, iscrowd = None)
            anns = self.coco.loadAnns(annIds)       # per image
            # print("idx: {}, image file name: {}".format(idx, img['file_name']))
            for ann in anns:
                bbox = ann['bbox']
                object_size = bbox[2] * bbox[3]
                if self.size_measure_by_ratio:
                    object_sizes.append(object_size/img_size)
                else:
                    object_sizes.append(object_size)
                image_sizes.append(img_size)
        object_sizes = np.array(object_sizes)
        image_sizes = np.array(image_sizes)
        
        if plot_pie:
            for idx, size_value in enumerate(self.size_set):
                if idx == 0:
                    size_nums.append(np.where(object_sizes <= size_value)[0].shape[0])
                elif idx != len(self.size_set) - 1:
                    size_nums.append(np.where(object_sizes <= size_value)[0].shape[0] - sum(size_nums[:]))
                else:
                    size_nums.append(len(object_sizes) - sum(size_nums[:]))

            print("size_nums: {}".format(size_nums))

            # plot
            if len(self.label_set) == 0:
                self.label_set = [str(_) for _ in self.size_set]
            
            fracs = size_nums[:]
            if False:
                explode = [0] * len(self.size_set)
                explode[0] = 0.1
                plt.axes(aspect=1)
                colors = cm.Set1(np.arange(len(self.size_set)) / float(len(self.size_set)))
                plt.pie(x = fracs, labels = self.label_set, explode = explode, colors = colors, autopct = '%3.1f %%', startangle = 0, pctdistance = 0.6)
            else:
                print("size set: ", self.label_set, "size number: ", fracs)
                print("size set: ", self.label_set, "size ratio: ", np.array(fracs)/sum(fracs))
                plt.bar(self.label_set, fracs, color='dodgerblue', alpha=0.75)
                plt.xticks(range(len(self.label_set)), self.label_set, rotation=0)
            if self.show_title:
                plt.title('Size ratio of {} in {} Dataset\n{}'.format(save_file_name[0], save_file_name[1], size_nums))
            save_file_name.append('pie')
        else:
            object_sizes = np.sqrt(np.array(object_sizes))
            image_sizes = np.sqrt(np.array(image_sizes))
            print("Max size: {}, Min size: {}".format(np.max(object_sizes), np.min(object_sizes)))
            print("Absolute Mean size: {}, Var size: {}".format(np.mean(object_sizes), np.std(object_sizes, ddof=1)))
            print("Relative Mean size: {}, Var size: {}".format(np.mean(object_sizes/image_sizes), np.std(object_sizes/image_sizes)))
            # sns_plot = sns.distplot(object_sizes, color="b", bins=30, kde_kws={"lw": 2})
            plt.hist(object_sizes, bins=np.arange(0, 64, 64//30), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.95)
            print("Total objects: {}".format(object_sizes.shape[0]))
            if self.show_title:
                plt.title('Size Distribution of {} in {} Dataset'.format(save_file_name[0], save_file_name[1]))
            plt.xlabel("Instances' sizes")
            plt.ylabel("Instance Count")
            save_file_name.append('hist')
        
        plt.savefig('{}.pdf'.format("_".join(save_file_name)), bbox_inches='tight', dpi=600, pad_inches=0.1)
        # plt.show()

    def class_size_distribution(self, coco_class=None, save_file_name=[], number=True):
        plt.clf()
        plt.figure(figsize=(6, 4.5))
        class_nums = defaultdict(lambda: 1)
        class_size = defaultdict(list)
        for idx, _ in enumerate(self.imgIds):
            img = self.coco.loadImgs(self.imgIds[idx])[0]

            annIds = self.coco.getAnnIds(imgIds = img['id'], catIds = self.catIds, iscrowd = None)
            anns = self.coco.loadAnns(annIds)       # per image
            # print("idx: {}, image file name: {}".format(idx, img['file_name']))
            for ann in anns:
                if coco_class == None:
                    coco_class = self.coco_class
                class_name = coco_class[ann['category_id']]
                area = np.sqrt(ann['area'])
                class_size[class_name].append(area)
                class_nums[class_name] += 1

        class_size ={key:class_size[key] for key in sorted(class_size)}
        class_nums ={key:class_nums[key] for key in sorted(class_nums)}

        if number:
            bar_name = np.array(list(class_nums.keys()) if self.class_instance == None else self.class_instance.full2abbr(list(class_nums.keys())))
            bar_value = np.array(list(class_nums.values()))
            sorted_indices = np.argsort(bar_value)[::-1]
            plt.bar(bar_name[sorted_indices], bar_value[sorted_indices], color='dodgerblue', alpha=0.75)
            plt.xticks(range(len(bar_name)), bar_name[sorted_indices], rotation=0)
            plt.xlabel('class')
            plt.ylabel('Instance Count')
            plt.yscale('log')
            save_file_name.append('bar')
        else:
            box_name = list(class_size.keys()) if self.class_instance == None else self.class_instance.full2abbr(list(class_size.keys()))
            box_value = class_size.values()
        
            bplot = plt.boxplot(box_value, vert=False, whis=1.5, widths=0.6, patch_artist=True, showfliers=False, showmeans=False, labels=box_name, notch=True)
            # plt.xticks(range(1, len(box_name) + 1), box_name, rotation=0)
            plt.xlabel("Instances' sizes")
            plt.ylabel('Class')

            # fill with colors
            colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            save_file_name.append('box')
        
        print("Classes: {}".format(class_nums))
        if self.show_title:
            plt.title('Class Numbers Distribution of {} in {} Dataset'.format(save_file_name[0], save_file_name[1]))
        
        plt.savefig('{}.pdf'.format("_".join(save_file_name)), bbox_inches='tight', dpi=600, pad_inches=0.1)
        # plt.show()

    def image_object_num_distribution(self, save_file_name=[]):
        plt.clf()
        plt.figure(figsize=(6, 4.5))
        object_nums = [0] * self.image_num

        for idx, _ in enumerate(self.imgIds):
            img = self.coco.loadImgs(self.imgIds[idx])[0]
            annIds = self.coco.getAnnIds(imgIds = img['id'], catIds = self.catIds, iscrowd = None)
            anns = self.coco.loadAnns(annIds)       # per image
            # print("idx: {}, image file name: {}".format(idx, img['file_name']))
            object_nums[idx] = len(anns)
                
        object_nums = np.array(object_nums)
        
        # object_sizes = np.sqrt(np.array(object_sizes))
        plt.hist(object_nums, bins=np.arange(0, 2700, 2700//80), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
        print("Max objects: {}, Image num: {}".format(np.max(object_nums), self.image_num))
        if self.show_title:
            plt.title('Object num Distribution of {} in {} Dataset'.format(save_file_name[0], save_file_name[1]))
        plt.xlim([-20, 2700])
        plt.xlabel('Instances')
        plt.ylabel('Image Count')
        plt.yscale('log')
        save_file_name.append('object_num')
        
        plt.savefig('{}.pdf'.format("_".join(save_file_name)), bbox_inches='tight', dpi=600, pad_inches=0.1)
        # plt.show()

    def object_aspect_ratio_distribution(self, save_file_name=[]):
        plt.clf()
        plt.figure(figsize=(6, 4.5))
        object_aspect_ratios = []
        for idx, _ in enumerate(self.imgIds):
            img = self.coco.loadImgs(self.imgIds[idx])[0]
            annIds = self.coco.getAnnIds(imgIds = img['id'], catIds = self.catIds, iscrowd = None)
            anns = self.coco.loadAnns(annIds)       # per image
            # print("idx: {}, image file name: {}".format(idx, img['file_name']))
            for ann in anns:
                bbox = ann['bbox']
                if bbox[3] == 0:
                    continue
                object_aspect_ratio = bbox[2] / float(bbox[3])
                object_aspect_ratios.append(object_aspect_ratio)
        object_aspect_ratios = np.array(object_aspect_ratios)

        # sns_plot = sns.distplot(object_sizes, color="b", bins=30, kde_kws={"lw": 2})
        plt.hist(object_aspect_ratios, bins=np.arange(0, 3, 3/10), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
        if self.show_title:
            plt.title('Aspect Ratio Distribution of {} in {} Dataset'.format(save_file_name[0], save_file_name[1]))
        plt.xlabel('size')
        save_file_name.append('aspect_ratio')
        
        plt.savefig('{}.pdf'.format("_".join(save_file_name)), bbox_inches='tight', dpi=600, pad_inches=0.1)
        # plt.show()

    def class_num_per_image(self, coco_class=None, save_file_name=[]):
        plt.clf()
        plt.figure(figsize=(6, 4.5))
        class_num_per_image = defaultdict(lambda : 1)
        
        for idx, _ in enumerate(self.imgIds):
            class_set_per_image = set()
            img = self.coco.loadImgs(self.imgIds[idx])[0]

            annIds = self.coco.getAnnIds(imgIds = img['id'], catIds = self.catIds, iscrowd = None)
            anns = self.coco.loadAnns(annIds)       # per image
            # print("idx: {}, image file name: {}".format(idx, img['file_name']))
            for ann in anns:
                if coco_class == None:
                    coco_class = self.coco_class
                class_name = coco_class[ann['category_id']]
                class_set_per_image.add(class_name)
            
            class_num_per_image[str(len(class_set_per_image))] +=1 
        print(class_num_per_image)
        plt.hist(class_num_per_image.values(), bins=np.arange(0, 8, 8/8), histtype='bar', facecolor='dodgerblue', alpha=0.75, rwidth=0.9)
        plt.xlabel('class')
        plt.ylabel('Instance Count')
        plt.yscale('log')
        save_file_name.append('class_num_per_image')
                
        if self.show_title:
            plt.title('Class Numbers Distribution of {} in {} Dataset'.format(save_file_name[0], save_file_name[1]))
        
        plt.savefig('{}.pdf'.format("_".join(save_file_name)), bbox_inches='tight', dpi=600, pad_inches=0.1)
        # plt.show()
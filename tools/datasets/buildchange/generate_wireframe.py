import os
import numpy as np
import shapely
import cv2
import json
from shapely import affinity
from multiprocessing import Pool
from functools import partial

import mmcv
import wwtool

import tqdm
import skimage.draw
from scipy.ndimage import zoom
from itertools import combinations
import matplotlib.pyplot as plt


class GenerateWireframe():
    def __init__(self,
                dst_version,
                city,
                multi_processing=False,
                num_processor=16):
        self.splitted_image_dir = './data/buildchange/{}/{}/images'.format(dst_version, city)
        self.splitted_label_dir = './data/buildchange/{}/{}/labels_json'.format(dst_version, city)
        self.wireframe_dir = './data/buildchange/v2/{}/labels_wireframe'.format(city)
        wwtool.mkdir_or_exist(self.wireframe_dir)
        self.city = city
        self.multi_processing = multi_processing
        self.pool = Pool(num_processor)

    def inrange(self, v, shape):
        return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]

    def to_int(self, x):
        return tuple(map(int, x))

    def save_heatmap(self, prefix, lines):
        im_rescale = (1024, 1024)
        heatmap_scale = (256, 256)

        fy, fx = heatmap_scale[1] / 1024, heatmap_scale[0] / 1024
        jmap = np.zeros((1,) + heatmap_scale, dtype=np.float32)
        joff = np.zeros((1, 2) + heatmap_scale, dtype=np.float32)
        lmap = np.zeros(heatmap_scale, dtype=np.float32)

        lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
        lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
        lines = lines[:, :, ::-1]

        junc = []
        jids = {}

        def jid(jun):
            jun = tuple(jun[:2])
            if jun in jids:
                return jids[jun]
            jids[jun] = len(junc)
            junc.append(np.array(jun + (0,)))
            return len(junc) - 1

        lnid = []
        lpos, lneg = [], []
        for v0, v1 in lines:
            lnid.append((jid(v0), jid(v1)))
            lpos.append([junc[jid(v0)], junc[jid(v1)]])

            vint0, vint1 = self.to_int(v0), self.to_int(v1)
            jmap[0][vint0] = 1
            jmap[0][vint1] = 1
            rr, cc, value = skimage.draw.line_aa(*self.to_int(v0), *self.to_int(v1))
            lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

        for v in junc:
            vint = self.to_int(v[:2])
            joff[0, :, vint[0], vint[1]] = v[:2] - vint - 0.5

        llmap = zoom(lmap, [0.5, 0.5])
        lineset = set([frozenset(l) for l in lnid])
        for i0, i1 in combinations(range(len(junc)), 2):
            if frozenset([i0, i1]) not in lineset:
                v0, v1 = junc[i0], junc[i1]
                vint0, vint1 = self.to_int(v0[:2] / 2), self.to_int(v1[:2] / 2)
                rr, cc, value = skimage.draw.line_aa(*vint0, *vint1)
                lneg.append([v0, v1, i0, i1, np.average(np.minimum(value, llmap[rr, cc]))])


        if len(lneg) == 0:
            return
        lneg.sort(key=lambda l: -l[-1])

        junc = np.array(junc, dtype=np.float32)
        Lpos = np.array(lnid, dtype=np.int)
        Lneg = np.array([l[2:4] for l in lneg][:4000], dtype=np.int)
        lpos = np.array(lpos, dtype=np.float32)
        lneg = np.array([l[:2] for l in lneg[:2000]], dtype=np.float32)


        # plt.subplot(131)
        # plt.imshow(lmap)
        # plt.subplot(132)
        # for i0, i1 in Lpos:
        #     plt.scatter(junc[i0][1] * 4, junc[i0][0] * 4)
        #     plt.scatter(junc[i1][1] * 4, junc[i1][0] * 4)
        #     plt.plot([junc[i0][1] * 4, junc[i1][1] * 4], [junc[i0][0] * 4, junc[i1][0] * 4])
        # plt.subplot(133), plt.imshow(lmap)
        # for i0, i1 in Lneg[:150]:
        #     plt.plot([junc[i0][1], junc[i1][1]], [junc[i0][0], junc[i1][0]])
        # plt.show()

        # For junc, lpos, and lneg that stores the junction coordinates, the last
        # dimension is (y, x, t), where t represents the type of that junction.  In
        # the wireframe dataset, t is always zero.


        np.savez_compressed(
            f"{prefix}_label.npz",
            aspect_ratio=1,
            jmap=jmap,  # [J, H, W]    Junction heat map
            joff=joff,  # [J, 2, H, W] Junction offset within each pixel
            lmap=lmap,  # [H, W]       Line heat map with anti-aliasing
            junc=junc,  # [Na, 3]      Junction coordinate
            Lpos=Lpos,  # [M, 2]       Positive lines represented with junction indices
            Lneg=Lneg,  # [M, 2]       Negative lines represented with junction indices
            lpos=lpos,  # [Np, 2, 3]   Positive lines represented with junction coordinates
            lneg=lneg,  # [Nn, 2, 3]   Negative lines represented with junction coordinates
        )

    def mask2lines(self, mask):
        point_num = len(mask)
        lines = []
        for idx in range(0, point_num - 2, 2):
            line = mask[idx:idx + 4]
            lines.append(line)

        end_line = [mask[-2], mask[-1], mask[0], mask[1]]
        lines.append(end_line)
        return lines


    def json2wireframe(self, image_fn):
        basename = wwtool.get_basename(image_fn)
        json_file = os.path.join(self.splitted_label_dir, basename + '.json')
        
        annotations = mmcv.load(json_file)['annotations']
        image_lines = []
        for annotation in annotations:
            roof_mask = annotation['roof']
            lines = self.mask2lines(roof_mask)
            image_lines += lines

        image_lines = np.array(image_lines).reshape(-1, 2, 2)
        np_save_file_prefix = os.path.join(self.wireframe_dir, basename)
        self.save_heatmap(np_save_file_prefix, image_lines)
            

    def core(self):
        if self.multi_processing:
            image_fn_list = os.listdir(self.splitted_label_dir)
            num_image = len(image_fn_list)
            worker = partial(self.json2wireframe)
            # self.pool.map(worker, image_fn_list)
            ret = list(tqdm.tqdm(self.pool.imap(worker, image_fn_list), total=num_image))
            self.pool.close()
            self.pool.join()
        else:
            image_fn_list = os.listdir(self.splitted_label_dir)
            progress_bar = mmcv.ProgressBar(len(image_fn_list))
            for _, image_fn in enumerate(image_fn_list):
                self.json2wireframe(image_fn)
                progress_bar.update()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


if __name__ == '__main__':
    # cities = ['shanghai', 'beijing', 'jinan', 'haerbin', 'chengdu']
    cities = ['shanghai']
    
    core_dataset_name = 'buildchange'
    dst_version = 'v2'
    sub_img_w, sub_img_h = 1024, 1024


    for city in cities:
        convert = GenerateWireframe(dst_version=dst_version, 
                                    city=city,
                                    multi_processing=True,
                                    num_processor=8)
        convert.core()
        print(f"finish processing {city}")

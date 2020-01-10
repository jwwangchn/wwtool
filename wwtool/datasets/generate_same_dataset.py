import os
import cv2

def generate_same_dataset(src_img_file, 
                          src_anno_file, 
                          dst_img_path, 
                          dst_anno_path, 
                          src_img_format='.png', 
                          src_anno_format='.txt', 
                          dst_img_format='.png', 
                          dst_anno_format='.txt', 
                          parse_fun=None, 
                          dump_fun=None, 
                          save_image=True):
    # process image
    if save_image:
        src_img_name = src_img_file.split('/')[-1]
        img = cv2.imread(src_img_file)
        dst_img_name = src_img_name.split(src_img_format)[0] + dst_img_format
        dst_img_file = os.path.join(dst_img_path, dst_img_name)
        cv2.imwrite(dst_img_file, img)

    # process label
    src_anno_name = src_anno_file.split('/')[-1]
    dst_anno_name = src_anno_name.split(src_anno_format)[0] + dst_anno_format
    dst_anno_file = os.path.join(dst_anno_path, dst_anno_name)
    objects = parse_fun(src_anno_file)
    dump_fun(objects, dst_anno_file)
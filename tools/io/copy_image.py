import wwtool


src_path = '/data/hrsc2016/v0/Test/AllImages'
dst_path = '/data/hrsc2016/v1/test/images'
wwtool.copy_image_files(src_path, dst_path, dst_file_format='png')
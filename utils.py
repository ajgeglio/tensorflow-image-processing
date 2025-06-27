from PIL import Image
import pandas as pd
import glob
import numpy as np
import os
import sys
import time
import os
import shutil

class ReturnTime:
    def __init__(self):
        pass

    @staticmethod
    def get_time_obj(time_s):
        if pd.notnull(time_s):
            return datetime.datetime.fromtimestamp(time_s)
        return np.nan

    @classmethod
    def get_Y(cls, time_s):
        dt = cls.get_time_obj(time_s)
        return dt.strftime('%Y') if isinstance(dt, datetime.datetime) else np.nan

    @classmethod
    def get_m(cls, time_s):
        dt = cls.get_time_obj(time_s)
        return dt.strftime('%m') if isinstance(dt, datetime.datetime) else np.nan

    @classmethod
    def get_d(cls, time_s):
        dt = cls.get_time_obj(time_s)
        return dt.strftime('%d') if isinstance(dt, datetime.datetime) else np.nan

    @classmethod
    def get_t(cls, time_s):
        dt = cls.get_time_obj(time_s)
        return dt.strftime('%H:%M:%S') if isinstance(dt, datetime.datetime) else np.nan

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def print_tensor_info(tensor):
        print("Tensor shape:", tensor.shape)
        print("Tensor dtype:", tensor.dtype)

    @staticmethod
    def del_files(filepath_list): 
        for filepath in filepath_list:
            if os.path.isfile(filepath):
                os.remove(filepath)
            else:
                # If it fails, inform the user.
                print("Error: %s file not found" % filepath)

    @staticmethod
    def copy_imgs_by_label(df, label, dest_folder):
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        img_pth_list = df.loc[df.label == label, 'image_path']
        num_imgs = len(img_pth_list)
        print("copying", num_imgs, "images")
        i=0
        for img_pth in img_pth_list:
            src = img_pth
            img_name = os.path.basename(img_pth)
            dest = os.path.join(dest_folder, img_name)
            # File copy was interrupted often due to network, added src/dest comparison
            if os.path.exists(dest):
                if os.stat(src).st_size == os.stat(dest).st_size:
                    i+=1
                else:
                    shutil.copy(src, dest)
                    i+=1
            else:
                shutil.copy(src, dest)
                i+=1
            print("Copying", i,"/",num_imgs, end='  \r')

    @staticmethod
    def clear_pycache(directory):
        for root, dirs, files in os.walk(directory):
            for d in dirs:
                if d == "__pycache__":
                    shutil.rmtree(os.path.join(root, d))

    @staticmethod
    def read_image_lst_txt(image_lst_pth):
        with open(image_lst_pth, 'r') as f:
            image_lst = f.read().splitlines()
        return image_lst

    @staticmethod
    def read_image_path_to_lst(image_dir):
        # Collect all supported image files from PrimaryImages subfolder
        patterns = [
            os.path.join(image_dir, "*.png"),
            os.path.join(image_dir, "*.jpg"),
            os.path.join(image_dir, "*.jpeg"),
            os.path.join(image_dir, "*.tif"),
            os.path.join(image_dir, "*.tiff"),
        ]
        image_files = []
        for pattern in patterns:
            image_files.extend(glob.glob(pattern))
        return image_files

    @staticmethod
    def is_supported_image_file(filename): 
        supported_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.tif'] 
        return any(filename.lower().endswith(ext) for ext in supported_formats)

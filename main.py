import json
import numpy as np
from PIL import Image
import os
import shutil   # for copy original image

from termcolor import colored   # print colored warnings, errors
from typing import Dict, List, Any

import glob
from tqdm import tqdm       # Status display

from maskUtils import polygons_to_bitmask, rle_to_bitmask

g_PATH_TO_CONFIG = './config.json'      # Set this to the config file

class Config:
    # Defaults
    input_dir = './chosen_bg_images_all/*.jpg'    # Use ASTERISK to denote all files in that type
    annotation_path = './chosen_bg_images_all/coco_annotations_trainval2014/annotations/instances_val2014.json' # The Annotation, containging category info, ids, and else
    output_dir = './results/'
    output_original_image = True
    path_to_config = g_PATH_TO_CONFIG

    def __init__(self, path_to_config) -> None:
        if os.path.isfile(path_to_config):
            self.path_to_config = path_to_config
            with open(path_to_config, 'r') as f:
                config_dict = json.load(f)
                self.input_dir = config_dict['input_dir']
                self.annotation_path = config_dict['annotation_path']
                self.output_dir = config_dict['output_dir']
                self.output_original_image = config_dict['output_original_image']

    def get_img_paths(self) -> List:
        '''
        Expand the asterisk, if reacheable
        '''
        img_paths = glob.glob(self.input_dir)
        return img_paths

    def get_parent_ann_dict(self):
        '''Load json file as dict, returns the parent dict'''
        with open(self.annotation_path) as f:
            tmpStr = f.readline()
        parent_dict = json.loads(tmpStr)   # The parsed dictionary object, containing annotations for all used categories
        return parent_dict




def mask_to_png(mask_arr: np.ndarray):
    rgb_arr = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 3), dtype='uint8')
    def map_to_rgb(is_white: bool):
        if is_white: 
            return np.array([255,255,255])
        else: return np.array([0,0,0])
    # Convert to RGB
    
    for row_num, row in enumerate(mask_arr):
        for col_num, entry in enumerate(row):
            rgb_arr[row_num][col_num] = map_to_rgb(entry)

    
    img = Image.fromarray(rgb_arr,mode='RGB')
    
    return img



def query_img_height_width(img_list: List[Dict], target_id: int):
    for img_dict in img_list:
        if img_dict['id'] != target_id: continue
        else:
            return img_dict['height'], img_dict['width']
        colored(f"Error: Unreached target id: {target_id}", "red")

def query_segmentation_catid_list(ann_list: List[Dict], target_id: int) -> List:
    returnList = []
    for ann_dict in ann_list:
        if ann_dict['image_id'] != target_id: continue
        else:
            returnList.append({"segmentation": ann_dict['segmentation'], "cat_id": ann_dict['category_id']})
    if len(returnList) == 0: colored(f"Warning: Unreached target id: {target_id}", "yellow")
    return returnList

def query_category_name(cat_list: List, target_cat_id: int):
    for cat_dict in cat_list:
        if cat_dict['id'] != target_cat_id: continue
        else:
            return cat_dict['name']
        colored(f"Error: Unreached target category id: {target_cat_id}", "red")

def parse_id_filenamenoext(path: str) -> int:
    # Because the ids are written in filename, we parse them
    # Located at the end 
    if not (os.path.isfile(path) and (path.endswith('.jpg') or path.endswith('.png'))):
        colored(f'Warning: {path} is not valid and will be skipped.', 'yellow')
    id_string = path[:-4].split('_')[-1]
    filenamenoext = path[:-4].split('/')[-1]

    return int(id_string), filenamenoext


def main():
    # The procedures
    config = Config(g_PATH_TO_CONFIG)
    if (not os.path.isdir(config.output_dir)): os.mkdir(config.output_dir)
    img_paths = config.get_img_paths()
    print(f'Starting to process {len(img_paths)} images')
    parent_dict = config.get_parent_ann_dict()

    for src_img_path in tqdm(img_paths):
        id, filenamenoext = parse_id_filenamenoext(src_img_path)

        # Annotation list, contains `segmentation`, `category_id`
        ann_list = parent_dict['annotations']
        # Images list, contains image `width` and `height`
        img_list = parent_dict['images']
        # Category list, contains `name` and `id`(category id)
        cat_list = parent_dict['categories']

        img_height, img_width = query_img_height_width(img_list, id)
        segmentation_list = query_segmentation_catid_list(ann_list, id)

        if len(segmentation_list) == 0:
            colored(f'Warning: cannot find segmentation info of image id {id}')
            continue
        if not os.path.isdir(os.path.join(config.output_dir, filenamenoext)): os.mkdir(os.path.join(config.output_dir, filenamenoext))
        for index, seg_catid_dict in enumerate(segmentation_list):
            if isinstance(seg_catid_dict['segmentation'], List):
                mask_arr = polygons_to_bitmask(seg_catid_dict['segmentation'], img_height, img_width)   # This is the true-false mask
            else: mask_arr = rle_to_bitmask(seg_catid_dict['segmentation'])
            cat_name = query_category_name(cat_list, seg_catid_dict['cat_id'])

            img = mask_to_png(mask_arr)
            img.save(os.path.join(config.output_dir, filenamenoext, f'{index}_{cat_name}.png'))

            if config.output_original_image:
                shutil.copy(src_img_path, os.path.join(config.output_dir, filenamenoext))





            


if (__name__ == '__main__'):
    main()
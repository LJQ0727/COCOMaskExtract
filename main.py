import json
import numpy as np
from PIL import Image
import os
import shutil   # for copy original image
import torch
import pickle
import cv2

from termcolor import colored   # print colored warnings, errors
from typing import Dict, List, Any

import glob
from tqdm import tqdm       # Status display

from maskUtils import polygons_to_bitmask, rle_to_bitmask

g_PATH_TO_CONFIG = './config.json'      # Set this to the config file
if torch.cuda.is_available(): device = torch.device('cuda')
else: device = torch.device('cpu')

class Config:
    # Defaults
    input_dir = './chosen_bg_images_all/*.jpg'    # Use ASTERISK to denote all files in that type
    annotation_path = './chosen_bg_images_all/coco_annotations_trainval2014/annotations/instances_val2014.json' # The Annotation, containging category info, ids, and else
    output_dir = './results/'
    output_original_image = True
    path_to_config = g_PATH_TO_CONFIG
    mask_lower_threshold = 0.2    # Mask area space / total area 
    mask_upper_threshold = 0.4    # Mask area space / total area 
    dilation_kernel_size = 10
    # Too small masks will be ignored

    def __init__(self, path_to_config) -> None:
        if os.path.isfile(path_to_config):
            self.path_to_config = path_to_config
            with open(path_to_config, 'r') as f:
                config_dict = json.load(f)
                self.input_dir = config_dict['input_dir']
                self.annotation_path = config_dict['annotation_path']
                self.output_dir = config_dict['output_dir']
                self.output_original_image = config_dict['output_original_image']
                self.mask_lower_threshold = config_dict['mask_lower_threshold']
                self.mask_upper_threshold = config_dict['mask_upper_threshold']
                self.dilation_kernel_size = config_dict['dilation_kernel_size']

                

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




def mask_to_png_and_save(mask_arr, dest_path: str, accept = True):
    # Use GPU to accelerate
    # mask_arr = torch.tensor(mask_arr, dtype=torch.uint8).to(device)
    mask_arr = 255 * mask_arr
    rgb_arr = torch.stack([mask_arr, mask_arr, mask_arr], dim=2)

    def save_img(tensor, dest_path: str):
        
        img = Image.fromarray(tensor,mode='RGB')
        img.save(dest_path)

    if accept:
        save_img(rgb_arr.cpu().numpy(), dest_path)



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

def get_mask_space(mask):
    # mask = torch.tensor(mask).to(device)
    return float(torch.sum(mask)) / (mask.shape[0] * mask.shape[1])

def query_category_name(cat_list: List, target_cat_id: int):
    for cat_dict in cat_list:
        if cat_dict['id'] != target_cat_id: continue
        else:
            return cat_dict['name']
        colored(f"Error: Unreached target category id: {target_cat_id}", "red")

def parse_id_filenamenoext(path: str):
    # Because the ids are written in filename, we parse them
    # Located at the end 
    if not (os.path.isfile(path) and (path.endswith('.jpg') or path.endswith('.png'))):
        colored(f'Warning: {path} is not valid and will be skipped.', 'yellow')
    id_string = path[:-4].split('_')[-1]
    filenamenoext = path[:-4].split('/')[-1]
    _, ext = os.path.splitext(path)

    return int(id_string), filenamenoext, ext


def main():
    # The procedures
    config = Config(g_PATH_TO_CONFIG)
    if (not os.path.isdir(config.output_dir)): os.mkdir(config.output_dir)
    img_paths = config.get_img_paths()
    print(f'Starting to process {len(img_paths)} images')
    parent_dict = config.get_parent_ann_dict()

    for src_img_path in tqdm(img_paths):
        id, filenamenoext, ext_name = parse_id_filenamenoext(src_img_path)
        # Annotation list, contains `segmentation`, `category_id`
        ann_list = parent_dict['annotations']
        # Images list, contains image `width` and `height`
        img_list = parent_dict['images']
        # Category list, contains `name` and `id`(category id)
        cat_list = parent_dict['categories']

        img_height, img_width = query_img_height_width(img_list, id)
        segmentation_list = query_segmentation_catid_list(ann_list, id)
        target_dir = os.path.join(config.output_dir, filenamenoext)

        if len(segmentation_list) == 0:
            colored(f'Warning: cannot find segmentation info of image id {id}')
            continue
        
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        masks_info = dict()

        for i, seg_catid_dict in enumerate(segmentation_list):
            if isinstance(seg_catid_dict['segmentation'], List):
                mask_arr = polygons_to_bitmask(seg_catid_dict['segmentation'], img_height, img_width)   # This is the true-false mask
            else: mask_arr = rle_to_bitmask(seg_catid_dict['segmentation'])
            cat_name = query_category_name(cat_list, seg_catid_dict['cat_id'])

            kernel = np.ones((config.dilation_kernel_size, config.dilation_kernel_size), dtype=np.uint8)
            mask_arr = cv2.dilate(mask_arr, kernel, 1)
            
            mask_arr = torch.tensor(mask_arr).to(device)

            masks_info[cat_name] = torch.logical_or(masks_info.get(cat_name, torch.zeros(mask_arr.shape).to(device)), mask_arr).to(dtype=torch.uint8)

        file_idx = 0
        pop_cats = []
        for (cat_name, mask) in masks_info.items():
            space = get_mask_space(mask)
            if cat_name not in pop_cats:
                if space >= config.mask_lower_threshold and space <= config.mask_upper_threshold:
                    out_filename = os.path.join(config.output_dir, filenamenoext, f'{file_idx}_{cat_name}.png')
                    mask_to_png_and_save(mask, out_filename)
                    masks_info[cat_name] = {"file_name": f'{file_idx}_{cat_name}.png', "area": round(space, 2)}
                    file_idx += 1
                else:
                    pop_cats.append(cat_name)

        for cat in pop_cats:
            masks_info.pop(cat)

        if masks_info == {}:    # This image and its masks will not be output
            shutil.rmtree(target_dir)
            continue

        output_info = {filenamenoext: masks_info}
        # Output .pkl file of file info
        with open(os.path.join(config.output_dir, filenamenoext, 'info.pkl'), 'wb') as f:
            pickle.dump(output_info, f)

        if config.output_original_image:
            if not os.path.isfile(os.path.join(config.output_dir, filenamenoext, ext_name)):
                shutil.copy(src_img_path, target_dir)



if __name__ == '__main__':
    main()
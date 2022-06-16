# MaskExtract

## Task
This repository is aimed to extract masks represented in .png format
from 80 selected categories.

## About `chose_bg_images_all`
This folder contains images from COCO val2014 dataset.
### About `coco_annotations_trainval2014`
This folder contains `json` files,
which can be parsed to identify human-checked regional and categorial information.

This project aims to process its useful information to generate masks,
which are black and white png files, for further use.


generate folders
from polygon (segmentation, annotation) to ...

## About configuration
Please configure these properties and make sure they stay in `config.json` in the same folder as `main.py`.
For example,

    "input_dir": "./chosen_bg_images_all/*.jpg",
    "output_dir": "./results/",
    "output_original_image": true,
    "annotation_dir": "./chosen_bg_images_all/coco_annotations_trainval2014/annotations/instances_val2014.json"
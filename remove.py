import argparse
from pydoc import classname
from PIL import Image
import cv2
import os
import random
import pdb
import numpy as np 
random.seed()


def get_class(mask_dir, dirname):
    classes = []
    for fname in os.listdir(mask_dir):
        if fname.split('.')[0] == dirname:
            continue
        fnam = fname.split('.')[0]
        classes.append(fnam.split('_')[1])
    classes = list(set(classes))
    return classes

def merge_mask(class_name, mask_dir, dirname, out_path):
    im = Image.open(os.path.join(mask_dir, dirname+'.jpg'))
    im = np.array(im)
    h,w = im.shape[:2]
    m_mask = np.zeros((h,w,3), dtype="uint8")
    for fname in os.listdir(mask_dir):
        if fname.split('.')[0] == dirname:
            continue
        fnam = fname.split('.')[0]
        cur_class = fnam.split('_')[1]
        if class_name!=cur_class:
            continue
        cur_mask = Image.open(os.path.join(mask_dir, fname))
        cur_mask = np.array(cur_mask)
        if cur_mask.shape[:2] != (h,w):
            pdb.set_trace()
        for i in range(h):
            for j in range(w):
                if (cur_mask[i][j]==[255,255,255]).all():
                    m_mask[i][j] = [255,255,255]
    # m_mask = Image.fromarray(m_mask)
    m_mask = cv2.cvtColor(m_mask, cv2.COLOR_RGB2BGR)
    kernel = np.ones((16,16), dtype="uint8")
    m_mask = cv2.dilate(m_mask, kernel, 1)
    # m_mask.save(out_path)
    cv2.imwrite(out_path, m_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--png_in_dir', type = str, required=True)
    parser.add_argument('--single_mask_dir', type =str, required=True)
    parser.add_argument('--merge_mask_dir', type =str, required=True)
    args = vars(parser.parse_args())
    cnt = 0 
    for dirname in os.listdir(args['single_mask_dir']):
        cur_img_path = os.path.join(args['single_mask_dir'], dirname)
        out_path = os.path.join(args['merge_mask_dir'], dirname+'_mask.png')
        class_list = get_class(cur_img_path, dirname)
        if len(class_list)==0:
            continue
        cnt+=1
        merge_mask(class_list[0], cur_img_path, dirname, out_path)
        os.system('cp '+os.path.join(args['png_in_dir'], dirname+'.png')+ ' '+args['merge_mask_dir'])
        if cnt >= 100:
            break
        


    



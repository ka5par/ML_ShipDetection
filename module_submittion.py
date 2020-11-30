## import

# //TODO get rid of nonsense imports. 
from collections import defaultdict
import sys
import argparse
import cv2 
import glob
import logging
import os
import time

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

import pycocotools.mask as mask_util
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

from detectron2.config import get_cfg

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def parse_args():
    parser = argparse.ArgumentParser(description='Create new submission file')
    parser.add_argument(
        '--test_folder',
        dest='test_folder',
        help='test data folder (/path/to/test)',
        default="input/test_v2/",
        type=str
    )
    parser.add_argument(
        '--submit_csv',
        dest='submit_csv',
        help='submit file name',
        default='submit.csv',
        type=str
    )
    parser.add_argument(
        '--config_file',
        dest='config_file',
        help='"config file name (.yaml)',
        default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
        type=str
    )
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='Path to model',
        default='/outputs',
        type=str
    )
    parser.add_argument(
        '--batch_size_per_image',
        dest='batch_size_per_image',
        help='Batch size per image (ROI heads per image)',
        default=512,
        type=int
    )
    parser.add_argument(
        '--num_classes',
        dest='num_classes',
        help='Number of classes to predict',
        default=1,
        type=int
    )
    parser.add_argument(
        '--nms_thres',
        dest='nms_thres',
        help='NMS Threshold',
        default=0,
        type=float
    )
    parser.add_argument(
        '--score_thres',
        dest='score_thres',
        help='Score threshold for true value.',
        default=0.8,
        type=float
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def create_test_datatset(dataset_dir):    
    img_dir = dataset_dir
    dataset_dicts = []
    
    for img_path in glob.glob(img_dir + '*.jpg'):
        record = {}
        file_path = img_path
        image_id = img_path.split('/')[-1].split('.')[0]
        record['file_name'] = file_path
        record['image_id'] = image_id
        dataset_dicts.append(record)
    
    return dataset_dicts

#https://www.kaggle.com/alexlzzz/rl-encoding
# https://github.com/pascal1129/kaggle_airbus_ship_detection
def rle_encode(img, shape=(768,768)):
    pixels = img.T.flatten()    # T is needed here.
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def extract_result(img_dict):
    
    img_id = img_dict['ImageId']
    predictions = img_dict['Output']
    img_mask = predictions['instances'].pred_masks.cpu().numpy()
    img_score = predictions["instances"].scores.cpu().numpy() 
    
    #none is bbox legacy
    return img_id, None, img_mask, img_score


        
# https://github.com/pascal1129/kaggle_airbus_ship_detection
def get_im_no_ship(df_with_ship, dataset_dir):
    im_all = os.listdir(dataset_dir)
    im_no_ship = list(set(im_all).difference(set(df_with_ship['ImageId'].tolist())))
    return im_no_ship

def get_im_list(df):
    df_with_ship = df[df['EncodedPixels'].notnull()]['ImageId']
    return list(set(df_with_ship))

def get_empty_list(length):
    list_empty = []
    for _ in range(length):
        list_empty.append('')
    return list_empty
   
def main(args):    
    
    # define inputs/outputs hardcoded bastards -.-
    dataset_dir = args.test_folder
    csv_origin = 'rle.csv'
    csv_submit = args.submit_csv
    
    print("Start creating files")
    print(dataset_dir)
    test_dataset = create_test_datatset(dataset_dir)

    # load model, config changes - predicting 768x768 masks, NMS 0
    DatasetCatalog.register("submit_test", create_test_datatset)
    od_dataset = MetadataCatalog.get("submit_test")

    # https://detectron2.readthedocs.io/modules/config.html
    # https://medium.com/@hirotoschwert/digging-into-detectron-2-part-5-6e220d762f9
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(args.config_file))
    cfg.MODEL.WEIGHTS = os.path.join(args.model_path, "model_final.pth")
    cfg.DATASETS.TEST = ("submit_test1", )
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thres  
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms_thres
    predictor = DefaultPredictor(cfg)

    outputs = []
    img_ids = []
    img_rle = []
    img_scores = []
    img_areas = []
    
    def masks_to_rle_csv(img_id, masks, scores): 
    
        index = np.argsort(-scores)          
        bg = np.zeros((768,768), dtype=np.uint8)   
        bg_list = []

        for i in index:

            mask = masks[i,:,:]
            if(mask is None):
                continue
            mask_xor = (mask^bg)&mask #bg == 0, = 1, 1^1 & mask
            area = mask_xor.sum()
            if(area == 0):
                continue
            bg += mask_xor   #NO OVERLAPS...  

            img_ids.append(img_id)
            img_rle.append(rle_encode(mask_xor))
            img_scores.append(scores[i])
            img_areas.append(area)
    
    for i in range(len(test_dataset)):
        
        img_id = test_dataset[i]['file_name'].split('/')[-1]

        inputs = cv2.imread(test_dataset[i]['file_name'])
        output = predictor(inputs) 

        outputs.append({ 'ImageId':img_id,'Output':output})

        if(i%1000 == 0):
            print(i,len(test_dataset))

    for i in range(len(outputs)):
        img_id, boxes, segms, img_score = extract_result(outputs[i])

        if segms is not None and len(segms) > 0:
            masks = np.array(segms)
            masks_to_rle_csv(img_id, masks, img_score)
        if(i%1000 == 0):
            print(i,len(outputs))    

    df = pd.DataFrame({'ImageId':img_ids, 'EncodedPixels':img_rle, 'confidence':img_scores, 'area':img_areas})
    df = df[['ImageId', 'EncodedPixels', 'confidence', 'area']]   # change the column index
    df.to_csv(csv_origin, index=False, sep=str(','))

    df_submit = df

    print("Detectron2:  %d instances,  %d images"  %(df_submit.shape[0], len(get_im_list(df_submit))))

    df_submit = df_submit[ (df_submit['area']>50) & (df_submit['confidence']>=0.85) ]
    
    def generate_final_csv(df_with_ship, dataset_dir = dataset_dir):
        print("Detectron2:  %d instances,  %d images"  %(df_with_ship.shape[0], len(get_im_list(df_with_ship))))
        im_no_ship = get_im_no_ship(df_with_ship, dataset_dir)
        # write dataframe into .csv file
        df_empty = pd.DataFrame({'ImageId':im_no_ship, 'EncodedPixels':get_empty_list(len(im_no_ship))})
        df_submit = pd.concat([df_with_ship, df_empty], sort=False)
        df_submit.drop(['area','confidence'], axis=1, inplace=True)
        df_submit.to_csv(csv_submit, index=False,sep=str(','))   # str(',') is needed
        print('Done!')
    
    generate_final_csv(df_submit)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
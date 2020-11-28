## import

# //TODO get rid of nonsense imports. 
import os
import copy
import cv2
import json
import pycocotools
import random 
import glob

import shapley 
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

assert torch.__version__.startswith("1.6")

import detectron2
import detectron2.data.transforms as T
import detectron2.utils.comm as comm

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg

from detectron2.data import MetadataCatalog,DatasetMapper,build_detection_train_loader,build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances 


from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler


def create_test_datatset():    
    img_dir = '/input/test_v2/'
    dataset_dicts = []
    
    for img_path in glob.glob(img_dir + '*.jpg'):
        record = {}
        file_path = img_path
        image_id = img_path.split('/')[-1].split('.')[0]
        record['file_name'] = file_path
        record['image_id'] = image_id
        dataset_dicts.append(record)
    
    return dataset_dicts

def PixelsToRLenc(pixels ,order='F',format=True):
    """
    Based off code by https://www.kaggle.com/alexlzzz
    pixels is a list of absolute pixel values which need to be converted. (1-243600)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not
    
    returns run length as an array or string (if format is True)
    """
    
    # Initialse empty array
    bytes = []
    for _ in range(0, 243600):
        bytes.append(0)
    
    # Place values from input list into the array
    for x in pixels:
        p = x - 1
        bytes[p] = 1
    
    runs = [] ## list of run lengths
    r = 0     ## the current run length
    pos = 1   ## count starts from 1 per WK
    for c in bytes:
        if ( c == 0 ):
            if r != 0:
                runs.append((pos, r))
                pos+=r
                r=0
            pos+=1
        else:
            r+=1

    #if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''
    
        for rr in runs:
            z+='{} {} '.format(rr[0],rr[1])
        return z[:-1]
    else:
        return runs
        
        
test_dataset = create_test_datatset()
img_ids = []
pred_string = []

DatasetCatalog.register("submit_test", create_test_datatset)
od_dataset = MetadataCatalog.get("submit_test")

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("submit_test", )
predictor = DefaultPredictor(cfg)

for test_data in test_dataset[24:25]:
    i = i + 1
    image_id = test_data['file_name'].split('/')[-1].split('.')[0]
    img = plt.imread(test_data['file_name'])
    outputs = predictor(img)
    preds = []
    for box,score in zip(outputs['instances'].get_fields()['pred_boxes'], outputs['instances'].get_fields()['scores']):
        bbox = []
        for idx in range(4): 
            bbox.append(box.data[idx].item())
        preds.append("{} {} {} {}".format(int(bbox[0]),int(bbox[1]), int(bbox[2]), int(bbox[3])))
        print(preds)
    if(i%1000 == 0):
        print(i,len(test_dataset))
    pred_string.append(" ".join(preds))
    img_ids.append(test_data['file_name'])
    


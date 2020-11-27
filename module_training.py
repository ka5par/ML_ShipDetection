## import

# //TODO get rid of nonsense imports. 
import os
import copy
import cv2
import json
import pycocotools
import random 

import numpy as np 
import pandas as pd
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

PATH = os.path.abspath(os.getcwd())

register_coco_instances("my_dataset_train",{},PATH + "/input/test_annotations.json",PATH + "/input/train_v2/") #TRAIN annotations got mixed up -.-
register_coco_instances("my_dataset_val",{},PATH + "/input/train_annotations.json",PATH + "/input/train_v2/")

my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

# //TODO Argumentize code. 
## configurations set up
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
#cfg.TEST.EVAL_PERIOD = 500 ## // TODO useful to set up eval_hook
cfg.DATALOADER.NUM_WORKERS = 4 ## 4 per gpu
cfg.SOLVER.IMS_PER_BATCH = 32 ## So many that it fulfills the gpu vram
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ship)

## configuration class

# https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
def custom_mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [T.RandomCrop("absolute",(256,256)), 
                      T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                      T.RandomFlip(prob=0.5, horizontal=True, vertical=False) 
                     ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

# //TODO make tensorboard work wihth loss_hook. 
## Tensorflow board
#%load_ext tensorboard
#%tensorboard --logdir logs


# //TODO make it check for checkpoints before ? 
## Start training.
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg) 
trainer.resume_or_load(resume=True)
trainer.train() #Trainer will throw out non-annotated pictures. 

## save the training.
checkpointer = DetectionCheckpointer(trainer.model, save_dir="./")
checkpointer.save("model_mask_resnet101_rcnn")  # save to save_dir


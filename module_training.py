# installation

#pip install -q -U torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#pip install -q


## import

import os
import copy
import cv2
import json
import pycocotools
import random
import torch
import torchvision
import detectron2

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


import LossEvalHook
assert torch.__version__.startswith("1.6")
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.checkpoint import DetectionCheckpointer

PATH = os.path.abspath(os.getcwd())

register_coco_instances("my_dataset_train",{},PATH + "/input/test_annotations.json",PATH + "/input/train_v2/") #TRAIN annotations got mixed up -.-
register_coco_instances("my_dataset_val",{},PATH + "/input/train_annotations.json",PATH + "/input/train_v2/")

my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

## configurations set up
# //TODO add validation and data augmentations
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.TEST.EVAL_PERIOD = 100
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.005
cfg.SOLVER.MAX_ITER = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ship)

## configuration class

# https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook.LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks

## Tensorflow board
#%load_ext tensorboard
#%tensorboard --logdir logs


## Start training.
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train() #Trainer will throw out non-annotated pictures.

## save the training.
checkpointer = DetectionCheckpointer(trainer.model, save_dir="./")
checkpointer.save("model_mask_resnet101_rcnn")  # save to save_dir


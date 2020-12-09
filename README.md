Instance segmentation framework for kaggle ship detection based Pytorch based toolboxes Detectron2 and Fastai.

Final presentation: https://docs.google.com/presentation/d/1O3DLAQ9SKukEivKACH7JCZqiKiia6ratmZuc1KuovQk/edit?usp=sharing

**Setup below:**

- Requirements:
    - Unix based OS (Linux/Mac OS).
    - **Nvidia** GPU with cuda 10.1 installed, and at least 10GB of vram.
    - Docker (version == 19.03) and docker-compose (version == 3.6)
    - 40GB of free space: ~10GB for docker image, ~30GB for data

*All other dependencies are taken care of by docker-compose.

### Data
```
kaggle competitions download -c airbus-ship-detection
```

Use the Kaggle API to download the dataset. https://github.com/Kaggle/kaggle-api

Put it in folder named "input"

### Model preprocessing / training and Kaggle submittion 

```
docker-compose up 
```

- After image has been set up open the notebook `module_notebook` and run all. 
- To use tensorboard, use another jupter notebook and call out: ```!tensorboard --logdir=runs --host=0.0.0.0``` and if run on a local machine go to link http://0.0.0.0:6006.  

# Quick overview: 

- Data Loader (module_preprocessing)
- Classifier (FastAI)
    - Output probability of ship on image.
    - Resnet34
- Instance Segmentation/Object Detection (Detectron2)
    - Output pixel mask of ships.
    - Augmentations 
        - Flips (Vertical 50%, Horisontal 50%)
        - Rotation (-20/+20 random rotation)
        - Random lighting (0.1 standard deviations -> 
    - Mask RCNN 50 layer pretrained on Coco .
        - 3 stage training (256x256,512x512,756x756)
    - Validation every 5k iterations. 
- Predictions on validation (module_submit)
- Kaggle submission (module_submit)


## Project steps  

- [X] Kaggle proptotype
- [X] Initial presentation
- [X] Data Loader module
- [X] Training module
- [X] Add data Augmentations
- [X] Submittion module
- [X] Classification module
- [X] Dockerize project
- [X] Argumentize code
- [X] Train
- [-] Train on cluster
- [X] Inference
- [X] Submit

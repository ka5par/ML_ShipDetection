Instance segmentation framework for kaggle ship detection[^8] using Pytorch based toolboxes Detectron2 and Fastai.

Final presentation [^2]

## Setup

- Requirements:
    - Unix based OS (Linux/Mac OS).
    - **Nvidia** GPU with cuda 10.1 installed, and at least 10GB of vram.
    - Docker (version == 19.03) and docker-compose (version == 3.6)
    - 40GB of free space: ~10GB for docker image, ~30GB for data

*All other dependencies are taken care of by docker-compose.

**Data**
```
kaggle competitions download -c airbus-ship-detection
```

Use the Kaggle API to download the dataset [^1]

Put it in folder named `input`

**Model preprocessing / training and Kaggle submittion**

```
docker-compose up 
```

- In the container, open the notebook `module_notebook` and run all. 
- To use tensorboard for metrics, use another jupter notebook and call out: ```!tensorboard --logdir=runs --host=0.0.0.0``` and if run on a local machine use a browser to go to http://0.0.0.0:6006.  

## Quick overview: 

- Data Loader (module_preprocessing) [^5]
- Classifier (FastAI) [^7] 
    - Output probability of ship on image.
    - Resnet34
- Instance Segmentation/Object Detection (Detectron2) [^6]
    - Output pixel mask of ships.
    - Augmentations [^3][^9]
        - Flips (Vertical 50%, Horisontal 50%)
        - Rotation (-20/+20 random rotation)
        - Random lighting (0.1 standard deviations)
    - Mask RCNN 50 layer pretrained on Coco .
        - 3 stage training (256x256,512x512,756x756) [^4]
    - Validation every 5k iterations. 
- Predictions on validation (module_submit) 
- Kaggle submission (module_submit) [^5]


## Project  

- [X] Kaggle proptotype
- [X] Initial presentation
- [X] Data Loader module
- [X] Training module
- [X] Add data Augmentations
- [X] Submit module
- [X] Classification module
- [X] Dockerize project
- [X] Argumentize code
- [X] Train
- [X] Inference
- [X] Submit


[^1]: https://github.com/Kaggle/kaggle-api
[^2]: https://docs.google.com/presentation/d/1O3DLAQ9SKukEivKACH7JCZqiKiia6ratmZuc1KuovQk/edit?usp=sharing
[^3]: https://jss367.github.io/Data-Augmentation-with-Detectron2.html
[^4]: https://www.kaggle.com/iafoss/unet34-dice-0-87/notebook
[^5]: https://github.com/pascal1129/kaggle_airbus_ship_detection
[^6]: https://github.com/facebookresearch/detectron2
[^7]: https://www.fast.ai/
[^8]: https://www.kaggle.com/c/airbus-ship-detection/overview
[^9]: https://link.springer.com/chapter/10.1007/978-981-15-5558-9_9

## Ideas 

- Mask RCNN (https://www.kaggle.com/hmendonca/airbus-mask-rcnn-and-coco-transfer-learning), YOLO (https://www.kaggle.com/smrutiranjanparida/yolov3-ship-detection), mmdetection (https://github.com/open-mmlab/mmdetection)


# Overview of the project.

ML_ShipDetection: https://www.kaggle.com/c/airbus-ship-detection

- Chat: **Slack**  
- Presentation 1: https://docs.google.com/presentation/d/1_MTQrBv8DQUJU2GY-gv-JzXXjAE8QjjIL-3lfwh7tqA/edit?usp=sharing
- Presentation 2: ***

## Data: 30GB of pictures

```
kaggle competitions download -c airbus-ship-detection
```

Use the Kaggle API to download the dataset.
https://github.com/Kaggle/kaggle-api

## Pipeline 

*** Open to discussion***
Data > Augmentation/Regularization > Instance Segmentation/Object Detection > Performance Validations on test > Predictions on val    

## Data and Data Augmentation


*** Open to discussion***
- Data modelled around architecture inputs. 
- Flips horisontal and vertical and randomized brightness (to lessen impacts of shadows - widely used in remote sensing pipelines)
 

## Architecture

*** Open to discussion *** 

## Validation

*** Open to discussion ***  

## Deadlines:
- Intermediate presentation: **Nov 16 - 18**
    - 5 min pres of current sitation, presenting the problem domain, and describing what has been done so far (have you collected all the required data), which problems you have encountered and what are your future steps.
- Final presentation **Dec 14 - 16**
    - longer, the above scope but more.
- Grading
  -amount and complexity of work performed (40%),
  -quality of presentation (30%),
  -degree to which you have completed the initial task (25%),
  -being on time (5%).

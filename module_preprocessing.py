## Preprocessing workflow

# 1. Converting into json cocolike format so it can be used in detectron 2 as a cocolike dataset.
# Code taken from: https://github.com/pascal1129/kaggle_airbus_ship_detection/blob/master/0_rle_to_coco/1_ships_to_coco.py
# 2. Convert json into training and testing data.
# Code taken from: https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py


## installs
# pip install -q -U "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
# pip install -q -U git+git://github.com/waspinator/pycococreator.git@0.2.0
# pip install -q -U "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

## import
import datetime
import fnmatch
import json
import os
import re

import funcy
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from pycococreatortools import pycococreatortools
from skimage.io import imread
from sklearn.model_selection import train_test_split


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def save_bad_ann(image_name, mask, segmentation_id):
    img = imread(os.path.join(IMAGE_DIR, image_name))
    fig, axarr = plt.subplots(1, 3)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(mask)
    axarr[2].imshow(img)
    axarr[2].imshow(mask, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    plt.savefig(os.path.join('./tmp', image_name.split('.')[0] + '_' + str(segmentation_id) + '.png'))
    plt.close()


def create_annotations():
    print("Started processing.")

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    for root, _, files in os.walk(IMAGE_DIR):
        image_paths = filter_for_jpeg(root, files)
        num_of_image_files = len(image_paths)

        for image_path in image_paths:
            image = Image.open(image_path)
            image_name = os.path.basename(image_path)
            image_info = pycococreatortools.create_image_info(
                image_id, image_name, image.size)
            coco_output["images"].append(image_info)

            rle_masks = df.loc[df['ImageId'] == image_name, 'EncodedPixels'].tolist()
            num_of_rle_masks = len(rle_masks)

            for index in range(num_of_rle_masks):
                binary_mask = rle_decode(rle_masks[index])
                class_id = 1
                category_info = {'id': class_id, 'is_crowd': 0}
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=2)

                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                else:
                    save_bad_ann(image_name, binary_mask, segmentation_id)

                segmentation_id = segmentation_id + 1
            if (image_id % 1000 == 0):
                print("Processing %d of %d is done. %d perc." % (
                    image_id, num_of_image_files, np.round(image_id / num_of_image_files, 2) * 100))
            image_id = image_id + 1

    return coco_output


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'info': info, 'licenses': licenses, 'images': images,
                   'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


# Read in the data.
PATH = os.path.abspath(os.getcwd())
TRAIN = PATH + '/input/train_v2/'
TEST = PATH + '/input/test_v2/'
SEGMENTATION = PATH + '/input/train_ship_segmentations_v2.csv'
exclude_list = ['6384c3e78.jpg', '13703f040.jpg', '14715c06d.jpg', '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']  # corrupted images

# 1.
dataset_train = TRAIN
csv_train = SEGMENTATION
IMAGE_DIR = dataset_train

df = pd.read_csv(csv_train)
df = df.dropna(axis=0)  # Drop where there are no ships.

INFO = {
    "description": "Kaggle Dataset",
    "url": "https://github.com/pascal1129",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "pascal1129",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'ship',
        'supercategory': 'ship',
    },
]

test = create_annotations()

with open(PATH + '/input/annotations.json', 'w') as output_json_file:
    json.dump(test, output_json_file, indent=4)

# 2.
ann_file = PATH + '/input/annotations.json'
train_file = PATH + '/input/train_annotations.json'
test_file = PATH + '/input/test_annotations.json'
split_size = 0.1

with open(ann_file, 'rt', encoding='UTF-8') as annotations:
    coco = json.load(annotations)
    info = coco['info']
    licenses = coco['licenses']
    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    number_of_images = len(images)

    images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

    images = funcy.lremove(lambda i: i['id'] not in images_with_annotations,
                           images)  # Removes all images without annotations.

    x, y = train_test_split(images, train_size=split_size)

    save_coco(train_file, info, licenses, x, filter_annotations(annotations, x), categories)
    save_coco(test_file, info, licenses, y, filter_annotations(annotations, y), categories)

    print("Saved {} entries in {} and {} in {}".format(len(x), train_file, len(y), test_file))

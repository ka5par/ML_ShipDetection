import datetime
import fnmatch
import json
import os
import re

#Oldie but goodie, should replace. We only run this thing once...
try: 
    from pycococreatortools import pycococreatortools
except ImportError:
    import pip
    print("Ignore warnings")
    pip.main(['install','q','-U','git+git://github.com/waspinator/pycococreator.git@0.2.0', 'funcy'])

import argparse
import sys
import funcy
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

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

def parse_args():
    parser = argparse.ArgumentParser(description='Create new annotations')
    parser.add_argument(
        '--train_folder',
        dest='train_folder',
        help='train data folder (/path/to/train)',
        default='/input/train_v2/',
        type=str
    )
    parser.add_argument(
        '--test_folder',
        dest='test_folder',
        help='test data folder (/path/to/test)',
        default="/input/test_v2/",
        type=str
    )
    parser.add_argument(
        '--seg',
        dest='seg',
        help='segmentations (/path/to/segmentation_csv)',
        default='/input/train_ship_segmentations_v2.csv',
        type=str
    )
    parser.add_argument(
        '--ann',
        dest='ann',
        help='annotations (/path/to/annotations.json)',
        default='/input/annotations.json',
        type=str
    )
    parser.add_argument(
        '--train_ann',
        dest='train_ann',
        help='segmentations (/path/to/train annotations.json)',
        default='/input/train_annotations.json',
        type=str
    )
    parser.add_argument(
        '--test_ann',
        dest='test_ann',
        help='test annotations (/path/to/test annotations.json)',
        default='/input/test_annotations.json',
        type=str
    )
    parser.add_argument(
        '--ann_bool',
        dest='ann_bool',
        help='create new annotations.json file from scratch (not reccomended if already created).',
        default=False,
        type=bool
    )
    parser.add_argument(
        '--train_split',
        dest='train_split',
        help='Train and test split',
        default=0.1,
        type=float
    )
    parser.add_argument(
        '--default',
        dest='default',
        help='Run with default settings. --default=True',
        default=False,
        type=bool
    )
    parser.add_argument(
        '--dataset_type',
        dest='dataset_type',
        help='0 - all images., 1 - equal dataset., 2 - remove all non-annotated.',
        default=0 ,
        type=int
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
# 1.
    # Read in the data.
    PATH = os.path.abspath(os.getcwd())

    TRAIN = PATH + args.train_folder #ARG
    TEST = PATH + args.test_folder #ARG
    SEGMENTATION = PATH + args.seg #ARG
    ANNOTATIONS_JSON = PATH + args.ann #ARG
    TRAIN_JSON = PATH + args.train_ann #ARG
    TEST_JSON = PATH + args.test_ann #ARG 
    create_new_annotations = args.ann_bool #Arg 
    split_size = args.train_split #ARG 
    dataset_type = args.dataset_type
    
    if create_new_annotations == True:
        print("Creating new annotations")
        
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

        with open(PATH + ANNOTATIONS_JSON, 'w') as output_json_file:
            json.dump(test, output_json_file, indent=4)
            
            

    # 2.
    with open(ANNOTATIONS_JSON, 'rt', encoding='UTF-8') as annotations:
        print("Creating new train/test split")         
        coco = json.load(annotations)
        print("Loaded annotataions file:", ANNOTATIONS_JSON)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        
        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
        if dataset_type == 1:
            print("Dataset equal annotated/not annotated images.") 
            
            img_ids = [int(image['id']) for image in images]
            img_ids_ann = [int(annotation['image_id']) for annotation in annotations]
            
            #Choose the same amount of no annotations.
 
            img_ids_no_ann = np.setdiff1d(img_ids,img_ids_ann)
            img_ids_no_ann = np.sort(np.random.choice(img_ids_no_ann, len(img_ids_ann))) 
            
            images_consolidated = np.append(img_ids_no_ann,np.array(images_with_annotations))
            images_consolidated = np.sort(images_consolidated)
            
            images = funcy.lremove(lambda i: i['id'] not in images_consolidated,
                               images) # Removes all not in consolidated.
        elif dataset_type == 2:
            print("Dataset remove not annotated images.")
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations,
                               images) # Removes all not annotated.
         

        val, train = train_test_split(images, train_size=split_size)
        
        save_coco(TRAIN_JSON, info, licenses, train, filter_annotations(annotations, train), categories)    
        save_coco(TEST_JSON, info, licenses, val, filter_annotations(annotations, val), categories)
        

        print("Saved {} entries in {} and {} in {}".format(len(train), TRAIN_JSON, len(val), TEST_JSON))

if __name__ == '__main__':
    args = parse_args()
    main(args)
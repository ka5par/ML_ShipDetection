import fastai
from fastai.vision.all import *
import argparse
import sys
import os
import glob
import pandas as pd


def output_check(output_csv_path):
    PATH = os.path.abspath(os.getcwd())

    if os.path.isfile(PATH + output_csv_path):
        raise argparse.ArgumentTypeError(
            "Output file allready exist: {}".format(PATH + output_csv_path)
        )
    else:
        return output_csv_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Determine the confidence if image contains a ship."
    )

    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        help="relative path to the model weights",
        default="/output/ship_yes_no.pkl",
    )

    parser.add_argument(
        "--image",
        dest="image",
        type=str,
        help="relative path to the image for single image classification",
    )

    parser.add_argument("--dir", dest="image_dir", type=str, help="directory of images")

    parser.add_argument(
        "--output",
        dest="output",
        type=output_check,
        help="path for output csv file",
        default="/output/ship_proba.csv",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def main(args):
    print("STARTED CLASSIFYING IMAGE WITH SHIP CONFIDENCE")
    PATH = os.path.abspath(os.getcwd())

    model = PATH + args.model
    learner = load_learner(model)

    if args.image:
        # Predicted class '0'/'1' is ship
        # Tensor of class
        # probabilities - tensor array of class probabilities
        predicted_class, tensor, probabilites = learner.predict(PATH + args.image)
        return probabilites.numpy()[1]

    if args.image_dir:
        df_probs = pd.DataFrame(columns=["image_id", "ship_proba"])

        images = glob.glob(PATH + args.image_dir + "*.jpg")
        print("TOTAL IMAGES: {}".format(len(images)))

        i = 1
        for img_path in glob.glob(PATH + args.image_dir + "*.jpg"):
            if (i % 1000) == 0:
                print("{} images classified".format(i))
            predicted_class, tensor, probabilites = learner.predict(img_path)

            filename = img_path.split("/")[-1]
            df_probs = df_probs.append(
                {
                    "image_id": filename.split(".")[0],
                    "ship_proba": probabilites.numpy()[1],
                },
                ignore_index=True,
            )

            i += 1
        df_probs.to_csv(PATH + args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)

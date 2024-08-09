import os
import argparse
import json
import cv2

import sys

sys.path.append("../..")

from utils.dataset_utils import *
from tqdm import tqdm
from typing import Dict


def verify_image_exist(json_data: Dict, image_folder: str) -> None:
    for image in tqdm(json_data["images"]):
        image_file = os.path.join(image_folder, image["file_name"])
        if not os.path.exists(image_file):
            print(f"Invalid image file: {image_file}")


def verify_image_dimension(json_data: Dict, image_folder: str) -> None:
    for image in tqdm(json_data["images"]):
        image_file = os.path.join(image_folder, image["file_name"])
        image_data = cv2.imread(image_file)
        height, width, _ = image_data.shape
        if height != image["height"] or width != image["width"]:
            print(f"Invalid image dimension: {image_file}")


def main(args):
    data_dir = args.data_dir
    print(f"Data Directory: {data_dir}")

    print(f"Reading train data ...")
    train_json_file = os.path.join(data_dir, "train.json")
    train_json = read_json(train_json_file)

    # Check all the image file are valid
    print(f"Checking image file ...")
    train_folder = os.path.join(data_dir, "train")
    verify_image_exist(train_json, train_folder)

    # Check all the image dimension are valid
    print(f"Checking image dimension ...")
    verify_image_dimension(train_json, train_folder)

    print(f"Reading val data ...")
    val_json_file = os.path.join(data_dir, "val.json")
    val_json = read_json(val_json_file)

    # Check all the image file are valid
    print(f"Checking image file ...")
    val_folder = os.path.join(data_dir, "val")
    verify_image_exist(val_json, val_folder)

    # Check all the image dimension are valid
    print(f"Checking image dimension ...")
    verify_image_dimension(val_json, val_folder)

    print(f"Verify Done!")


if __name__ == "__main__":
    DEFAULT_DATA_DIR = "/mnt/hdd/davidwong/data/trash/combined_dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Path to Ocean Plastic Dataset. Defualt: {DEFAULT_DATA_DIR}",
    )
    args = parser.parse_args()
    main(args)

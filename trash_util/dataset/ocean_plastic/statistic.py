import os
import argparse
import json

import sys

sys.path.append("../..")

from utils.dataset_utils import *


def count_images(json_data: dict) -> int:
    return len(json_data["images"])


def count_bbox(json_data: dict) -> int:
    return len(json_data["annotations"])


def category_statistic(json_data: dict) -> None:
    for category in json_data["categories"]:
        print(f"Category ID: {category['id']}, Name: {category['name']}")


def main(args):
    data_dir = args.dataset_dir
    print(f"Dataset Directory: {data_dir}")

    print(f"Reading train data ...")
    train_json_file = os.path.join(data_dir, "train_annotations.json")
    train_json = read_json(train_json_file)
    train_images = count_images(train_json)
    print(f"Number of train images: {train_images}")
    train_bbox = count_bbox(train_json)
    print(f"Number of train bounding boxes: {train_bbox}")
    category_statistic(train_json)

    print()
    print(f"Reading val data ...")
    val_json_file = os.path.join(data_dir, "val_annotations.json")
    val_json = read_json(val_json_file)
    val_images = count_images(val_json)
    print(f"Number of val images: {val_images}")
    val_bbox = count_bbox(val_json)
    print(f"Number of val bounding boxes: {val_bbox}")
    category_statistic(val_json)

    print()
    print(f"Reading test data ...")
    test_json_file = os.path.join(data_dir, "test_annotations.json")
    test_json = read_json(test_json_file)
    test_images = count_images(test_json)
    print(f"Number of test images: {test_images}")
    test_bbox = count_bbox(test_json)
    print(f"Number of test bounding boxes: {test_bbox}")
    category_statistic(test_json)


if __name__ == "__main__":
    DEFAULT_DATASET_DIR = "/mnt/hdd/davidwong/data/trash/Ocean_Plastic"
    parser = argparse.ArgumentParser(description="Statistic of Ocean Plastic Dataset")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=f"Path to Ocean Plastic Dataset. Defualt: {DEFAULT_DATASET_DIR}",
    )
    args = parser.parse_args()
    main(args)

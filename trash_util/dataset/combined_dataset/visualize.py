import os
import argparse
import matplotlib.pyplot as plt
import random
import cv2

import sys

sys.path.append("../..")
from utils.dataset_utils import *

from typing import Dict, List


def main(args):
    data_dir = args.data_dir
    print(f"Data Directory: {data_dir}")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    num_samples = args.vis_samples

    train_json_path = os.path.join(data_dir, "train.json")
    train_json = read_json(train_json_path)

    target_classes = args.target_classes
    id_to_category_name = get_id_to_category_names(train_json)

    for category_id, category_name in id_to_category_name.items():
        print(f"Category ID: {category_id}, Name: {category_name}")

    if target_classes is None:
        image_datas = train_json["images"]
        image_ids = [image_data["id"] for image_data in image_datas]
        random.shuffle(image_ids)
        image_ids = image_ids[:num_samples]
    else:
        for target_class in target_classes:
            category_name = id_to_category_name[target_class]
            print(f"Target Class: {category_name}")

        image_ids = get_image_ids_by_category_ids(train_json, target_classes)
        image_ids = image_ids[:num_samples]

    images_datas = get_images_by_image_ids(train_json, image_ids)
    annotations_datas = get_annotation_by_image_ids(train_json, image_ids)

    idx = 0
    for image_id, image_data in images_datas.items():
        image_path = os.path.join(data_dir, "train", image_data["file_name"])
        image = read_image(image_path)
        annotations = annotations_datas[image_id]
        bbox_list = []
        class_name_list = []
        for annotation in annotations:
            bbox = annotation["bbox"]
            category_id = annotation["category_id"]
            category_name = id_to_category_name[category_id]
            bbox_list.append(bbox)
            class_name_list.append(category_name)
        vis_image = vis_bbox(image, bbox_list, class_name_list)
        vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vis_image_path = os.path.join(output_dir, f"{idx}.jpg")
        print(f"Result saved to {vis_image_path}")
        cv2.imwrite(vis_image_path, vis_image)
        idx += 1


if __name__ == "__main__":
    DEFAULT_DATA_DIR = "/mnt/hdd/davidwong/data/trash/combined_dataset"
    DEFAULT_OUTPUT_DIR = "/mnt/hdd/davidwong/vis/combined_dataset"
    DEFAULT_VIS_SAMPLES = 5
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Path to Ocean Plastic Dataset. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Path to output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--vis_samples",
        type=int,
        default=DEFAULT_VIS_SAMPLES,
        help=f"Number of samples to visualize. Default: {DEFAULT_VIS_SAMPLES}",
    )
    parser.add_argument(
        "--target_classes",
        type=int,
        nargs="+",
        default=None,
    )
    args = parser.parse_args()
    main(args)

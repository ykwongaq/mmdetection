import os
import json
import shutil
import argparse

from typing import List, Dict, Tuple

import sys

sys.path.append("../..")
from utils.dataset_utils import *


def gen_category_datas() -> List[Dict]:
    category_data = {}
    category_data["id"] = 0
    category_data["name"] = "trash"
    category_data["supercategory"] = "trash"
    return [category_data]


def collect_image_and_annotations(json_file: str) -> List[Tuple[Dict, List[Dict]]]:
    json_data = read_json(json_file)
    images = json_data["images"]
    annotations = json_data["annotations"]

    image_id_to_annotations = {}
    for annotation in annotations:
        image_id = annotation["image_id"]
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(annotation)

    image_and_annotations = []
    for image in images:
        image_id = image["id"]
        if image_id in image_id_to_annotations:
            image_and_annotations.append((image, image_id_to_annotations[image_id]))

    return image_and_annotations


def convert_category_id(
    data: List[Tuple[Dict, List[Dict]]], target_classes: List[int]
) -> List[Tuple[Dict, List[Dict]]]:
    converted_data = []
    for images, annotations in data:
        converted_annotations = []
        for annotation in annotations:
            category_id = annotation["category_id"]
            if category_id in target_classes:
                converted_annotation = annotation.copy()
                converted_annotation["category_id"] = 0
                converted_annotations.append(converted_annotation)

        converted_data.append((images, converted_annotations))
    return converted_data


def write_json(data: Dict, json_file: str):
    with open(json_file, "w") as file:
        json.dump(data, file)


class DataMover:
    """
    Use a class to keep track of the idx
    """

    def __init__(self, output_dir):
        self.image_idx = 0
        self.annotation_idx = 0
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.output_train_dir = os.path.join(output_dir, "train")
        self.output_val_dir = os.path.join(output_dir, "val")

        os.makedirs(self.output_train_dir, exist_ok=True)
        os.makedirs(self.output_val_dir, exist_ok=True)

        self.output_train_image_datas = []
        self.output_train_annotation_datas = []

        self.output_val_image_datas = []
        self.output_val_annotation_datas = []

    def move_train_data(self, data, image_folder):
        for image, annotations in data:
            image_path = os.path.join(image_folder, image["file_name"])
            ext = os.path.splitext(image["file_name"])[1]
            new_filename = f"{self.image_idx:06d}{ext}"

            new_image_path = os.path.join(self.output_train_dir, new_filename)
            shutil.copy(image_path, new_image_path)

            for annotation in annotations:
                annotation["image_id"] = self.image_idx
                annotation["id"] = self.annotation_idx
                self.annotation_idx += 1

            image["file_name"] = new_filename
            image["id"] = self.image_idx
            self.image_idx += 1

            self.output_train_image_datas.append(image)
            self.output_train_annotation_datas.extend(annotations)

    def move_val_data(self, data, image_folder):
        for image, annotations in data:
            image_path = os.path.join(image_folder, image["file_name"])
            ext = os.path.splitext(image["file_name"])[1]
            new_filename = f"{self.image_idx:06d}{ext}"

            new_image_path = os.path.join(self.output_val_dir, new_filename)
            shutil.copy(image_path, new_image_path)

            for annotation in annotations:
                annotation["image_id"] = self.image_idx
                annotation["id"] = self.annotation_idx
                self.annotation_idx += 1

            image["file_name"] = new_filename
            image["id"] = self.image_idx
            self.image_idx += 1

            self.output_val_image_datas.append(image)
            self.output_val_annotation_datas.extend(annotations)

    def get_train_json(self):
        return {
            "images": self.output_train_image_datas,
            "annotations": self.output_train_annotation_datas,
            "categories": gen_category_datas(),
        }

    def get_val_json(self):
        return {
            "images": self.output_val_image_datas,
            "annotations": self.output_val_annotation_datas,
            "categories": gen_category_datas(),
        }


def main():
    ocean_plastic_folder = "/mnt/hdd/davidwong/data/trash/Ocean_Plastic"
    ocean_target_classes = [1]

    trash_can_folder = "/mnt/hdd/davidwong/data/trash/TrashCan"
    trash_can_target_classes = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    trash_icra19_folder = "/mnt/hdd/davidwong/data/trash/trash_ICRA19/dataset"
    trash_icra19_target_classes = [1, 3, 4, 5, 7, 9, 10]

    output_dir = "/mnt/hdd/davidwong/data/trash/combined_dataset"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Collecting ocean plastic train data from {ocean_plastic_folder}")
    ocean_train_json_file = os.path.join(ocean_plastic_folder, "train_annotations.json")
    ocean_train_datas = collect_image_and_annotations(ocean_train_json_file)
    ocean_train_datas = convert_category_id(ocean_train_datas, ocean_target_classes)

    print(f"Collecting ocean plastic val data from {ocean_plastic_folder}")
    ocean_val_json_file = os.path.join(ocean_plastic_folder, "val_annotations.json")
    ocean_val_datas = collect_image_and_annotations(ocean_val_json_file)
    ocean_val_datas = convert_category_id(ocean_val_datas, ocean_target_classes)

    print(f"Collecting ocean plastic test data from {ocean_plastic_folder}")
    ocean_test_json_file = os.path.join(ocean_plastic_folder, "test_annotations.json")
    ocean_test_datas = collect_image_and_annotations(ocean_test_json_file)
    ocean_test_datas = convert_category_id(ocean_test_datas, ocean_target_classes)

    print(f"Collecting trash can train data from {trash_can_folder}")
    trash_can_train_json_file = os.path.join(
        trash_can_folder, "instances_train_trashcan.json"
    )
    trash_can_train_datas = collect_image_and_annotations(trash_can_train_json_file)
    trash_can_train_datas = convert_category_id(
        trash_can_train_datas, trash_can_target_classes
    )

    print(f"Collecting trash can val data from {trash_can_folder}")
    trash_can_val_json_file = os.path.join(
        trash_can_folder, "instances_val_trashcan.json"
    )
    trash_can_val_datas = collect_image_and_annotations(trash_can_val_json_file)
    trash_can_val_datas = convert_category_id(
        trash_can_val_datas, trash_can_target_classes
    )

    print(f"Collecting trash icra19 train data from {trash_icra19_folder}")
    trash_icra19_train_json_file = os.path.join(trash_icra19_folder, "train.json")
    trash_icra19_train_datas = collect_image_and_annotations(
        trash_icra19_train_json_file
    )
    trash_icra19_train_datas = convert_category_id(
        trash_icra19_train_datas, trash_icra19_target_classes
    )

    print(f"Collecting trash icra19 val data from {trash_icra19_folder}")
    trash_icra19_val_json_file = os.path.join(trash_icra19_folder, "val.json")
    trash_icra19_val_datas = collect_image_and_annotations(trash_icra19_val_json_file)
    trash_icra19_val_datas = convert_category_id(
        trash_icra19_val_datas, trash_icra19_target_classes
    )

    print(f"Collecting trash icra19 test data from {trash_icra19_folder}")
    trash_icra19_test_json_file = os.path.join(trash_icra19_folder, "test.json")
    trash_icra19_test_datas = collect_image_and_annotations(trash_icra19_test_json_file)
    trash_icra19_test_datas = convert_category_id(
        trash_icra19_test_datas, trash_icra19_target_classes
    )

    output_train_folder = os.path.join(output_dir, "train")
    output_val_folder = os.path.join(output_dir, "val")

    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_val_folder, exist_ok=True)

    mover = DataMover(output_dir)

    print(f"Moving ocean train data to {output_train_folder}")
    mover.move_train_data(
        ocean_train_datas, os.path.join(ocean_plastic_folder, "train")
    )

    print(f"Moving ocean val data to {output_val_folder}")
    mover.move_val_data(ocean_val_datas, os.path.join(ocean_plastic_folder, "valid"))

    print(f"Moving ocean test data to {output_train_folder}")
    mover.move_train_data(ocean_test_datas, os.path.join(ocean_plastic_folder, "test"))

    print(f"Moving trash can train data to {output_train_folder}")
    mover.move_train_data(
        trash_can_train_datas,
        os.path.join(trash_can_folder, "train"),
    )

    print(f"Moving trash can val data to {output_val_folder}")
    mover.move_val_data(
        trash_can_val_datas,
        os.path.join(trash_can_folder, "val"),
    )

    print(f"Moving trash icra19 train data to {output_train_folder}")
    mover.move_train_data(
        trash_icra19_train_datas,
        os.path.join(trash_icra19_folder, "train"),
    )

    print(f"Moving trash icra19 val data to {output_val_folder}")
    mover.move_val_data(
        trash_icra19_val_datas,
        os.path.join(trash_icra19_folder, "val"),
    )

    print(f"Moving trash icra19 test data to {output_train_folder}")
    mover.move_train_data(
        trash_icra19_test_datas,
        os.path.join(trash_icra19_folder, "test"),
    )

    output_train_json = mover.get_train_json()
    output_val_json = mover.get_val_json()

    output_train_json_file = os.path.join(output_dir, "train.json")
    write_json(output_train_json, output_train_json_file)

    output_val_json_file = os.path.join(output_dir, "val.json")
    write_json(output_val_json, output_val_json_file)


if __name__ == "__main__":
    main()

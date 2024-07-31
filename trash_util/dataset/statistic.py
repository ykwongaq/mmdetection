import argparse
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.util import *

def count_images(data: dict) -> int:
    """
    Count the number of images in the dataset
    """
    return len(data["images"])

def count_annotations(data: dict) -> int:
    """
    Count the number of annotations in the dataset
    """
    return len(data["annotations"])

def count_annotations_per_category(data: dict) -> dict:
    """
    Count the number of annotations per category in the dataset
    """
    category_count = {}

    category_name_map = {}
    for category in data["categories"]:
        category_name_map[category["id"]] = category["name"]

    for annotation in data["annotations"]:
        category_id = annotation["category_id"]
        category_name = category_name_map[category_id]
        if category_name in category_count:
            category_count[category_name] += 1
        else:
            category_count[category_name] = 1
    return category_count


def main(args):
    json_file = args.json_path

    print(f"Reading json file from {json_file} ...")
    data = read_json(json_file)

    image_count = count_images(data)
    print(f"Number of images: {image_count}")

    annotation_count = count_annotations(data)
    print(f"Number of annotations: {annotation_count}")

    category_count = count_annotations_per_category(data)
    print("Number of annotations per category:")
    for category, count in category_count.items():
        print(f"{category}: {count}")

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistic of dataset")
    parser.add_argument("--json_path", type=str, help="Path to json file", default="/mnt/hdd/davidwong/data/trash/TrashCan/instances_train_trashcan.json")
    args = parser.parse_args()
    main(args)
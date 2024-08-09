import os
import argparse
import json
import xml.etree.ElementTree as ET
import datetime

from typing import List, Dict, Tuple


def write_json(json_data: Dict, path: str) -> None:
    """
    Write the given json data to the given path
    """
    with open(path, "w") as file:
        json.dump(json_data, file)


def gen_image_data(root) -> Dict:
    """
    Create image data from the given xml root.
    The data is in COCO format.
    Note that the id is not create here.

    Args:
        root: The root of the xml file

    Returns:
        image_data: The image data
    """
    filename = root.find("filename").text
    filename = os.path.basename(filename)

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    license = 1

    # data captured time in coco format
    data_captured = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    image_data = {
        "file_name": filename,
        "width": width,
        "height": height,
        "id": None,
        "license": license,
        "date_captured": data_captured,
    }

    return image_data


def gen_annotations(xml_root, name_to_idx: Dict) -> List[Dict]:
    """
    Create annotation data from the given xml root.
    The data is in COCO format.
    Note that the id is not create here.

    Args:
        xml_root: The root of the xml file
        name_to_idx: A dictionary mapping category name to category id

    Returns:
        annotation_datas: The annotation data
    """
    objects = xml_root.findall("object")
    annotation_datas = []
    for obj in objects:
        category = obj.find("name").text
        if category == "platstic":
            category = "plastic"
        if category == "papper":
            category = "paper"
        category_id = name_to_idx[category]

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        annotation_data = {
            "id": None,
            "image_id": None,
            "category_id": category_id,
            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
            "iscrowd": 0,
            "area": (xmax - xmin) * (ymax - ymin),
        }
        annotation_datas.append(annotation_data)

    return annotation_datas


def gen_image_and_annotation_data(
    xml_file: str, name_to_idx: Dict
) -> Tuple[Dict, List[Dict]]:
    """
    Create image and annotation data from the given xml file.
    The data is in COCO format.
    Note that the id is not create here.

    Args:
        xml_file: The xml file to read
        name_to_idx: A dictionary mapping category name to category

    Returns:
        image_data: The image data
        annotation_datas: The annotation data
    """
    root = read_xml(xml_file)
    image_data = gen_image_data(root)
    annotation_datas = gen_annotations(root, name_to_idx)
    return image_data, annotation_datas


def gen_image_and_annotations_datas(
    xml_files: List[str], name_to_idx: Dict
) -> Tuple[List[Dict], List[Dict]]:
    image_idx = 0
    annotation_idx = 0

    image_data_list = []
    annotation_data_list = []

    for xml_file in xml_files:
        image_data, annotation_datas = gen_image_and_annotation_data(
            xml_file, name_to_idx
        )
        image_data["id"] = image_idx

        for annotation_data in annotation_datas:
            annotation_data["id"] = annotation_idx
            annotation_data["image_id"] = image_idx
            annotation_idx += 1

        image_idx += 1

        image_data_list.append(image_data)
        annotation_data_list.extend(annotation_datas)

    return image_data_list, annotation_data_list


def read_xml(xml_file: str) -> ET.Element:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root


def gen_category_datas(xml_files: List[str]) -> List[Dict]:
    categories = set()
    for xml_file in xml_files:
        root = read_xml(xml_file)
        objects = root.findall("object")
        for obj in objects:
            category = obj.find("name").text
            if category == "platstic":
                category = "plastic"
            if category == "papper":
                category = "paper"
            categories.add(category)

    categories = list(categories)
    categories.sort()
    category_datas = []
    for idx, category in enumerate(categories):
        category_data = {
            "id": idx,
            "name": category,
            "supercategory": category,
        }
        category_datas.append(category_data)

    return category_datas


def gen_ann_json(folder: str, category_datas: Dict) -> dict:
    filenames = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(folder)]
    xml_files = [os.path.join(folder, f"{f}.xml") for f in filenames]
    name_to_idx = {
        category_data["name"]: category_data["id"] for category_data in category_datas
    }

    image_datas, annotation_datas = gen_image_and_annotations_datas(
        xml_files, name_to_idx
    )

    ann_json = {
        "images": image_datas,
        "annotations": annotation_datas,
        "categories": category_datas,
    }

    return ann_json


def main(args):

    data_dir = args.data_dir
    print(f"Data Directory: {data_dir}")

    train_folder = os.path.join(data_dir, "train")

    filenames = [
        os.path.splitext(os.path.basename(f))[0] for f in os.listdir(train_folder)
    ]
    xml_files = [os.path.join(train_folder, f"{f}.xml") for f in filenames]
    category_datas = gen_category_datas(xml_files)

    print(f"Generating annotation json for train folder: {train_folder}")
    train_json = gen_ann_json(train_folder, category_datas)
    output_path = os.path.join(data_dir, "train.json")
    print(f"Writing annotation json to: {output_path}")
    write_json(train_json, output_path)

    val_folder = os.path.join(data_dir, "val")
    print(f"Generating annotation json for val folder: {val_folder}")
    val_json = gen_ann_json(val_folder, category_datas)
    output_path = os.path.join(data_dir, "val.json")
    print(f"Writing annotation json to: {output_path}")
    write_json(val_json, output_path)

    test_folder = os.path.join(data_dir, "test")
    print(f"Generating annotation json for test folder: {test_folder}")
    test_json = gen_ann_json(test_folder, category_datas)
    output_path = os.path.join(data_dir, "test.json")
    print(f"Writing annotation json to: {output_path}")
    write_json(test_json, output_path)


if __name__ == "__main__":
    DEFAULT_DATA_DIR = "/mnt/hdd/davidwong/data/trash/trash_ICRA19/dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Path to Trash Dataset. Default: {DEFAULT_DATA_DIR}",
    )
    args = parser.parse_args()
    main(args)

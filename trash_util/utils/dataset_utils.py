import json
import cv2
import numpy as np

from typing import List, Dict


def read_image(image_path: str) -> np.ndarray:
    """
    Read the given image
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_json(json_path: str) -> dict:
    """
    Read the given json file
    """
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


def get_annotation_by_image_ids(json_data: dict, image_ids: list) -> dict:
    """
    Get annotations by image ids
    """
    annotations = {}
    for annotation in json_data["annotations"]:
        if annotation["image_id"] in image_ids:
            if annotation["image_id"] not in annotations:
                annotations[annotation["image_id"]] = []
            annotations[annotation["image_id"]].append(annotation)
    return annotations


def get_images_by_image_ids(json_data: dict, image_ids: list) -> dict:
    """
    Get images by image ids
    """
    images = {}
    for image in json_data["images"]:
        if image["id"] in image_ids:
            images[image["id"]] = image
    return images


def add_bbox_and_text_to_image(
    image: np.ndarray,
    bbox: List[int],
    text: str,
    line_color: List[int] = [255, 0, 0],
    text_color: List[int] = [255, 0, 0],
    line_thickness: int = 2,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.6,
    font_thickness: int = 2,
) -> np.ndarray:
    """
    Add bounding box and text to the image

    Args:
        image: Image to add bounding box and text
        bbox: Bounding box (x, y, w, h), where x, y are the top-left corner of the bounding box, and w, h are the width and height of the bounding box. Note that they are not normalized
        text: Text to add
    """
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(image, (x, y), (int(x + w), (y + h)), line_color, line_thickness)

    # Calculate text size to make a background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )

    # Draw background rectangle for text
    cv2.rectangle(
        image,
        (x, y - text_height - baseline),
        (x + text_width, y),
        (255, 255, 255),
        cv2.FILLED,
    )

    # Add text
    cv2.putText(
        image,
        text,
        (x, y - baseline),
        font,
        font_scale,
        text_color,
        font_thickness,
    )

    return image


def vis_bbox(
    image: np.ndarray, bbox: List[List[int]], class_names: List[str]
) -> np.ndarray:
    """
    Visualize bounding box on the image

    Args:
        image: Image to visualize
        bbox: List of bounding boxes
        class_names: List of class names

    Returns:
        Image with bounding boxes
    """
    for bbox, class_name in zip(bbox, class_names):
        image = add_bbox_and_text_to_image(image, bbox, class_name)
    return image


def get_id_to_category_names(json_data: dict) -> dict:
    """
    Get id to category names
    """
    id_to_category_names = {}
    for category in json_data["categories"]:
        id_to_category_names[category["id"]] = category["name"]
    return id_to_category_names


def get_annotations_by_category_ids(
    json_data: Dict, category_ids: List[int]
) -> List[Dict]:
    annotations = json_data["annotations"]
    annotations = [
        annotation
        for annotation in annotations
        if annotation["category_id"] in category_ids
    ]
    return annotations


def get_images_by_category_ids(json_data: Dict, category_ids: List[int]) -> List[Dict]:
    annotations = get_annotations_by_category_ids(json_data, category_ids)
    image_ids = [annotation["image_id"] for annotation in annotations]
    images = json_data["images"]
    images = [image for image in images if image["id"] in image_ids]
    return images


def get_image_ids_by_category_ids(
    json_data: Dict, category_ids: List[int]
) -> List[int]:
    annotations = get_annotations_by_category_ids(json_data, category_ids)
    image_ids = [annotation["image_id"] for annotation in annotations]
    return image_ids

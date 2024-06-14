import json
import os
import argparse
import jsonlines


def read_json(json_file: str):
    with open(json_file, "r") as file:
        json_data = json.load(file)
    return json_data


def write_json(json_data, json_file: str):
    with jsonlines.open(json_file, mode="w") as writer:
        writer.write_all(json_data)


def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def main(args):

    json_file = args.json_file

    json_data = read_json(json_file)

    IMAGE_ID_TO_FILE_NAME = {}
    for image in json_data["images"]:
        IMAGE_ID_TO_FILE_NAME[image["id"]] = image["file_name"]

    IMAGE_ID_TO_HEIGHT = {}
    for image in json_data["images"]:
        IMAGE_ID_TO_HEIGHT[image["id"]] = image["height"]

    IMAGE_ID_TO_WIDTH = {}
    for image in json_data["images"]:
        IMAGE_ID_TO_WIDTH[image["id"]] = image["width"]

    CATEGORY_ID_TO_NAME = {}
    for category in json_data["categories"]:
        CATEGORY_ID_TO_NAME[category["id"]] = category["name"]

    metas = []
    for annotation in json_data["annotations"]:
        image_id = annotation["image_id"]
        filename = IMAGE_ID_TO_FILE_NAME[image_id]
        height = IMAGE_ID_TO_HEIGHT[image_id]
        width = IMAGE_ID_TO_WIDTH[image_id]

        caption = annotation["caption"]
        phrase = caption
        tokens_positive = [[0, len(phrase) - 1]]

        bbox = annotation["bbox"]
        bbox = xywh_to_xyxy(bbox)

        meta = {
            "filename": filename,
            "height": height,
            "width": width,
            "grounding": {
                "caption": caption,
                "regions": [
                    {
                        "bbox": bbox,
                        "phrase": phrase,
                        "tokens_positive": tokens_positive,
                    }
                ],
            },
        }

        metas.append(meta)

    output_file = json_file.replace(".json", "_processed.json")
    write_json(metas, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON file")
    parser.add_argument("--json_file", type=str, help="Path to JSON file")
    args = parser.parse_args()
    main(args)

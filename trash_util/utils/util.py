import json

def read_json(json_path: str) -> dict:
    """
    Read the given json file
    """
    with open(json_path, "r") as file:
        data = json.load(file)
    return data
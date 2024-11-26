import json
from pathlib import Path

def extract_data_from_json(json_file: str) -> list[dict]:
    """
    Extracts data from a JSON file and returns it as a list of dictionaries.

    Parameters:
        json_file (str): The JSON file to read. This can be:
                                      - A string representing the file path.
                                      - A Path object from the pathlib library.
    Returns:
        list: A list of dictionaries containing the extracted data, or an empty list if the file is not found or invalid.
    """
    # Convert Path to string if necessary
    if isinstance(json_file, Path):
        json_file = str(json_file)
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{json_file}' does not exist.")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file}' is not a valid JSON file.")
        return []

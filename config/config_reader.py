# config_reader.py

from pathlib import Path
import yaml

def config_read_yaml(yaml_file="config.yml", base_dir=None):
    """
    Reads a YAML configuration file and returns the loaded data.

    Parameters:
    - yaml_file (str): the filename of the YAML file to read.
    - base_dir (str, optional): the base directory where the YAML file is located. If not specified, it uses the directory of the script.

    Returns:
    - dict: the data loaded from the YAML file.
    """

    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    else:
        base_dir = Path(base_dir) 

    yaml_path = base_dir / yaml_file
    
    try:
        with open(yaml_path, "r") as fp:
            return yaml.safe_load(fp)
    except FileNotFoundError:
        print(f"Error: The file {yaml_path} was not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
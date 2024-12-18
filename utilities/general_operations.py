from pathlib import Path

def create_directory(dir_out: str, gitkeep: bool = True) -> None:
    """
    Creates the specified directory and an empty .gitkeep file within it.

    Parameters:
        dir_out (str): The path to the directory to be created.
        gitkeep (bool): If True, create an empty .gitkeep file.
    Returns:
        None
    """
    # Convert dir_out to a Path object
    directory_path = Path(dir_out)

    # Check if directory already exists
    if directory_path.exists():
        print(f"The folder '{directory_path}' already exists.")
    else:
        # Create the directory, including any necessary intermediate directories
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {directory_path}")

    # Create the .gitkeep file within the directory
    if gitkeep:
        gitkeep_file = directory_path / ".gitkeep"
        if gitkeep_file.exists():
            print(f"The file '{gitkeep_file}' already exists.")
        else:
            gitkeep_file.touch()
            print(f"File .gitkeep created: {gitkeep_file}")

from pathlib import Path

def create_nested_directory(dir_parts: list, gitkeep: bool = True) -> None:
    """
    Creates a nested directory structure from a list of directory names 
    and optionally adds a .gitkeep file at the deepest level.

    Parameters:
        dir_parts (list[str]): List of directory names representing the nested path.
        gitkeep (bool): If True, create an empty .gitkeep file at the deepest level.
    Returns:
        None
    """
    if not dir_parts:
        print("Error: The list of directory parts is empty.")
        return

    # Construct the full path from the list of directory parts
    directory_path = Path(*dir_parts)

    # Check if the directory already exists
    if directory_path.exists():
        print(f"The folder '{directory_path}' already exists.")
    else:
        # Create the directory, including any necessary intermediate directories
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {directory_path}")

    # Optionally create the .gitkeep file at the deepest level
    if gitkeep:
        gitkeep_file = directory_path / ".gitkeep"
        if gitkeep_file.exists():
            print(f"The file '{gitkeep_file}' already exists.")
        else:
            gitkeep_file.touch()
            print(f"File .gitkeep created: {gitkeep_file}")
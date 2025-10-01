import os
from typing import Union

def rename_files(folder_path: str) -> None:
    """
    Renames files in the specified folder. Files starting with 'date_' are renamed to a sequential format 'cam1 (i).png',
    'cam2 (i).png', etc., where i is the index in the sorted list of files.

    Args:
        folder_path (str): The path to the folder containing files to rename.

    Returns:
        None
    """
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Sort files to ensure consistent numbering
    files.sort()
    
    # Initialize counter
    i = 1  # Change this to the next number in the sequence
    
    # Iterate through the files and rename them
    for filename in files:
        # Check if the filename starts with 'date_'
        if filename.startswith('date_'):
            # Construct the new file name
            new_filename = f'pi8_{filename}'  # Change this to pi1, pi2, etc. as needed
            
            # Get the full file path
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            
            # Rename the file
            os.rename(old_file, new_file)
            
            # Increment the counter
            i += 1

if __name__ == "__main__":
    folder_path = 'path/to/folder'  # Change this to the path of the folder containing the files
    rename_files(folder_path)
import os
import pandas as pd
from typing import Union

def reorder_csv_files(input_dir: str) -> None:
    """
    Processes CSV files in the specified directory. Removes rows with NaN values in 'Center_X' or 'Center_Y',
    sorts the data by 'Filename', and saves the cleaned and sorted data back to the original CSV files.

    Args:
        input_dir (str): The path to the directory containing CSV files to process.

    Returns:
        None
    """
    # Define the naming conventions and range
    prefixes = [f'cam{i}' for i in range(1, 9)]
    suffixes = ['_nir.csv', '_rgb.csv']

    # Get list of CSV files in the directory
    all_files = os.listdir(input_dir)
    csv_files = [f for f in all_files if any(f.startswith(prefix) and f.endswith(suffix) for prefix in prefixes for suffix in suffixes)]

    # Process each CSV file
    for csv_file in csv_files:
        # Read the CSV file
        file_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(file_path)
        
        # Remove rows with NaN in Center_X or Center_Y
        df = df.dropna(subset=['Center_X', 'Center_Y'])
        
        # Sort by Filename
        df = df.sort_values(by='Filename')
        
        # Save the cleaned and sorted dataframe back to the CSV file
        df.to_csv(file_path, index=False)
        
        print(f"Processed {csv_file}")

if __name__ == "__main__":
    input_dir = '../../data/2024_outputs/panel_detection_output/csv_outputs'  # Change this to the correct directory path
    reorder_csv_files(input_dir)
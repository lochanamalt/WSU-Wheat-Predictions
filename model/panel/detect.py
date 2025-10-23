import cv2
from pathlib import Path
from ultralytics import YOLO
import csv
from typing import List, Tuple
import os

from paths import RAW_IMG_DIR

# Load the YOLOv8 model
model: YOLO = YOLO('model/panel/best.pt')


def process_images(input_folder: str, output_folder: str, centers_list: List[Tuple[str, int, int, int, int]]) -> None:
    """Process 2024_images from the input folder and save the processed 2024_images to the output folder.
    
    Args:
        input_folder (str): The path to the input folder containing 2024_images.
        output_folder (str): The path to the output folder where processed 2024_images will be saved.
        centers_list (List[Tuple[str, int, int, int, int]]): A list to store the filename and center coordinates of detected objects.
    """
    input_path: Path = Path(input_folder)
    output_path: Path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_path in input_path.glob('*.png'):
        print(f'Processing {img_path}')
        img = cv2.imread(str(img_path))
        results = model(img)

        for result in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2 = map(int, result[:4])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            one_side_length = 10
            x1_new = int(center_x - one_side_length / 2)
            y1_new = int(center_y - one_side_length / 2)
            x2_new = int(center_x + one_side_length / 2)
            y2_new = int(center_y + one_side_length / 2)
            cv2.rectangle(img, (x1_new, y1_new), (x2_new, y2_new), (0, 0, 255), 2)
            centers_list.append((img_path.name, center_x, center_y, 10, 10))

        cv2.imwrite(str(output_path / img_path.name), img)
        print(f"Processed {img_path.name} with center at ({center_x}, {center_y})")

def save_to_csv(centers_list: List[Tuple[str, int, int, int, int]], output_csv: str) -> None:
    """Save the center coordinates to a CSV file.
    
    Args:
        centers_list (List[Tuple[str, int, int, int, int]]): The list of center coordinates to be saved.
        output_csv (str): The path to the output CSV file.
    """
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Center_X', 'Center_Y', 'Width', 'Height'])
        writer.writerows(centers_list)
    print(f"Saved center coordinates to {output_csv}")

def split_csv(input_csv: str, nir_csv: str, rgb_csv: str) -> None:
    """Split the CSV file into NIR and RGB based on the Center_X value.
    
    Args:
        input_csv (str): The path to the input CSV file containing center coordinates.
        nir_csv (str): The path to the output CSV file for NIR data.
        rgb_csv (str): The path to the output CSV file for RGB data.
    """
    nir_centers: List[List[str]] = []
    rgb_centers: List[List[str]] = []

    with open(input_csv, mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        for row in reader:
            center_x = int(row[1])
            if center_x < 640:
                nir_centers.append(row)
            else:
                row[1] = str(center_x - 640)
                rgb_centers.append(row)

    with open(nir_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(nir_centers)
    print(f"Saved NIR center coordinates to {nir_csv}")

    with open(rgb_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rgb_centers)
    print(f"Saved RGB center coordinates to {rgb_csv}")

for i in range(1, 9):
    project_root = os.path.dirname(os.path.abspath(__file__))

    centers_list: List[Tuple[str, int, int, int, int]] = []  # Reset centers_list for each new camera folder
    process_images(os.path.join(RAW_IMG_DIR, f'cam{i}'),
                   os.path.join(project_root, f'../../data/2024_outputs/panel_detection_output/cam{i}'), centers_list)

    combined_csv: str = os.path.join(project_root, f'../../data/2024_outputs/panel_detection_output/csv_outputs/cam{i}.csv')

    save_to_csv(centers_list, combined_csv)

    nir_csv: str = os.path.join(project_root, f'../../data/2024_outputs/panel_detection_output/csv_outputs/cam{i}_nir.csv')
    rgb_csv: str = os.path.join(project_root, f'../../data/2024_outputs/panel_detection_output/csv_outputs/cam{i}_rgb.csv')
    split_csv(combined_csv, nir_csv, rgb_csv)

print("Processing complete.")

import csv
import os
from typing import Dict, Tuple

import cv2
import numpy as np
from numpy.random.mtrand import Sequence

# Calibration data for 8 cameras
calibration_data_nir: Dict[str, Dict[str, float]] = {
    'cam1': {'Blue': 10.77, 'Green': 12.57, 'Red': 12.24},
    'cam2': {'Blue': 12.18, 'Green': 13.82, 'Red': 11.67},
    'cam3': {'Blue': 11.58, 'Green': 12.63, 'Red': 11.35},
    'cam4': {'Blue': 11.27, 'Green': 12.49, 'Red': 10.99},
    'cam5': {'Blue': 11.22, 'Green': 14.87, 'Red': 12.59},
    'cam6': {'Blue': 11.13, 'Green': 12.04, 'Red': 11.10},
    'cam7': {'Blue': 11.36, 'Green': 12.53, 'Red': 11.35},
    'cam8': {'Blue': 11.00, 'Green': 14.01, 'Red': 12.27}
}
calibration_data_rgb: Dict[str, Dict[str, float]] = {
    'cam1': {'Blue': 11.02, 'Green': 12.55, 'Red': 12.85},
    'cam2': {'Blue': 12.97, 'Green': 13.88, 'Red': 13.31},
    'cam3': {'Blue': 11.79, 'Green': 12.54, 'Red': 12.19},
    'cam4': {'Blue': 11.92, 'Green': 12.62, 'Red': 12.11},
    'cam5': {'Blue': 11.92, 'Green': 14.69, 'Red': 14.72},
    'cam6': {'Blue': 11.64, 'Green': 12.28, 'Red': 11.86},
    'cam7': {'Blue': 12.13, 'Green': 12.87, 'Red': 12.26},
    'cam8': {'Blue': 11.72, 'Green': 13.99, 'Red': 14.05}
}


def radiometric_correction(img: np.ndarray,
                           lut_values: Dict[str, float],
                           ref_panel_bgr: Sequence[float]) -> np.ndarray:
    """
    Applies radiometric correction to an image based on camera calibration data.

    Args:
        img (np.ndarray): The input image array.
        lut_values (Dict[str, float]): Lookup table values for correction.
        ref_panel_bgr (Sequence[float]): Reference panel BGR values.

    Returns:
        np.ndarray: The radiometric corrected image.
    """

    (img_original_B, img_original_G, img_original_R, img_A) = cv2.split(img)

    blue_band_correction_factor = (lut_values['Blue'] * 0.01 * 255) / ref_panel_bgr[0]
    green_band_correction_factor = (lut_values['Green'] * 0.01 * 255) / ref_panel_bgr[1]
    red_band_correction_factor = (lut_values['Red'] * 0.01 * 255) / ref_panel_bgr[2]

    img_corrected_B = blue_band_correction_factor * img_original_B
    img_corrected_G = green_band_correction_factor * img_original_G
    img_corrected_R = red_band_correction_factor * img_original_R

    # Clip values to ensure they are within the correct range
    return cv2.merge([img_corrected_B, img_corrected_G, img_corrected_R])



def apply_correction_to_all_images(input_directory: str, output_directory: str, csv_dir: str) -> None:
    """
    Applies radiometric correction to all images in the input directory
    and saves the corrected images to the output directory.

    Args:
        input_directory (str): Path to the directory containing input images.
        output_directory (str): Path to the directory where corrected images will be saved.

    Returns:
        None
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for cam_name in calibration_data_nir.keys():
        cam_nir_input_dir = os.path.join(input_directory, f'{cam_name}_nir')
        cam_rgb_input_dir = os.path.join(input_directory, f'{cam_name}_rgb')
        cam_nir_output_dir = os.path.join(output_directory, f'{cam_name}_nir')
        cam_rgb_output_dir = os.path.join(output_directory, f'{cam_name}_rgb')
        cam_nir_csv = f"{csv_dir}\\{cam_name}_nir.csv"
        cam_rgb_csv = f"{csv_dir}\\{cam_name}_rgb.csv"

        if not os.path.exists(cam_nir_output_dir):
            os.makedirs(cam_nir_output_dir)
        if not os.path.exists(cam_rgb_output_dir):
            os.makedirs(cam_rgb_output_dir)

        lut_values_nir = calibration_data_nir[cam_name]
        lut_values_rgb = calibration_data_rgb[cam_name]

        # process nir csv
        image_correction_per_camera_csv_file(cam_nir_csv, cam_nir_input_dir, cam_nir_output_dir, lut_values_nir)

        # process rgb csv
        image_correction_per_camera_csv_file(cam_rgb_csv, cam_rgb_input_dir, cam_rgb_output_dir, lut_values_rgb)


def image_correction_per_camera_csv_file(cam_csv, cam_input_dir, cam_output_dir, lut_values):
    with open(cam_csv, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row['Filename']
            img_path = os.path.join(cam_input_dir, filename)
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            # Compute ROI boundaries
            mean_bgr = compute_reference_panel_mean_digital_numbers(image, row)


            # Apply radiometric correction
            img_corrected = radiometric_correction(image, lut_values, mean_bgr)

            # Save the corrected image
            output_path = os.path.join(cam_output_dir, filename)
            cv2.imwrite(output_path, img_corrected)
            print(f"Saved corrected image: {output_path}")


def compute_reference_panel_mean_digital_numbers(image, row):
    half_side = int(row['Width']) // 2
    x_start = max(int(row['Center_X']) - half_side, 0)
    x_end = min(int(row['Center_X']) + half_side, image.shape[1])
    y_start = max(int(row['Center_Y']) - half_side, 0)
    y_end = min(int(row['Center_Y']) + half_side, image.shape[0])
    roi = image[y_start:y_end, x_start:x_end]
    (roi_B, roi_G, roi_R, roi_A) = cv2.split(roi)

    # Compute average for each band
    mean_per_band =  [np.mean(roi_B), np.mean(roi_G), np.mean(roi_R)]
    return mean_per_band


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))

    input_dir = os.path.join(project_root, '..\..\data\images')
    output_dir = os.path.join(project_root, '..\..\data\corrected_images')
    csv_folder = os.path.join(project_root, '..\..\panel_detection_output\csv_outputs')

    # Apply radiometric corrections to all images in the dataset
    apply_correction_to_all_images(input_dir, output_dir, csv_folder)

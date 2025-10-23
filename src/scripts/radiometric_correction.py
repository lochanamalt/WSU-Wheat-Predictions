import csv
import os
from typing import Dict, Tuple, Any, List

import cv2
import numpy as np
from numpy import ndarray
from numpy.random.mtrand import Sequence

from paths import ALIGNED_RGB_DIR, RAW_IMG_DIR, CORRECTED_DN_IMG_DIR, CORRECTED_RF_IMG_DIR, PANEL_DETECT_CSV_OUTPUT

# Calibration data for 8 cameras
# calibration_data_nir: Dict[str, Dict[str, float]] = {
#     'cam1': {'Blue': 10.77, 'Green': 12.57, 'Red': 12.24},
#     'cam2': {'Blue': 12.18, 'Green': 13.82, 'Red': 11.67},
#     'cam3': {'Blue': 11.58, 'Green': 12.63, 'Red': 11.35},
#     'cam4': {'Blue': 11.27, 'Green': 12.49, 'Red': 10.99},
#     'cam5': {'Blue': 11.22, 'Green': 14.87, 'Red': 12.59},
#     'cam6': {'Blue': 11.13, 'Green': 12.04, 'Red': 11.10},
#     'cam7': {'Blue': 11.36, 'Green': 12.53, 'Red': 11.35},
#     'cam8': {'Blue': 11.00, 'Green': 14.01, 'Red': 12.27}
# }
calibration_data_nir_new: Dict[str, Dict[str, float]] = {
    'cam1': {'Blue': 11.03, 'Green': 12.58, 'Red': 12.30},
    'cam2': {'Blue': 12.47, 'Green': 13.84, 'Red': 11.73},
    'cam3': {'Blue': 11.86, 'Green': 12.65, 'Red': 11.41},
    'cam4': {'Blue': 11.54, 'Green': 12.51, 'Red': 11.04},
    'cam5': {'Blue': 11.49, 'Green': 14.89, 'Red': 12.66},
    'cam6': {'Blue': 11.40, 'Green': 12.05, 'Red': 11.15},
    'cam7': {'Blue': 11.63, 'Green': 12.55, 'Red': 11.41},
    'cam8': {'Blue': 11.27, 'Green': 14.03, 'Red': 12.33}
}
# calibration_data_rgb: Dict[str, Dict[str, float]] = {
#     'cam1': {'Blue': 11.02, 'Green': 12.55, 'Red': 12.85},
#     'cam2': {'Blue': 12.97, 'Green': 13.88, 'Red': 13.31},
#     'cam3': {'Blue': 11.79, 'Green': 12.54, 'Red': 12.19},
#     'cam4': {'Blue': 11.92, 'Green': 12.62, 'Red': 12.11},
#     'cam5': {'Blue': 11.92, 'Green': 14.69, 'Red': 14.72},
#     'cam6': {'Blue': 11.64, 'Green': 12.28, 'Red': 11.86},
#     'cam7': {'Blue': 12.13, 'Green': 12.87, 'Red': 12.26},
#     'cam8': {'Blue': 11.72, 'Green': 13.99, 'Red': 14.05}
# }
calibration_data_rgb_new: Dict[str, Dict[str, float]] = {
    'cam1': {'Blue': 11.14, 'Green': 12.54, 'Red': 12.85},
    'cam2': {'Blue': 13.12, 'Green': 13.86, 'Red': 13.32},
    'cam3': {'Blue': 11.92, 'Green': 12.52, 'Red': 12.19},
    'cam4': {'Blue': 12.05, 'Green': 12.60, 'Red': 12.12},
    'cam5': {'Blue': 12.06, 'Green': 14.66, 'Red': 14.72},
    'cam6': {'Blue': 11.77, 'Green': 12.26, 'Red': 11.87},
    'cam7': {'Blue': 12.27, 'Green': 12.85, 'Red': 12.26},
    'cam8': {'Blue': 11.85, 'Green': 13.97, 'Red': 14.05}
}

def radiometric_correction(img: np.ndarray,
                           lut_values: Dict[str, float],
                           ref_panel_bgr: Sequence[float]) -> tuple[ndarray | Any, ndarray | Any]:
    """
    Applies radiometric correction to an image based on camera calibration data.

    Args:
        img (np.ndarray): The input image array.
        lut_values (Dict[str, float]): Lookup table values for correction.
        ref_panel_bgr (Sequence[float]): Reference panel BGR values.

    Returns:
        np.ndarray: The radiometric corrected image.
    """

    channels = cv2.split(img)

    if len(channels) == 3:
        (img_original_B, img_original_G, img_original_R) = cv2.split(img)
    else:
        (img_original_B, img_original_G, img_original_R, img_original_A) = cv2.split(img)



    blue_band_correction_factor = (lut_values['Blue'] * 0.01 * 255) / ref_panel_bgr[0]
    green_band_correction_factor = (lut_values['Green'] * 0.01 * 255) / ref_panel_bgr[1]
    red_band_correction_factor = (lut_values['Red'] * 0.01 * 255) / ref_panel_bgr[2]

    img_corrected_B_DN = blue_band_correction_factor * img_original_B
    img_corrected_G_DN = green_band_correction_factor * img_original_G
    img_corrected_R_DN = red_band_correction_factor * img_original_R

    img_corrected_B_RF = img_corrected_B_DN / 255
    img_corrected_G_RF = img_corrected_G_DN / 255
    img_corrected_R_RF = img_corrected_R_DN / 255

    # Clip values to ensure they are within the correct range
    return (cv2.merge([img_corrected_B_DN, img_corrected_G_DN, img_corrected_R_DN]),
            cv2.merge([img_corrected_B_RF, img_corrected_G_RF, img_corrected_R_RF]))



def apply_correction_to_all_images(input_directory: str, output_directory_digital_number: str,
                                   output_directory_reflectance_value: str, csv_dir: str) -> None:
    """
    Applies radiometric correction to all images in the input directory
    and saves the corrected images to the output directory.

    Args:
        input_directory (str): Path to the directory containing input images.
        output_directory_digital_number (str): Path to the directory where corrected images with digital number will be saved.
        output_directory_reflectance_value (str): Path to the directory where corrected images with reflectance value will be saved.
        csv_dir (str): Path to the directory where csv file with reflectance panel coordinates is located.

    Returns:
        None
    """
    if not os.path.exists(output_directory_digital_number):
        os.makedirs(output_directory_digital_number)

    for cam_name in calibration_data_nir_new.keys():
        cam_nir_input_dir = os.path.join(input_directory, f'{cam_name}_nir')
        cam_rgb_input_dir = os.path.join(input_directory, f'{cam_name}_rgb')
        cam_nir_dn_output_dir = os.path.join(output_directory_digital_number, f'{cam_name}_nir')
        cam_rgb_dn_output_dir = os.path.join(output_directory_digital_number, f'{cam_name}_rgb')
        cam_nir_rf_output_dir = os.path.join(output_directory_reflectance_value, f'{cam_name}_nir')
        cam_rgb_rf_output_dir = os.path.join(output_directory_reflectance_value, f'{cam_name}_rgb')

        cam_nir_csv = f"{csv_dir}\\{cam_name}_nir.csv"
        cam_rgb_csv = f"{csv_dir}\\{cam_name}_rgb.csv"

        if not os.path.exists(cam_nir_dn_output_dir):
            os.makedirs(cam_nir_dn_output_dir)
        if not os.path.exists(cam_rgb_dn_output_dir):
            os.makedirs(cam_rgb_dn_output_dir)
        if not os.path.exists(cam_nir_rf_output_dir):
            os.makedirs(cam_nir_rf_output_dir)
        if not os.path.exists(cam_rgb_rf_output_dir):
            os.makedirs(cam_rgb_rf_output_dir)

        lut_values_nir = calibration_data_nir_new[cam_name]
        lut_values_rgb = calibration_data_rgb_new[cam_name]

        panel_reflectance_nir_csv: str = f"{csv_dir}\\{cam_name}_panel_reflectance_nir.csv"
        panel_reflectance_rgb_csv: str = f"{csv_dir}\\{cam_name}_panel_reflectance_rgb.csv"
        # process nir csv
        image_correction_per_camera_csv_file(cam_nir_csv, cam_nir_input_dir, cam_nir_dn_output_dir,
                                             cam_nir_rf_output_dir, lut_values_nir, panel_reflectance_nir_csv)
        # process rgb csv
        image_correction_per_camera_csv_file(cam_rgb_csv, cam_rgb_input_dir,
                                             cam_rgb_dn_output_dir, cam_rgb_rf_output_dir, lut_values_rgb,panel_reflectance_rgb_csv)


def image_correction_per_camera_csv_file(cam_csv, cam_input_dir, cam_output_dir_dn, cam_output_dir_rf, lut_values,
                                         panel_reflectance_csv):
    panel_reflectances = []
    with open(cam_csv, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row['Filename']
            img_path = os.path.join(cam_input_dir, filename)
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            # Compute ROI boundaries
            mean_bgr = compute_reference_panel_mean_digital_numbers(image, row)

            panel_reflectances.append((filename, mean_bgr[0],mean_bgr[1], mean_bgr[2]))

            # Apply radiometric correction
            img_corrected_DN, img_corrected_RF = radiometric_correction(image, lut_values, mean_bgr)

            # Save the corrected image
            output_path_dn = os.path.join(cam_output_dir_dn, filename + '.tif')
            output_path_rf = os.path.join(cam_output_dir_rf, filename + '.tif')
            cv2.imwrite(output_path_dn, img_corrected_DN)
            cv2.imwrite(output_path_rf, img_corrected_RF)

            print(f"Saved corrected image with digital numbers: {output_path_dn}")
            print(f"Saved corrected image with reflectance values: {output_path_rf}")
    save_to_csv(panel_reflectances, panel_reflectance_csv)

def save_to_csv(rf_values: List[Tuple[str, float, float, float]], output_csv: str) -> None:
    """Save the center coordinates to a CSV file.

    Args:
        rf_values (List[Tuple[str, float, float, float]]: The list of reflectance values of panels
        output_csv (str): The path to the output CSV file.
    """
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'B', 'G', 'R'])
        writer.writerows(rf_values)
    print(f"Saved center coordinates to {output_csv}")


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


def correct_missing_panel_images_per_camera(cam_input_dir, cam_output_dir_dn, cam_output_dir_rf,
                                            lut_values, processed_filenames, other_cams_reflectance_file_list, is_aligned_images: bool):

    for filename in os.listdir(cam_input_dir):
        if is_aligned_images:
            correct_filename = filename.split('aligned_')[1]
        else:
            correct_filename = filename

        print(correct_filename)

        if correct_filename.endswith('.png') and correct_filename not in processed_filenames :
            print(f"Processing {correct_filename}")

            data_and_hour_of_filename = correct_filename.split('.')[0]

            matched_reflectance_row = None

            for reflectance_filenames in other_cams_reflectance_file_list:
                with open(reflectance_filenames, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row and row['Filename'].startswith(data_and_hour_of_filename):  # assuming first column
                            matched_reflectance_row = row
                            print("Match found")
                            break
                if matched_reflectance_row:
                    break

            if matched_reflectance_row is None:
                print(f"No reflectance row found for {filename}")
            else:

                img_path = os.path.join(cam_input_dir, filename)
                image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                mean_R = float(matched_reflectance_row["R"])
                mean_G = float(matched_reflectance_row["G"])
                mean_B = float(matched_reflectance_row["B"])

                img_corrected_DN, img_corrected_RF = radiometric_correction(image, lut_values, [mean_B, mean_G, mean_R])

                # Save the corrected image
                output_path_dn = os.path.join(cam_output_dir_dn, correct_filename + '.tif')
                output_path_rf = os.path.join(cam_output_dir_rf, correct_filename + '.tif')
                cv2.imwrite(output_path_dn, img_corrected_DN)
                cv2.imwrite(output_path_rf, img_corrected_RF)

                print(f"Saved corrected image with digital numbers: {output_path_dn}")
                print(f"Saved corrected image with reflectance values: {output_path_rf}")



def correct_missing_panel_images(input_dir, output_dir_dn, output_dir_rf, csv_folder):
    for cam_name in calibration_data_nir_new.keys():
        cam_nir_input_dir = os.path.join(input_dir, f'{cam_name}_nir')
        cam_rgb_input_dir = os.path.join(input_dir, f'{cam_name}_rgb')
        cam_nir_dn_output_dir = os.path.join(output_dir_dn, f'{cam_name}_nir')
        cam_rgb_dn_output_dir = os.path.join(output_dir_dn, f'{cam_name}_rgb')
        cam_nir_rf_output_dir = os.path.join(output_dir_rf, f'{cam_name}_nir')
        cam_rgb_rf_output_dir = os.path.join(output_dir_rf, f'{cam_name}_rgb')

        lut_values_nir = calibration_data_nir_new[cam_name]
        lut_values_rgb = calibration_data_rgb_new[cam_name]

        panel_reflectance_nir_csv: str = f"{csv_folder}\\{cam_name}_panel_reflectance_nir.csv"

        other_cams = [k for k in calibration_data_nir_new.keys() if k != cam_name]
        other_cams_nir_reflectance_file_list = [f"{csv_folder}\\{cam}_panel_reflectance_nir.csv" for cam in other_cams]
        other_cams_rgb_reflectance_file_list = [f"{csv_folder}\\{cam}_panel_reflectance_rgb.csv" for cam in other_cams]


        processed_filenames = get_processed_images(panel_reflectance_nir_csv)

        correct_missing_panel_images_per_camera(cam_nir_input_dir, cam_nir_dn_output_dir, cam_nir_rf_output_dir,
                                                lut_values_nir, processed_filenames,
                                                other_cams_nir_reflectance_file_list, False)
        correct_missing_panel_images_per_camera(cam_rgb_input_dir, cam_rgb_dn_output_dir, cam_rgb_rf_output_dir,
                                                lut_values_rgb, processed_filenames,
                                                other_cams_rgb_reflectance_file_list, False)

def get_processed_images(filename):
    images_list = []

    with open(filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            images_list.append(row["Filename"])  # row[0] is the first column

    return images_list


def correct_aligned_rgb_per_camera(aligned_rgb_input_dir, cam_rgb_dn_output_dir, cam_rgb_rf_output_dir,
                                   panel_reflectance_rgb_csv, lut_values_rgb, failed_images_list):
    with open(panel_reflectance_rgb_csv, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row)
            filename = row['Filename']
            roi_B = float(row['B'])
            roi_G = float(row['G'])
            roi_R = float(row['R'])

            img_path = os.path.join(aligned_rgb_input_dir, f"aligned_{filename}")
            print("Processing image:", filename)
            if os.path.exists(img_path):
                image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                # Apply radiometric correction
                img_corrected_DN, img_corrected_RF = radiometric_correction(image, lut_values_rgb, [roi_B, roi_G, roi_R])

                # Save the corrected image
                output_path_dn = os.path.join(cam_rgb_dn_output_dir, filename + '.tif')
                output_path_rf = os.path.join(cam_rgb_rf_output_dir, filename + '.tif')
                cv2.imwrite(output_path_dn, img_corrected_DN)
                cv2.imwrite(output_path_rf, img_corrected_RF)

                print(f"Saved corrected image with digital numbers: {output_path_dn}")
                print(f"Saved corrected image with reflectance values: {output_path_rf}")
            else:
                failed_images_list.append(img_path)


def correct_aligned_rgb_images(aligned_rgb_directory, output_dir_dn, output_dir_rf, csv_folder):
    failed_images = []
    for cam_no in range(1, 9):
        aligned_rgb_input_dir = os.path.join(aligned_rgb_directory, f'cam{cam_no}_rgb')
        cam_rgb_dn_output_dir = os.path.join(output_dir_dn, f'cam{cam_no}_rgb_aligned')
        cam_rgb_rf_output_dir = os.path.join(output_dir_rf, f'cam{cam_no}_rgb_aligned')

        if not os.path.exists(cam_rgb_dn_output_dir):
            os.makedirs(cam_rgb_dn_output_dir)

        if not os.path.exists(cam_rgb_rf_output_dir):
            os.makedirs(cam_rgb_rf_output_dir)

        lut_values_rgb = calibration_data_rgb_new[f"cam{cam_no}"]

        panel_reflectance_rgb_csv: str = f"{csv_folder}\\cam{cam_no}_panel_reflectance_rgb.csv"

        correct_aligned_rgb_per_camera(aligned_rgb_input_dir, cam_rgb_dn_output_dir, cam_rgb_rf_output_dir,
                                       panel_reflectance_rgb_csv, lut_values_rgb, failed_images)

    print("failed images: ", failed_images)


def correct_aligned_rgb_missing_panel_images(aligned_rgb_directory, output_dir_dn, output_dir_rf, csv_folder):
    for cam_name in calibration_data_rgb_new.keys():
        cam_rgb_input_dir = os.path.join(aligned_rgb_directory, f'{cam_name}_rgb')
        cam_rgb_dn_output_dir = os.path.join(output_dir_dn, f'{cam_name}_rgb_aligned')
        cam_rgb_rf_output_dir = os.path.join(output_dir_rf, f'{cam_name}_rgb_aligned')

        lut_values_rgb = calibration_data_rgb_new[cam_name]

        panel_reflectance_rgb_csv: str = f"{csv_folder}\\{cam_name}_panel_reflectance_rgb.csv"

        other_cams = [k for k in calibration_data_rgb_new.keys() if k != cam_name]
        other_cams_rgb_reflectance_file_list = [f"{csv_folder}\\{cam}_panel_reflectance_rgb.csv" for cam in other_cams]

        processed_filenames = get_processed_images(panel_reflectance_rgb_csv)

        correct_missing_panel_images_per_camera(cam_rgb_input_dir, cam_rgb_dn_output_dir, cam_rgb_rf_output_dir,
                                                lut_values_rgb, processed_filenames,
                                                other_cams_rgb_reflectance_file_list, True)


if __name__ == "__main__":
    input_dir = RAW_IMG_DIR
    aligned_rgb_directory = ALIGNED_RGB_DIR
    output_dir_dn = CORRECTED_DN_IMG_DIR
    output_dir_rf = CORRECTED_RF_IMG_DIR
    csv_folder = PANEL_DETECT_CSV_OUTPUT

    # Apply radiometric corrections to all raw images
    # To correct the aligned rgb images, we need to identify the panel reflectance of the rgb images that are not aligned
    # This function corrects and saves the panel reflectance of both noir and rgb images,
    #apply_correction_to_all_images(input_dir, output_dir_dn, output_dir_rf, csv_folder)

    # There are some images that does not have reflectance panels.
    # The function below is to correct those images based on another camera's panel reflectance values captured on the same day, same time.
    #correct_missing_panel_images(input_dir, output_dir_dn, output_dir_rf,csv_folder)

    # The following functions are to correct the aligned rgb images based on saved rgb panel reflectance values
    correct_aligned_rgb_images(aligned_rgb_directory, output_dir_dn, output_dir_rf, csv_folder)
    correct_aligned_rgb_missing_panel_images(aligned_rgb_directory, output_dir_dn, output_dir_rf, csv_folder)


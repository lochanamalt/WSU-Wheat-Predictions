import os
from datetime import datetime

import cv2

from paths import RAW_IMG_DIR, YEAR


def split_images(input_folder: str, output_folder: str) -> None:
    """
    Processes NIR images from camera folders, extracting and saving the left half of each image.

    Args:
        input_folder (str): The path to the folder containing camera subfolders with NIR images.
        output_folder (str): The path to the folder where the left halves of the NIR images will be saved.

    Returns:
        None
    """

    pi_prefix = "cam"
    number_of_cams = 8
    if YEAR == 2025:
        pi_prefix = "pi"
        number_of_cams = 17

    cutoff = datetime(YEAR, 7, 13)

    # List all camera folders
    cam_folders = [f'{pi_prefix}{i}' for i in range(1, number_of_cams+1)]

    for cam in cam_folders:
        cam_path = os.path.join(input_folder, cam)
        output_cam_ir_path = os.path.join(output_folder, f'{cam}_nir')
        output_cam_rgb_path = os.path.join(output_folder, f'{cam}_rgb')

        os.makedirs(output_cam_ir_path, exist_ok=True)
        os.makedirs(output_cam_rgb_path, exist_ok=True)

        # List all image files in the current camera folder
        for filename in os.listdir(cam_path):
            if filename.endswith('.png'):
                image_path = os.path.join(cam_path, filename)

                date_part = filename.split("_")[1]  # "2-7-2025"
                day, month, year = map(int, date_part.split("-"))
                file_date = datetime(year, month, day)


                if file_date > cutoff:
                    print(f"After July 20: {filename}")
                    os.remove(image_path)
                    continue

                img = cv2.imread(image_path)  # BGR format

                img_width = img.shape[1]
                half_img_width = img_width // 2

                if cam == "pi10":
                    # Left half (RGB)
                    rgb_half = img[:, :half_img_width]
                    # Right half (NoIR)
                    noir_half = img[:, half_img_width:]
                else:
                    # Left half (NoIR)
                    noir_half = img[:, :half_img_width]
                    # Right half (RGB)
                    rgb_half = img[:, half_img_width:]

                noir_mean = noir_half.mean(axis=(0, 1))
                rgb_mean = rgb_half.mean(axis=(0, 1))

                if ((noir_mean > 239.9).all() or (noir_mean > 252).sum() >=2 or (rgb_mean > 239.9).all()
                        or (rgb_mean > 252).sum() >=2) :
                    print(f"============== Saturated: {filename} =================================")
                    os.remove(image_path)
                    continue

                # Save the left half
                cv2.imwrite(os.path.join(output_cam_ir_path, f'{filename}'), noir_half)
                cv2.imwrite(os.path.join(output_cam_rgb_path, f'{filename}'), rgb_half)

                print(f"Processed {filename}")

if __name__ == "__main__":

    data_folder = RAW_IMG_DIR
    input_folder, output_folder = data_folder, data_folder

    # Run the desired function
    split_images(input_folder, output_folder)
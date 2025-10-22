import os
from PIL import Image

from paths import RAW_IMG_DIR


def process_rgb_images(input_folder: str, output_folder_rgb: str) -> None:
    """
    Processes RGB images from camera folders, extracting and saving the right half of each image.

    Args:
        input_folder (str): The path to the folder containing camera subfolders with RGB images.
        output_folder_rgb (str): The path to the folder where the right halves of the RGB images will be saved.

    Returns:
        None
    """
    # List all camera folders
    cam_folders = [f'cam{i}' for i in range(1, 9)]

    for cam in cam_folders:
        cam_path = os.path.join(input_folder, cam)
        output_cam_rgb_path = os.path.join(output_folder_rgb, f'{cam}_rgb')

        os.makedirs(output_cam_rgb_path, exist_ok=True)
        # List all image files in the current camera folder
        for filename in os.listdir(cam_path):
            if filename.endswith('.png'):
                image_path = os.path.join(cam_path, filename)
                
                # Open the image
                img = Image.open(image_path)
                
                # Check the image size to ensure it is 1280 x 928
                if img.size != (1280, 928):
                    print(f"Skipping {image_path}: unexpected image size {img.size}")
                    continue
                
                # Split the image to get the right half
                right_half = img.crop((640, 0, 1280, 928))
                
                # Save the right half
                right_half.save(os.path.join(output_cam_rgb_path, f'{os.path.splitext(filename)[0]}{os.path.splitext(filename)[1]}'))
                
                print(f"Processed right side of {filename}")

def process_nir_images(input_folder: str, output_folder_ir: str) -> None:
    """
    Processes NIR images from camera folders, extracting and saving the left half of each image.

    Args:
        input_folder (str): The path to the folder containing camera subfolders with NIR images.
        output_folder_ir (str): The path to the folder where the left halves of the NIR images will be saved.

    Returns:
        None
    """
    # List all camera folders
    cam_folders = [f'cam{i}' for i in range(1, 9)]

    for cam in cam_folders:
        cam_path = os.path.join(input_folder, cam)
        output_cam_ir_path = os.path.join(output_folder_ir, f'{cam}_nir')
        
        os.makedirs(output_cam_ir_path, exist_ok=True)
        
        # List all image files in the current camera folder
        for filename in os.listdir(cam_path):
            if filename.endswith('.png'):
                image_path = os.path.join(cam_path, filename)
                
                # Open the image
                img = Image.open(image_path)
                
                # Check the image size to ensure it is 1280 x 928
                if img.size != (1280, 928):
                    print(f"Skipping {image_path}: unexpected image size {img.size}")
                    continue
                
                # Split the image to get the left half
                left_half = img.crop((0, 0, 640, 928))
                
                # Save the left half
                left_half.save(os.path.join(output_cam_ir_path, f'{os.path.splitext(filename)[0]}{os.path.splitext(filename)[1]}'))
                
                print(f"Processed left side of {filename}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_folder = RAW_IMG_DIR

    input_folder, output_folder_rgb, output_folder_ir = data_folder, data_folder, data_folder

    # Run the desired function
    process_rgb_images(input_folder, output_folder_rgb)
    process_nir_images(input_folder, output_folder_ir)
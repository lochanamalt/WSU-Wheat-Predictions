"""
@author: Lochana Marasinghe
@date: 7/14/2025
@description: 
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_IMG_DIR = "C:\\Users\\lochana.marasingha\\Documents\\Wheat Project\\2024_field_season_wheat\\for_img_processing\\2024_images"
CORRECTED_RF_IMG_DIR = "C:\\Users\\lochana.marasingha\\Documents\\Wheat Project\\2024_field_season_wheat\\for_img_processing\\corrected_images_reflectance_value_new"
CORRECTED_DN_IMG_DIR = "C:\\Users\\lochana.marasingha\\Documents\\Wheat Project\\2024_field_season_wheat\\for_img_processing\\corrected_images_digital_number_new"

ALIGNED_RGB_DIR = "C:\\Users\\lochana.marasingha\\Documents\\Wheat Project\\2024_field_season_wheat\\for_img_processing\\aligned_rgb"

PANEL_DETECT_CSV_OUTPUT = os.path.join(PROJECT_ROOT, 'data\\2024_outputs\\panel_detection_output\\csv_outputs_new')
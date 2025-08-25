"""
@author: Lochana Marasinghe
@date: 5/5/2025
@description: 
"""
import csv
import os

import cv2
import numpy as np
from typing import List, Tuple

def red(pi_red, pi_blue):
    return (0.953 * pi_red) - nir(pi_blue)

def green(pi_green, pi_blue):
    return (0.846 * pi_green) - nir(pi_blue)

def blue(pi_blue):
    return (0.884 * pi_blue) - nir(pi_blue)

def nir(pi_blue):
    return 0.832 * pi_blue

# NDVI range must be -1 to 1
def ndvi(pi_blue, pi_red) :
    return (nir(pi_blue) - red(pi_red=pi_red, pi_blue=pi_blue)) / (nir(pi_blue) + red(pi_red=pi_red, pi_blue=pi_blue))

def gndvi(pi_blue, pi_green) :
    return (nir(pi_blue) - green(pi_green=pi_green, pi_blue=pi_blue)) / (nir(pi_blue) + green(pi_green=pi_green, pi_blue=pi_blue))

def savi(pi_blue, pi_red) :
    return (1.5 * (nir(pi_blue)- red(pi_red=pi_red, pi_blue=pi_blue))) / (nir(pi_blue) + red(pi_red=pi_red, pi_blue=pi_blue) + 0.5)

def sr(pi_blue, pi_red) :
    return nir(pi_blue=pi_blue)/ red(pi_red=pi_red, pi_blue=pi_blue)

def evi(b,r) :
    numerator = (2.5 * (nir(pi_blue=b) - red(pi_red=r, pi_blue=b)))
    denominator = (nir(pi_blue=b) + (6 * red(pi_red=r, pi_blue=b)) - (7.5 * blue(pi_blue=b)) + 1)
    return numerator / denominator

def rdvi(b,r) :
    return (nir(pi_blue=b) - red(pi_blue=b,pi_red=r)) / (np.sqrt(nir(pi_blue=b) + red(pi_blue=b,pi_red=r)))

def CIgreen( g, b) :
    return (nir(pi_blue=b)/green(pi_green=g, pi_blue=b))-1

def calculate_vi(image_directory: str, output_directory: str) -> None:

    input_folders = [os.path.join(image_directory, f"cam{i}_nir") for i in range(1, 9)]
    output_folders = [os.path.join(output_directory, f"cam{i}") for i in range(1, 9)]

    print(input_folders)
    print("Calculating VI...")
    for index,cam_folder in enumerate(input_folders):
        # vi_list: List[Tuple[str, float, float, float, float, float, float, float]] = []
        cam_output_ndvi_colored = os.path.join(output_folders[index], f"cam{index+1}_ndvi_colored")
        cam_output_gndvi_colored = os.path.join(output_folders[index], f"cam{index+1}_gndvi_colored")
        cam_output_savi_colored = os.path.join(output_folders[index], f"cam{index+1}_savi_colored")
        cam_output_sr_colored = os.path.join(output_folders[index], f"cam{index+1}_sr_colored")
        cam_output_evi_colored = os.path.join(output_folders[index], f"cam{index+1}_evi_colored")
        cam_output_rdvi_colored = os.path.join(output_folders[index], f"cam{index+1}_rdvi_colored")
        cam_output_cigreen_colored = os.path.join(output_folders[index], f"cam{index+1}_cigreen_colored")

        cam_output_ndvi = os.path.join(output_folders[index], f"cam{index+1}_ndvi")
        cam_output_gndvi = os.path.join(output_folders[index], f"cam{index+1}_gndvi")
        cam_output_savi = os.path.join(output_folders[index], f"cam{index+1}_savi")
        cam_output_sr = os.path.join(output_folders[index], f"cam{index+1}_sr")
        cam_output_evi = os.path.join(output_folders[index], f"cam{index+1}_evi")
        cam_output_rdvi = os.path.join(output_folders[index], f"cam{index+1}_rdvi")
        cam_output_cigreen = os.path.join(output_folders[index], f"cam{index+1}_cigreen")

        if not os.path.exists(cam_output_ndvi):
            os.makedirs(cam_output_ndvi)

        if not os.path.exists(cam_output_gndvi):
            os.makedirs(cam_output_gndvi)

        if not os.path.exists(cam_output_savi):
            os.makedirs(cam_output_savi)

        if not os.path.exists(cam_output_sr):
            os.makedirs(cam_output_sr)

        if not os.path.exists(cam_output_evi):
            os.makedirs(cam_output_evi)

        if not os.path.exists(cam_output_rdvi):
            os.makedirs(cam_output_rdvi)

        if not os.path.exists(cam_output_cigreen):
            os.makedirs(cam_output_cigreen)

        if not os.path.exists(cam_output_ndvi_colored):
            os.makedirs(cam_output_ndvi_colored)

        if not os.path.exists(cam_output_gndvi_colored):
            os.makedirs(cam_output_gndvi_colored)

        if not os.path.exists(cam_output_savi_colored):
            os.makedirs(cam_output_savi_colored)

        if not os.path.exists(cam_output_sr_colored):
            os.makedirs(cam_output_sr_colored)

        if not os.path.exists(cam_output_evi_colored):
            os.makedirs(cam_output_evi_colored)

        if not os.path.exists(cam_output_rdvi_colored):
            os.makedirs(cam_output_rdvi_colored)

        if not os.path.exists(cam_output_cigreen_colored):
            os.makedirs(cam_output_cigreen_colored)

        for file in os.listdir(cam_folder):
            if file.endswith('.tif'):
                image_path = os.path.join(cam_folder, file)
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                (img_B, img_G, img_R) = cv2.split(img)

                img_B = img_B.astype('float32')
                img_B[img_B == 0] = 1e-5

                img_G = img_G.astype('float32')
                img_G[img_G == 0] = 1e-5

                img_R = img_R.astype('float32')
                img_R[img_R == 0] = 1e-5

                calc_ndvi = ndvi(img_B, img_R)
                calc_gndvi = gndvi(img_B, img_G)
                calc_evi = evi(img_B, img_R)
                calc_savi = savi(img_B, img_R)
                calc_sr = sr(img_B, img_R)
                calc_rdvi = rdvi(img_B, img_R)
                calc_CIgreen = CIgreen(img_G, img_B)

                # Clip values for visualization and normalize to 0-255
                ndvi_normalized = cv2.normalize(calc_ndvi, None, 0, 255, cv2.NORM_MINMAX)
                ndvi_normalized = ndvi_normalized.astype(np.uint8)
                ndvi_colored = cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_VIRIDIS)

                gndvi_normalized = cv2.normalize(calc_gndvi, None, 0, 255, cv2.NORM_MINMAX)
                gndvi_normalized = gndvi_normalized.astype(np.uint8)
                gndvi_colored = cv2.applyColorMap(gndvi_normalized, cv2.COLORMAP_VIRIDIS)

                savi_normalized = cv2.normalize(calc_sr, None, 0, 255, cv2.NORM_MINMAX)
                savi_normalized = savi_normalized.astype(np.uint8)
                savi_colored = cv2.applyColorMap(savi_normalized, cv2.COLORMAP_VIRIDIS)

                sr_normalized = cv2.normalize(calc_sr, None, 0, 255, cv2.NORM_MINMAX)
                sr_normalized = sr_normalized.astype(np.uint8)
                sr_colored = cv2.applyColorMap(sr_normalized, cv2.COLORMAP_VIRIDIS)

                evi_normalized = cv2.normalize(calc_evi, None, 0, 255, cv2.NORM_MINMAX)
                evi_normalized = evi_normalized.astype(np.uint8)
                evi_colored = cv2.applyColorMap(evi_normalized, cv2.COLORMAP_VIRIDIS)

                rdvi_normalized = cv2.normalize(calc_rdvi, None, 0, 255, cv2.NORM_MINMAX)
                rdvi_normalized = rdvi_normalized.astype(np.uint8)
                rdvi_colored = cv2.applyColorMap(rdvi_normalized, cv2.COLORMAP_VIRIDIS)

                cigreen_normalized = cv2.normalize(calc_CIgreen, None, 0, 255, cv2.NORM_MINMAX)
                cigreen_normalized = cigreen_normalized.astype(np.uint8)
                cigreen_colored = cv2.applyColorMap(cigreen_normalized, cv2.COLORMAP_VIRIDIS)

                cv2.imwrite(os.path.join(cam_output_ndvi_colored, file), ndvi_colored)
                cv2.imwrite(os.path.join(cam_output_gndvi_colored, file), gndvi_colored)
                cv2.imwrite(os.path.join(cam_output_savi_colored, file), savi_colored)
                cv2.imwrite(os.path.join(cam_output_sr_colored, file), sr_colored)
                cv2.imwrite(os.path.join(cam_output_evi_colored, file), evi_colored)
                cv2.imwrite(os.path.join(cam_output_rdvi_colored, file), rdvi_colored)
                cv2.imwrite(os.path.join(cam_output_cigreen_colored, file), cigreen_colored)

                cv2.imwrite(os.path.join(cam_output_ndvi, file),calc_ndvi)
                cv2.imwrite(os.path.join(cam_output_gndvi, file), calc_gndvi)
                cv2.imwrite(os.path.join(cam_output_savi, file), calc_savi)
                cv2.imwrite(os.path.join(cam_output_sr, file), calc_sr)
                cv2.imwrite(os.path.join(cam_output_evi, file), calc_evi)
                cv2.imwrite(os.path.join(cam_output_rdvi, file), calc_rdvi)
                cv2.imwrite(os.path.join(cam_output_cigreen, file), calc_CIgreen)



        # with open(cam_output_vi, mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['Filename', 'NDVI', 'GNDVI', 'SAVI', 'SR', 'EVI', 'RDVI', 'CIgreen'])
        #     writer.writerows(vi_list)
        # print(f"Saved VIs to {cam_output_vi}")



if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(project_root)

    corrected_img_directory = os.path.join(project_root, '..\..\data\\2024_outputs\corrected_images_reflectance_value')
    output_vi_directory = os.path.join(project_root, '..\..\data\\2024_outputs\\vegetation_indices')

    calculate_vi(corrected_img_directory, output_vi_directory)
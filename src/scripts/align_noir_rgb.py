"""
@author: Lochana Marasinghe
@date: 9/2/2025
@description: 
"""
import logging
import os
from enum import Enum

import cv2
import numpy as np

from paths import YEAR
from src.helper.directory_helper import get_noir_dir, Mode, get_rgb_dir

class AlignMethod(Enum):
    ECC = 1
    ORB_RANSAC = 2 # ORB = Oriented FAST and Rotated BRIEF


# | Method               | ORB_RANSAC                               | ECC-based alignment                                 |
# | ---------------------| ---------------------------------------- | --------------------------------------------------- |
# | Core algorithm       | ORB feature matching + RANSAC homography | ECC (Enhanced Correlation Coefficient) optimization |
# | Type                 | Feature-based                            | Intensity-based (direct)                            |
# | Transformation model | Homography (projective, 8 DOF)           | Affine or homography (depending on `warp_mode`)     |
# | Key function         | `cv2.findHomography()`                   | `cv2.findTransformECC()`                            |
#


def to_gray_float(img):
    if img is None:
        raise ValueError("Image is None")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return np.array(gray, dtype=np.float32) / 255.0


def edge_representations(im01):
    # Use edges to reduce RGB NoIR photometric mismatch
    im_blur = cv2.GaussianBlur(im01, (5,5), 0)
    # Sobel magnitude
    gx = cv2.Sobel(im_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(im_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # Normalize
    mmin, mmax = np.percentile(mag, [1, 99])
    mag = np.clip((mag - mmin)/(mmax - mmin + 1e-12), 0, 1)
    return mag

def orb_ransac_alignment(source_img_grey, output_img_grey, img_to_align):
    # Edge/gradient representation to reduce modality gap
    source_img = edge_representations(source_img_grey)
    output_img = edge_representations(output_img_grey)

    # Detect key-points and descriptors
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, fastThreshold=5)
    k1 = orb.detectAndCompute((source_img * 255).astype(np.uint8), None)
    k2 = orb.detectAndCompute((output_img * 255).astype(np.uint8), None)

    if k1 is None or k2 is None or k1[1] is None or k2[1] is None:
        raise Exception("ORB alignment: No key points detected")
    kp1, des1 = k1
    kp2, des2 = k2
    if des1 is None or des2 is None:
        raise Exception("ORB alignment: No descriptors detected")

    # Match descriptors using Brute Force Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # Gets the two best matches for each descriptor.
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]
    if len(good) < 10:
        raise Exception("ORB alignment: Not enough good matches found")

    # Extract matched keypoint coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # Estimate homography (using RANSAC), threshold = 3 controls how tolerant RANSAC is to outliers
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    if H is None:
        raise Exception("ORB alignment: Could not find homography")

    height, width = source_img.shape
    aligned_img = cv2.warpPerspective(img_to_align, H, ( width, height), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)

    # Compute reprojection error
    src_pts_hom = cv2.convertPointsToHomogeneous(src_pts).reshape(-1, 3).T
    proj_pts_hom = H @ src_pts_hom
    proj_pts_hom /= proj_pts_hom[2, :]

    proj_pts = proj_pts_hom[:2, :].T

    dst_pts_2d = dst_pts.reshape(-1, 2)
    errors = np.linalg.norm(dst_pts_2d - proj_pts, axis=1)

    mean_error = np.mean(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    return aligned_img, H, {'mean_error': mean_error, 'median_error': median_error, 'max_error': max_error, 'min_error': min_error}


def ecc_alignment(noir_grey, rgb_grey, rgb_img):
    # Define motion model
    warp_mode = cv2.MOTION_AFFINE  # or MOTION_AFFINE, MOTION_HOMOGRAPHY
    # Initialize warp matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 6000, 1e-7)
    (cc, warp_matrix) = cv2.findTransformECC(noir_grey, rgb_grey, warp_matrix, warp_mode, criteria)
    # Apply transformation
    print(rgb_img.shape)
    height, width, _ = rgb_img.shape
    aligned_img = cv2.warpAffine(rgb_img, warp_matrix, (width, height),
                             flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)

    return aligned_img, warp_matrix

def align_rgb_to_noir(rgb_img, noir_img, align_method: AlignMethod):
    """
    Align RGB (color) onto NIR.
    Inputs: rgb_bgr (H,W,3), nir_img (H,W,3). Returns aligned_color, warp, mode_used.
    """
    # Convert to grayscale first for alignment
    noir_grey = to_gray_float(noir_img)
    rgb_grey = to_gray_float(rgb_img)

    reproj_errors = None

    if align_method == AlignMethod.ECC:
        aligned_im, warp = ecc_alignment(noir_grey, rgb_grey, rgb_img)
    elif align_method == AlignMethod.ORB_RANSAC:
        aligned_im, warp, reproj_errors = orb_ransac_alignment(noir_grey, rgb_grey, rgb_img)
    else:
        raise ValueError("Invalid align method")

    return aligned_im, warp, reproj_errors


if __name__ == "__main__":

    pi_prefix = "cam"
    number_of_cams = 8
    if YEAR == 2025:
        pi_prefix = "pi"
        number_of_cams = 17

    # Collect the failing images, try the failed images again using the other alignment method
    orb_failed_images = []

    all_mean_errors = []
    all_median_errors = []
    all_max_errors = []
    all_min_errors = []

    # Alternate the alignment method here
    alignment_method = AlignMethod.ORB_RANSAC

    input_img_directory = get_noir_dir(Mode.RAW_DATA)
    output_directory = get_rgb_dir(Mode.RAW_DATA)

    for cam_no in range(1, number_of_cams+1):
        cam_noir_input_dir = os.path.join(input_img_directory, f'{pi_prefix}{cam_no}_nir')
        cam_rgb_input_dir = os.path.join(input_img_directory, f'{pi_prefix}{cam_no}_rgb')
        cam_rgb_output_dir = os.path.join(output_directory, f'{pi_prefix}{cam_no}_rgb')

        if not os.path.exists(cam_rgb_output_dir):
            os.makedirs(cam_rgb_output_dir)

        for filename in os.listdir(cam_noir_input_dir):
            if filename.endswith('.png'):
                print(f"Processing: {filename}")

                input_noir_file_path = os.path.join(cam_noir_input_dir, f"{filename}")
                input_rgb_file_path = os.path.join(cam_rgb_input_dir, filename)

                output_file_path = os.path.join(cam_rgb_output_dir, f"aligned_{filename}")

                img_noir_color = cv2.imread(input_noir_file_path, cv2.IMREAD_UNCHANGED)
                img_rgb_color = cv2.imread(input_rgb_file_path, cv2.IMREAD_UNCHANGED)

                try:
                    aligned, warp, reproj_errors  = align_rgb_to_noir(img_rgb_color, img_noir_color, alignment_method)

                    if reproj_errors:
                        print(reproj_errors)
                        median_error = float(reproj_errors['median_error'])

                        if median_error > 130:
                            print("Median error is too high")
                            orb_failed_images.append(f"{cam_no}_{filename}")
                            continue
                        else:
                            all_mean_errors.append(reproj_errors['mean_error'])
                            all_median_errors.append(reproj_errors['median_error'])
                            all_max_errors.append(reproj_errors['max_error'])
                            all_min_errors.append(reproj_errors['min_error'])

                    cv2.imwrite(output_file_path, aligned)

                except Exception as e:
                    logging.error(f"Failed to apply warp for image: {filename}")
                    logging.error(e)
                    orb_failed_images.append(f"{cam_no}_{filename}")

    print("Mean of mean errors:", np.mean(all_mean_errors))
    print("Median of mean errors:", np.median(all_mean_errors))
    print("Std of mean errors:", np.std(all_mean_errors))
    print("Max of max errors:", np.max(all_max_errors))
    print("Min of min errors:", np.max(all_min_errors))

    # 2024 data
    # Mean of mean errors: 20.91938822698406
    # Median of mean errors: 14.022722318184657
    # Std of mean errors: 25.943595154696826
    # Max of max errors: 2883.5286065514365
    # Min of min errors: 0.4785633181425849

    # 2025 data
    # Mean of mean errors: 54.58779457544186
    # Median of mean errors: 43.65315825640729
    # Std of mean errors: 53.33973360073811
    # Max of max errors: 9387.932117410168
    # Min of min errors: 0.7471738573418135

    # Try the failed images with ECC method (A better transition is implemented in the Azure deployed version.)

    alignment_method = AlignMethod.ECC
    ecc_failed_images = []

    for failed_filename in orb_failed_images:
        cam_no = int(failed_filename.split('_')[0])
        filename = failed_filename.split("_", 1)[1]

        cam_noir_input_dir = os.path.join(input_img_directory, f'{pi_prefix}{cam_no}_nir')
        cam_rgb_input_dir = os.path.join(input_img_directory, f'{pi_prefix}{cam_no}_rgb')
        cam_rgb_output_dir = os.path.join(output_directory, f'{pi_prefix}{cam_no}_rgb')

        if not os.path.exists(cam_rgb_output_dir):
            os.makedirs(cam_rgb_output_dir)

        print(f"Processing: {filename}")

        input_noir_file_path = os.path.join(cam_noir_input_dir, f"{filename}")
        input_rgb_file_path = os.path.join(cam_rgb_input_dir, filename)

        output_file_path = os.path.join(cam_rgb_output_dir, f"aligned_{filename}")

        img_noir_color = cv2.imread(input_noir_file_path, cv2.IMREAD_UNCHANGED)
        img_rgb_color = cv2.imread(input_rgb_file_path, cv2.IMREAD_UNCHANGED)

        try:
            aligned, warp, reproj_errors = align_rgb_to_noir(img_rgb_color, img_noir_color, alignment_method)
            cv2.imwrite(output_file_path, aligned)

        except Exception as e:
            logging.error(f"Failed to apply warp for image: {filename}")
            logging.error(e)
            ecc_failed_images.append(f"{cam_no}_{filename}")

    print(orb_failed_images)
    print(ecc_failed_images)



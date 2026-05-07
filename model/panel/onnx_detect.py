import csv
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from paths import RAW_IMG_DIR, PANEL_DETECT_IMG_OUTPUT, YEAR, PANEL_DETECT_CSV_OUTPUT

# ONNX
MODEL_PATH = "yolo12s_custom_panel_detection_combined_int8.onnx"
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

print("Output shape:", session.get_outputs()[0].shape)

def letterbox(img, size=640, color=(114,114,114)):
    # scale and padded to be 640 x 640 without stretching
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, left, top

def preprocess(img, size=640):
    img_lb, scale, dx, dy = letterbox(img, size)
    img_lb = img_lb[:, :, ::-1]  # converts BGR to RGB
    img_lb = img_lb.astype(np.float32)
    img_lb /= 255.0 # normalize

    # Converts Height, Width, Channels (OpenCV) to Channels, Height, Width (YOLO)
    img_lb = np.transpose(img_lb, (2, 0, 1))
    img_lb = np.expand_dims(img_lb, 0)
    return img_lb, scale, dx, dy

def get_boxes(predictions, scale, dx, dy, img_w, img_h, conf_threshold=0.3):

    boxes = []
    scores = []
    centers = []
    # predictions format is [cx, cy, w, h, confidence_score]
    for row in predictions:
        score = row[4]
        if score < conf_threshold:
            continue

        cx, cy, w, h = row[:4]

        # Remove padding and rescale
        cx = (cx - dx) / scale
        cy = (cy - dy) / scale
        w  = w / scale
        h  = h / scale

        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)

        # Clip to image
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        boxes.append([x1, y1, int(w), int(h)])
        scores.append(float(score))
        centers.append((x1, y1, x2, y2, float(score)))

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.4)

    filtered_centers = []
    if len(indices) > 0:
        for i in indices.flatten():
            filtered_centers.append(centers[i])
    return filtered_centers


def infer(img):

    processed_img, scale, dx, dy = preprocess(img)

    predictions = session.run(
        None,
        {input_name: processed_img}
    )[0]
    predictions = predictions[0].T    # (1,5,8400) to (8400,5)
    return predictions, scale, dx, dy

def draw_boxes(img, boxes, center_list, image_path):

    for x1,y1,x2,y2,score in boxes:

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        s = 12

        cv2.rectangle(
            img,
            (cx - s//2, cy - s//2),
            (cx + s//2, cy + s//2),
            (0,0,255),
            1
        )
        print("Score:", score)

        center_list.append((image_path, cx, cy, s, s))

    return img

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

        predictions, scale, dx, dy = infer(img)

        boxes = get_boxes(
            predictions,
            scale,
            dx,
            dy,
            img.shape[1],
            img.shape[0],
            conf_threshold=0.6,
        )

        if not boxes:
            print("No panel detected")
        print("No of boxes:", len(boxes))

        img = draw_boxes(img, boxes, centers_list, img_path.name)

        cv2.imwrite(str(output_path / img_path.name), img)
        print(f"Processed {img_path.name}")

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

pi_prefix = "cam"
if YEAR == 2025:
    pi_prefix = "pi"

for i in range(1, 2):

    centers_list: List[Tuple[str, int, int, int, int]] = []  # Reset centers_list for each new camera folder
    process_images(os.path.join(RAW_IMG_DIR, f'{pi_prefix}{i}'),
                   os.path.join(PANEL_DETECT_IMG_OUTPUT, f'cam{i}'), centers_list)

    combined_csv: str = os.path.join(PANEL_DETECT_CSV_OUTPUT, f'cam{i}.csv')
    #
    save_to_csv(centers_list, combined_csv)
    #
    nir_csv: str = os.path.join(PANEL_DETECT_CSV_OUTPUT, f'cam{i}_nir.csv')
    rgb_csv: str = os.path.join(PANEL_DETECT_CSV_OUTPUT, f'cam{i}_rgb.csv')
    split_csv(combined_csv, nir_csv, rgb_csv)

print("Processing complete.")

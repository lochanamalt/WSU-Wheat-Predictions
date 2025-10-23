"""
@author: Lochana Marasinghe
@date: 9/17/2025
@description: 
"""
from enum import Enum

from paths import RAW_IMG_DIR, CORRECTED_RF_IMG_DIR, ALIGNED_RGB_DIR

class Mode(Enum):
    RAW_DATA = 1
    CORRECTED_RF = 2


def get_noir_dir(mode: Mode) -> str:
    if mode == Mode.RAW_DATA:
        return  RAW_IMG_DIR
    else:
        return CORRECTED_RF_IMG_DIR

def get_rgb_dir(mode: Mode) -> str:
    if mode == Mode.RAW_DATA:
        return ALIGNED_RGB_DIR
    else:
        return CORRECTED_RF_IMG_DIR


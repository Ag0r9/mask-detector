import json

import numpy as np
import cv2


def getJSON(file_path_and_name):
    with open(file_path_and_name,'r') as f:
        return json.load(f)


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))
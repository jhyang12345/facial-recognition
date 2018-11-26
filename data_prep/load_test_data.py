import os, sys
from data_prep.prepare_dataset import preprocess_image, img_to_array, normalize_array
import numpy as np

def load_image_from_path(image_path):
    arr = preprocess_image(image_path)
    arr = np.asarray([arr], dtype=np.float32)
    return arr

def load_images_from_directory(directory_path):
    images = os.listdir(directory_path)
    arr = []
    for image in images:
        full_path = os.path.join(directory_path, image)
        arr.append(preprocess_image(full_path))
    arr = np.asarray(arr, dtype=np.float32)
    return arr

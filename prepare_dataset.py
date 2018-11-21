import os
from collections import defaultdict
import random
from PIL import Image
from sklearn.datasets import load_files
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np

# data ratio should be close to actual data distribution
# load files function reads directory and extracts filename and target class
# files should be under directory/target_class/files...
def load_dataset(positive_path, negative_path):
    positive_images = os.listdir(positive_path)
    negative_images = os.listdir(negative_path)
    positives = []
    negatives = []
    for image in positive_images:
        full_path = os.path.join(positive_path, image)
        positives.append(preprocess_image(full_path))
    for image in negative_images:
        full_path = os.path.join(negative_path, image)
        negatives.append(preprocess_image(full_path))

# must normalize image prior to adding to list
def preprocess_image(full_path):
    img = load_img(image_path, target_size=(128, 128))
    img = img_to_array(img)
    img = normalize_array(img)
    return img

# unless necessary, all images will be normalized to a -1 ~ 1 range
def normalize_array(np_array):
    half = 255 / 2
    np_array = np_array - half
    np_array = np_array / half
    return np_array

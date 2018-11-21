import os
from collections import defaultdict
import random
from PIL import Image
from sklearn.datasets import load_files
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np

def load_dataset(positive_path, negative_path):
    positive_data = load_files(positive_path)
    negative_data = load_files(negative_path)

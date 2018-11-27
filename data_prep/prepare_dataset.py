import os, sys
from collections import defaultdict
import random
from PIL import Image
from sklearn.datasets import load_files
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from data_prep.augment_dataset import augment_image

# data ratio should be close to actual data distribution
# load files function reads directory and extracts filename and target class
# files should be under directory/target_class/files...
# positive is comprised of manual_filter and filtered_positive
def load_dataset_files(positive_path="datasets/positive", negative_path="datasets/negative",
                            validation_ratio=0.2):
    positive_images = filter_images_in_path(positive_path)
    negative_images = filter_images_in_path(negative_path)
    positive_images = [os.path.join(positive_path, image) for image in positive_images]
    negative_images = [os.path.join(negative_path, image) for image in negative_images]
    positive_output = [[1] for _ in range(len(positive_images))]
    negative_output = [[0] for _ in range(len(negative_images))]

    input_dataset = positive_images + negative_images
    output_dataset = positive_output + negative_output

    dataset_size = len(positive_images + negative_images)
    validation_size = int(dataset_size * validation_ratio)
    validation_indexes = random.sample(list(range(dataset_size)), validation_size)
    training_indexes = [i for i in list(range(dataset_size)) if i not in validation_indexes]
    random.shuffle(validation_indexes)
    random.shuffle(training_indexes)

    training_input = [input_dataset[i] for i in training_indexes]
    training_output = [output_dataset[i] for i in training_indexes]
    validation_input = [input_dataset[i] for i in validation_indexes]
    validation_output = [output_dataset[i] for i in validation_indexes]

    return training_input, training_output, validation_input, validation_output

    # positives = []
    # negatives = []
    # for image in positive_images:
    #     full_path = os.path.join(positive_path, image)
    #     positives.append(preprocess_image(full_path))
    # for image in negative_images:
    #     full_path = os.path.join(negative_path, image)
    #     negatives.append(preprocess_image(full_path))
    # return positives, negatives

# must normalize image prior to adding to list
def preprocess_image(full_path):
    img = load_img(full_path, target_size=(128, 128))
    img = img_to_array(img)
    img = normalize_array(img)
    return img

# unless necessary, all images will be normalized to a -1 ~ 1 range
def normalize_array(np_array):
    half = 255 / 2
    np_array = np_array - half
    np_array = np_array / half
    return np_array

def filter_images_in_path(filter_path, min_size=80):
    images = os.listdir(filter_path)
    ret = []
    for image in images:
        full_path = os.path.join(filter_path, image)
        im = Image.open(full_path)
        if(im.size[0] > min_size):
            ret.append(image)
    return ret

def augmented_dataset(input_files, output):
    print(len(input_files))
    augmented_input = []
    augmented_output = []
    for i, image in enumerate(input_files):
        augmented_images = [normalize_array(arr) for arr in augment_image(image)]
        augmented_input = augmented_input + augmented_images
        augmented_output = augmented_output + [output[i] for _ in range(len(augmented_images))]
    return augmented_input, augmented_output

def load_dataset(validation_ratio=0.2):
    training_input_files, training_output, validation_input_files, validation_output = load_dataset_files()

    # need to apply preprocessing to arrays before using them as input
    # TODO: need to append to output to match the size of augmented input
    training_input = []
    validation_input = []
    training_input, training_output = augmented_dataset(training_input_files, training_output)
    validation_input, validation_output = augmented_dataset(validation_input_files, validation_output)

    training_input = np.asarray(training_input, dtype=np.float32)
    training_output = np.asarray(training_output, dtype=np.float32)
    validation_input = np.asarray(validation_input, dtype=np.float32)
    validation_output = np.asarray(validation_output, dtype=np.float32)
    return training_input, training_output, validation_input, validation_output

if __name__ == '__main__':
    training_input, training_output, validation_input, validation_output = load_dataset()
    print(training_input.shape, training_output.shape)
    print(validation_input.shape, validation_output.shape)

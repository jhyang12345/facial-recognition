import os, sys
from argparse import ArgumentParser
from PIL import Image
from data_prep.prepare_dataset import filter_images_in_path
from util import create_and_return_directory, copy_file
import random

def sample_files_from_directory(directory, count):
    images = filter_images_in_path(directory)
    sample_count = min(count, len(images))
    print("Sample Count: {}".format(sample_count))
    sampled_images = random.sample(images, sample_count)
    destination_root = os.path.join(directory, "..")
    sampled_directory = os.path.join(destination_root, "sampled")
    sampled_directory = create_and_return_directory(sampled_directory)
    for image in sampled_images:
        full_path = os.path.join(directory, image)
        copy_file(full_path, sampled_directory)


def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory")
    parser.add_argument("-c", "--count", dest="count")
    args = parser.parse_args()
    directory = ""
    count = 100
    if args.directory:
        directory = args.directory
    else:
        print("No directory given!")
        return
    if args.count:
        count = int(args.count)
    sample_files_from_directory(directory, count)


if __name__ == '__main__':
    main()

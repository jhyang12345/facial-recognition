import sys, os
from optparse import OptionParser
from cnn_models.hotdog import DeepDog
from cnn_models.cnn import CNN
from data_prep.load_test_data import load_image_from_path, load_images_from_directory

def test_model_individual(model, image_path):
    input_ = load_image_from_path(image_path)

def test_model_directory(model, image_directory):
    input_ = load_images_from_directory(image_directory)

def main():
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="path")
    parser.add_option("-d", "--directory", dest="directory")
    parser.add_option("-m", "--model", dest="model")
    options, args = parser.parse_args()
    path = ""
    directory = ""
    model_type = ""
    if options.path:
        print("Path accepted")
        path = options.path
    elif options.directory:
        print("Directory accepted")
        directory = options.directory
    else:
        print("No path or directory given!")
        return

    if options.model:
        model_type = options.model.lower()
    else:
        model_type = "default"

    model = None
    if model_type == "hotdog":
        model = DeepDog()
    elif model_type == "cnn":
        model = CNN()
    else:
        model = DeepDog()

    try:
        model.load_model()
    except Exception as e:
        print("Model should be pretrained!")



if __name__ == '__main__':
    main()

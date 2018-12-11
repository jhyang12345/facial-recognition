import sys, os
from argparse import ArgumentParser
import numpy as np
from data_prep.prepare_dataset import load_dataset_from_file
from cnn_models.hotdog import DeepDog
from cnn_models.cnn import CNN

from config_helper import retrieve_option_model

# There should be only one person match per image
def get_boolean_from_output(output_data, threshold=0.5):
    ret = []
    if not output_data.size:
        return ret
    max_value = np.max(output_data)
    for data in output_data:
        if data[0] > threshold and data[0] == max_value:
            ret.append(True)
        else:
            ret.append(False)
    return ret

# pass loaded model as parameter
# ImageFeeder is imported on the fly, to work with environments without face_recognition
def test_model_individual(model, image_path):
    from data_prep.image_pipeline import ImageFeeder
    image_feeder = ImageFeeder(image_path)
    input_data = image_feeder.input_data
    output_data = model.model.predict(input_data)
    print(output_data)
    location_values = get_boolean_from_output(output_data)
    image_feeder.set_location_values(location_values)
    return location_values

def test_model_and_save(model, image_path):
    from data_prep.image_pipeline import ImageFeeder
    image_feeder = ImageFeeder(image_path)
    input_data = image_feeder.input_data
    output_data = model.model.predict(input_data)
    print(output_data)
    location_values = get_boolean_from_output(output_data)
    image_feeder.set_location_values(location_values)
    image_feeder.save_drawn_image(location_values)

def evaluate_model(model, input_data, output_data):
    print("Beginning evaluation")
    score = model.model.evaluate(input_data, output_data)
    print(score)

# def test_model_directory(model, image_directory):
#     input_ = load_images_from_directory(image_directory)

def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", dest="path")
    parser.add_argument("-d", "--directory", dest="directory")
    parser.add_argument("-m", "--model", dest="model")
    parser.add_argument("-v", "--validate", action="store_true", dest="validate")
    parser.add_argument("-s", "--save", action="store_true", dest="save")
    args = parser.parse_args()
    path = ""
    directory = ""
    model_type = ""
    validate = ""

    model = retrieve_option_model(args.model)
    try:
        model.load_model()
    except Exception as e:
        print("Model should be pretrained!")

    if args.validate:
        _, __, validation_input, validation_output = \
            load_dataset_from_file()
        evaluate_model(model, validation_input, validation_output)
        return

    if args.path:
        print("Path accepted")
        path = args.path
    elif args.directory:
        print("Directory accepted")
        directory = args.directory
    else:
        print("No path or directory given!")
        return

    if args.save:
        test_model_and_save(model, path)

    print(test_model_individual(model, path))

if __name__ == '__main__':
    main()

import sys, os
from optparse import OptionParser
import numpy as np
from cnn_models.hotdog import DeepDog
from cnn_models.cnn import CNN
from data_prep.load_test_data import load_image_from_path, load_images_from_directory
from data_prep.image_pipeline import ImageFeeder, ImageDisplayer
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
def test_model_individual(model, image_path):
    image_feeder = ImageFeeder(image_path)
    input_data = image_feeder.input_data
    output_data = model.model.predict(input_data)
    print(output_data)
    location_values = get_boolean_from_output(output_data)
    image_feeder.set_location_values(location_values)
    return location_values

# def test_model_directory(model, image_directory):
#     input_ = load_images_from_directory(image_directory)

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

    model = retrieve_option_model(options.model)
    try:
        model.load_model()
    except Exception as e:
        print("Model should be pretrained!")
    print(test_model_individual(model, path))

if __name__ == '__main__':
    main()

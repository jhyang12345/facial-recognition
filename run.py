import sys
from optparse import OptionParser
from data_prep.prepare_dataset import load_dataset
from cnn_models.hotdog import DeepDog
from cnn_models.cnn import CNN
from trainers.trainer import Trainer

def main():
    parser = OptionParser()
    parser.add_option("-m", "--model", dest="model")
    options, args = parser.parse_args()
    model = None
    if options.model:
        model_type = options.model.lower()
        if model_type == "hotdog":
            model = DeepDog()
        elif model_type == "cnn":
            model = CNN()
        else:
            model = DeepDog()
    else:
        print("No Model specified!")
    Trainer(model)

if __name__ == '__main__':
    main()

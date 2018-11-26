import sys
from optparse import OptionParser
from data_prep.prepare_dataset import load_dataset
from cnn_models.hotdog import DeepDog
from cnn_models.cnn import CNN
from trainers.trainer import Trainer
from config_helper import retrieve_option_model

def main():
    parser = OptionParser()
    parser.add_option("-m", "--model", dest="model")
    options, args = parser.parse_args()
    model = None
    if options.model:
        model = retrieve_option_model(options.model)
    else:
        print("No Model specified!")
        model = retrieve_option_model("")
    Trainer(model)

if __name__ == '__main__':
    main()

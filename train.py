import sys
from optparse import OptionParser
from argparse import ArgumentParser
from data_prep.prepare_dataset import load_dataset
from cnn_models.hotdog import DeepDog
from cnn_models.cnn import CNN
from cnn_models.cnn_pool import CNNPool
from cnn_models.cnn_dropout import CNNDropout
from trainers.trainer import Trainer
from config_helper import retrieve_option_model

def train_all():
    print("Training all!")
    deep_dog = DeepDog()
    Trainer(deep_dog)
    print("Trained DeepDog")
    cnn = CNN()
    Trainer(cnn)


def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", dest="model")
    parser.add_argument("-a", "--all", action="store_true", dest="all")
    args = parser.parse_args()
    print(args)
    if args.all:
        train_all()
    model = None
    if args.model:
        model = retrieve_option_model(options.model)
    else:
        print("No Model specified!")
        model = retrieve_option_model("")
    Trainer(model)

if __name__ == '__main__':
    main()

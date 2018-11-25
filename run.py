import sys
from data_prep.prepare_dataset import load_dataset
from cnn_models.hotdog import DeepDog
from trainers.trainer import Trainer

def main():
    Trainer(DeepDog())

if __name__ == '__main__':
    main()

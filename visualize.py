import sys, os
from argparse import ArgumentParser
from config_helper import retrieve_option_model
from quiver_engine import server

def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", dest="model")
    parser.add_argument("-q", "--quiver", action="store_true", dest="quiver")
    args = parser.parse_args()

    model_type = ""
    if args.model:
        model_type = args.model

    model = retrieve_option_model(model_type)
    model.load_model()

    if args.quiver:
        server.launch(
            model.model,
            input_folder="./examples/test_cut",
        )

if __name__ == '__main__':
    main()

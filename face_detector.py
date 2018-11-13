import os, sys
from PIL import Image
import numpy as np
from argparse import ArgumentParser

def main(argv):
    print(argv)
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", required=False)
    args = vars(parser.parse_args())
    if args["path"]:
        print("Path accepted")
        path = args["path"][-1]
    else:
        print("No path given!")

if __name__ == '__main__':
    main(sys.argv[1:])

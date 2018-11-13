import os, sys
from PIL import Image
import face_recognition
import numpy as np
from argparse import ArgumentParser

def find_face_locations(image_path):
    print(image_path)
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image,
            number_of_times_to_upsample=0, model="cnn")
    print(face_locations)

def main(argv):
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", required=False)
    args = vars(parser.parse_args())
    path = ""
    if args["path"]:
        print("Path accepted")
        path = args["path"].split()[-1]
    else:
        print("No path given!")
        return
    find_face_locations(path)

if __name__ == '__main__':
    main(sys.argv[1:])

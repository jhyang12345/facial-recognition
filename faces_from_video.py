import sys, os
import cv2
from PIL import Image
import numpy as np
from argparse import ArgumentParser

def get_faces(video_file_path, skip_frame=20,
            destination_path="outputs_from_video"):
    cap = cv2.VideoCapture(video_file_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if i % skip_frame == 0:
            pass
    cap.release()

def main(sys.argv):
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
    iterate_over_directory(path)

if __name__ == '__main__':
    main(sys.argv[1:])

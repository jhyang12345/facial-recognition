import sys, os
import cv2
from PIL import Image
import numpy as np
from optparse import OptionParser

def get_faces(video_file_path, skip_frame=20,
            destination_path="outputs_from_video"):
    print(video_file_path)
    cap = cv2.VideoCapture(video_file_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("Iterating frame: {}".format(i))
        if i % skip_frame == 0:
            pass
    cap.release()

def main(argv):
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="path")
    options, args = parser.parse_args()
    print(options, args)
    path = ""
    if options.path:
        print("Path accepted")
        path = options.path
        get_faces(path)
    else:
        print("No path given!")
        return

if __name__ == '__main__':
    main(sys.argv[1:])

# [故 김광석 22주기 헌정영상] 잊어야 한다는 마음으로 – 아이유-ZXmoJu81e6A.webm

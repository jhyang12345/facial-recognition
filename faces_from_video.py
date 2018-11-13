import sys, os
import cv2
from PIL import Image
import numpy as np
from optparse import OptionParser
from face_detector import find_face_locations, save_faces, get_new_file_name

def save_faces_from_frame(frame, face_locations, destination_path="outputs_from_video"):
    i = 0
    for (top, right, bottom, left) in face_locations:
        sub_face = frame[top:bottom, left:right]
        file_name = get_new_file_name() + "_" + str(i)
        cv2.imwrite(os.path.join(destination_path, file_name) + ".jpg", sub_face)

def get_faces(video_file_path, skip_frame=40,
            ):
    print(video_file_path)
    cap = cv2.VideoCapture(video_file_path)
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if i % skip_frame == 0:
            print("Iterating frame: {}".format(i))
            face_locations = find_face_locations(frame)
            save_faces_from_frame(frame, face_locations)
            pass
        i += 1
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
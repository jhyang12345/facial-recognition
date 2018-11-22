import os, sys, time
from PIL import Image
import cv2
import face_recognition
from face_aligner import get_eyes_angle
from mtcnn.mtcnn import MTCNN
import numpy as np
from argparse import ArgumentParser

def get_new_file_name():
    value = int(time.time() * 1000)
    return str(value)

class MTCNN_detector:
    def __init__(self, output_path, filename):
        self.detector = None
        self.output_path = output_path
        self.filename = filename.split(".")[0]
        self.minimum_threshold = 100
        self.init_detector()

    def init_detector(self):
        self.detector = MTCNN()

    def mtcnn_find_faces(self, img):
        return self.detector.detect_faces(img)

    def get_bounding_boxes(self, img):
        ret = []
        for match in self.mtcnn_find_faces(img):
            ret.append(match["box"])
        return ret

    # [x, y, width, height]
    def get_image_crop_bounds(self, img):
        boxes = self.get_bounding_boxes(img)
        ret = []
        for box in boxes:
            ret.append(self.get_centered_box(box))
        return ret

    def get_centered_box(self, box):
        x, y, width, height = box
        length = max(width, height)
        if(width == length):
            y = int(y - (length - height) / 2)
        if(height == length):
            x = int(x - (length - width) / 2)
        return [x, y, length, length]

    # image is already opencv frame
    def crop_images_with_box(self, img, frame):
        bounds = self.get_image_crop_bounds(img)
        i = 0
        for bound in bounds:
            x, y, width, height = bound
            if width <= self.minimum_threshold or height <= self.minimum_threshold:
                continue
            try:
                print("iterating through bound", bound)
                sub_image = img[y:y+height, x:x+width]
                cv2.imwrite(os.path.join(self.output_path, "{}_{}_{}.jpg".format(self.filename, frame, i)), sub_image)
            except Exception as e:
                print(e)
                continue
            i += 1

def save_faces(im, face_locations, output_path="resized_faces"):
    extension = im.filename.split('.')[-1]
    img = np.asarray(im)
    for (top, right, bottom, left) in face_locations:
        sub_image = img[top:bottom, left:right] # left upper right lower
        sub_image = Image.fromarray(np.uint8(sub_image))
        sub_image.save(
            os.path.join(output_path, get_new_file_name() + ".jpg"))
    print("Saved {} new images".format(len(face_locations)))

# face_recognition loaded image file passed to image
def find_face_locations(image, image_path=""):
    face_locations = face_recognition.face_locations(image,
            number_of_times_to_upsample=0, model="cnn")
    print("Processing: {}, Number of faces found: {}"
        .format(os.path.basename(image_path), len(face_locations)))
    return face_locations

def find_face_locations_with_path(image_path):
    image = face_recognition.load_image_file(image_path)
    return find_face_locations(image, image_path)

def find_face_landmarks_with_path(image_path):
    image = face_recognition.load_image_file(image_path)
    face_landmarks_list = face_recognition.face_landmarks(image)
    face_locations = find_face_locations(image, image_path)
    for landmark in face_landmarks_list:
        angle, eye_center = get_eyes_angle(landmark, image)
        continue

def iterate_over_directory(directory_path):
    files = os.listdir(directory_path)
    cur_dir = directory_path
    for image in files:
        abs_path = os.path.join(cur_dir, image)
        if "gif" in image:
            continue
        try:
            locations = find_face_locations_with_path(abs_path)
            if(locations):
                save_faces(Image.open(abs_path), locations)
        except Exception as e:
            print(e)
            print("Something went wrong with file:", abs_path)

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
    # iterate_over_directory(path)
    find_face_landmarks_with_path(path)

if __name__ == '__main__':
    main(sys.argv[1:])

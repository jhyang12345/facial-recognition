import os, sys, time
from PIL import Image
import cv2
import face_recognition
from face_aligner import FaceAligner
import numpy as np
from argparse import ArgumentParser

def get_new_file_name():
    value = int(time.time() * 1000)
    return str(value)

def save_faces(im, face_locations, output_path="manual_filter"):
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
            number_of_times_to_upsample=1, model="cnn")
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

def handle_image_faces(image_path):
    image = face_recognition.load_image_file(image_path)
    handle_image_array_faces(image)

def handle_image_array_faces(image, base_image_name="", output_directory="manual_filter"):
    face_locations = face_recognition.face_locations(image,
            number_of_times_to_upsample=1, model="cnn")
    face_landmarks_list = face_recognition.face_landmarks(image)
    if len(face_locations) != len(face_landmarks_list):
        print("landmarks and face_locations do not match! Found faces: {}".format(len(face_locations)))
        save_faces(Image.fromarray(image), face_locations)
        return
    aligner = FaceAligner(output_directory=output_directory)
    for i in range(len(face_landmarks_list)):
        face_location = face_locations[i]
        face_landmarks = face_landmarks_list[i]
        if base_image_name:
            aligner.save_rotated_face(face_location, face_landmarks, image, file_name="{}_{}.jpg".format(base_image_name, i))
        else:
            aligner.save_rotated_face(face_location, face_landmarks, image, file_name="{}_{}.jpg".format(get_new_file_name(), i))

def iterate_over_directory(directory_path):
    files = os.listdir(directory_path)
    cur_dir = directory_path
    for image in files:
        abs_path = os.path.join(cur_dir, image)
        if "gif" in image:
            continue
        try:
            handle_image_faces(abs_path)
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
    iterate_over_directory(path)
    # handle_image_faces(path)

if __name__ == '__main__':
    main(sys.argv[1:])

import os, sys, time
from PIL import Image
import face_recognition
import numpy as np
from argparse import ArgumentParser

def get_new_file_name():
    value = int(time.time() * 1000)
    return str(value)

def save_faces(im, face_locations, output_path="resized_faces"):
    extension = im.filename.split('.')[-1]
    img = np.asarray(im)
    for (top, right, bottom, left) in face_locations:
        sub_image = img[top:bottom, left:right] # left upper right lower
        sub_image = Image.fromarray(np.uint8(sub_image))
        sub_image.save(
            os.path.join(output_path, get_new_file_name() + ".png"))
    print("Saved {} new images".format(len(face_locations)))

def find_face_locations(image, image_path=""):
    face_locations = face_recognition.face_locations(image,
            number_of_times_to_upsample=0, model="cnn")
    print("Processing: {}, Number of faces found: {}"
        .format(os.path.basename(image_path), len(face_locations)))
    return face_locations

def find_face_locations_with_path(image_path):
    image = face_recognition.load_image_file(image_path)
    return find_face_locations(image, image_path)

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
    iterate_over_directory(path)

if __name__ == '__main__':
    main(sys.argv[1:])

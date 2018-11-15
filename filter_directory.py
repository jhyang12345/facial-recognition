import sys, os
from PIL import Image
from util import create_and_return_directory, copy_file
from vgg_face import VGG_face

def filter_directory(directory, positive_path="filtered_positive",
            negative_path="filtered_negative", reference_image="reference.jpg"):
    face = VGG_face()
    images = os.listdir(directory)
    for image in images:
        abs_path = os.path.join(directory, image)
        try:
            im = Image.open(abs_path)
            if(im.size[0] < 128 or im.size[1] < 128): continue
        except Exception as e:
            print("Skipping")
            continue
        cosine, is_match = face.verify_face("reference.jpg", abs_path)
        if is_match:
            copy_file(abs_path, positive_path)
            print("Positive match:", image)
        else:
            copy_file(abs_path, negative_path)
            print("Negative match:", image)
        if (face.change_anchor(cosine)):
            reference_image = abs_path
            print("Changing anchor", reference_image)

if __name__ == '__main__':
    directory = sys.argv[-1]
    filter_directory(directory)

import sys, os
from vgg_face import VGG_face

def filter_directory(directory, positive_path="filtered_positive", negative_path="filtered_negative"):
    face = VGG_face()
    images = os.listdir(directory)
    for image in images:
        abs_path = os.path.join(directory, image)
        
    print(images)

if __name__ == '__main__':
    directory = sys.argv[-1]
    filter_directory(directory)

import sys
import matplotlib.pyplot as plt
from face_detector import find_face_locations_with_path, find_face_locations
import numpy as np
from PIL import Image, ImageDraw
from data_prep.prepare_dataset import normalize_array

def resize_for_display(pil_image, max_width=600):
    wpercent = (max_width / float(pil_image.size[0]))
    hsize = int((float(pil_image.size[1]) * float(wpercent)))
    pil_image = pil_image.resize((max_width, hsize), Image.ANTIALIAS)
    return pil_image

def draw_rect(drawcontext, xy, outline=(0, 100, 255), width=4):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

def display_image_array(arr):
    plt.imshow(arr)
    plt.show()

def display_image_with_drawn_boundaries(pil_image, locations):
    temp_image = pil_image.copy()
    draw = ImageDraw.Draw(temp_image)
    for (top, right, bottom, left) in locations:
        draw_rect(draw, ((left, top), (right, bottom)))
    display_image_array(np.asarray(temp_image))

def display_cut_images(pil_image, locations):
    img = np.asarray(pil_image)
    i = 0
    for (top, right, bottom, left) in locations:
        sub_image = img[top:bottom, left:right] # left upper right lower
        display_image_array(sub_image)


class ImageFeeder:
    def __init__(self, full_path):
        # maintain an array of face locations
        self.locations = []
        self.full_path = full_path
        self.image_to_input()

    def image_to_input(self):
        full_path = self.full_path
        self.locations = find_face_locations_with_path(full_path)
        im = np.asarray(Image.open(full_path))
        if not self.locations:
            print("0 faces found!")
            return
        input_data = []
        for location in self.locations:
            top, right, bottom, left = location
            sub_image = im[top:bottom, left:right]
            input_data.append(normalize_array(sub_image))
        input_data = np.asarray(input_data, dtype=np.float32)
        return input_data

class ImageDisplayer:
    def __init__(self, full_path, resize=True):
        self.locations = []
        self.full_path = full_path
        image = self.image_resize()
        display_image_with_drawn_boundaries(image, self.locations)
        display_cut_images(image, self.locations)

    def image_resize(self):
        full_path = self.full_path
        image = Image.open(full_path)
        image = resize_for_display(image)
        # find locations after resizing
        self.locations = find_face_locations(np.asarray(image))
        if not self.locations:
            print("0 faces found!")
            return
        return image


if __name__ == '__main__':
    image_path = sys.argv[-1]
    ImageDisplayer(image_path)

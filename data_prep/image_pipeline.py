import sys
from matplotlib.pyplot import imshow
from face_detector import find_face_locations_with_path
import numpy as np
from PIL import Image, ImageDraw
from data_prep.prepare_dataset import normalize_array

def draw_rect(drawcontext, xy, outline=(0, 100, 255), width=4):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

def get_image_with_drawn_boundaries(full_path, locations):
    pil_image = Image.open(full_path)
    draw = ImageDraw.Draw(pil_image)
    for (top, right, bottom, left) in locations:
        draw_rect(draw, ((left, top), (right, bottom)))
    imshow(np.asarray(pil_image))

class ImageFeeder:
    def __init__(self, full_path):
        # maintain an array of face locations
        self.locations = []
        self.full_path = full_path
        self.image_to_input()
        get_image_with_drawn_boundaries(full_path, self.locations)

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

if __name__ == '__main__':
    image_path = sys.argv[-1]
    ImageFeeder(image_path)

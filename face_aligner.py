import os
import numpy as np
from PIL import Image, ImageDraw
import cv2

necessary_landmarks = ["left_eye", "right_eye", "nose_bridge"]

def is_necessary_landmarks_there(landmark):
    for point in necessary_landmarks:
        if point not in landmark.keys():
            return False
    return True

# pass array instead of image
def draw_circle(x, y, pil_image, radius=2):
    d = ImageDraw.Draw(pil_image)
    center_x = int(x)
    center_y = int(y)
    bounding_box = [center_x-radius, center_y-radius, center_x+radius, center_y+radius]
    d.ellipse(bounding_box, fill=(202, 163, 255, 255))

def draw_landmarks(left_center, right_center, nose_center, image_array):
    pil_image = Image.fromarray(image_array)
    draw_circle(left_center[0], left_center[1], pil_image)
    draw_circle(right_center[0], right_center[1], pil_image)
    draw_circle(nose_center[0], nose_center[1], pil_image)
    pil_image.save("output.jpg")

class FaceAligner:
    def __init__(self, desired_left_eye=(0.35, 0.35), desired_image_width=128,
            desired_image_height=128, output_directory="manual_filter"):
        self.desired_left_eye = desired_left_eye
        self.desired_right_eye = (1 - desired_left_eye[0], desired_left_eye[1])
        self.desired_image_width = desired_image_width
        self.desired_image_height = desired_image_height
        self.output_directory = output_directory

    def align(self, image, rect):
        pass

        # image is face_recognition opened image
    def get_eyes_angle(self, landmark, image_array):
        if not is_necessary_landmarks_there(landmark):
            return 0
        left_eye = landmark["left_eye"]
        right_eye = landmark["right_eye"]
        nose_bridge = landmark["nose_bridge"]
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        # nose_center is not reliable for center of face
        nose_center = np.mean(nose_bridge, axis=0)
        dY = right_center[1] - left_center[1]
        dX = right_center[0] - left_center[0]
        eye_center = np.mean([left_center, right_center], axis=0).astype("int")
        # draw_landmarks(left_center, right_center, eye_center, image_array)
        print("Found centers")
        angle = np.degrees(np.arctan2(dY, dX))
        return angle, left_center, right_center, eye_center

    def get_image_relative_scale(self, rectangle, left_eye, right_eye):
        top, right, bottom, left = rectangle
        image_width = right - left
        eye_gap = np.sqrt((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2)
        eye_gap_ratio_distance = (self.desired_right_eye[0] - self.desired_left_eye[0])
        scale = image_width * eye_gap_ratio_distance / eye_gap
        print("Got scale:", scale)
        return scale

    def save_rotated_face(self, rectangle, landmark, image_array, file_name="output.jpg"):
        top, right, bottom, left = rectangle
        angle, left_eye, right_eye, eye_center = self.get_eyes_angle(landmark, image_array)
        zoom_scale = self.get_image_relative_scale(rectangle, left_eye, right_eye)
        eye_center = (eye_center[0], eye_center[1])

        M = cv2.getRotationMatrix2D(eye_center, angle, zoom_scale)
        # update the translation component of the matrix
        image_width = right - left
        image_height = bottom - top
        tX = image_width * 0.5
        tY = image_height * self.desired_left_eye[1]
        M[0, 2] += (tX - eye_center[0])
        M[1, 2] += (tY - eye_center[1])

        # apply the affine transformation
        (w, h) = (image_width, image_height)
        output = cv2.warpAffine(image_array, M, (w, h),
            flags=cv2.INTER_CUBIC)
        Image.fromarray(np.uint8(output)).save(os.path.join(self.output_directory, file_name))

        # dummy_image = Image.fromarray(image_array).rotate(angle, center=eye_center)
        # dummy_array = np.asarray(dummy_image)
        # top, right, bottom, left = rectangle
        # sub_image = dummy_array[top:bottom, left:right]
        # sub_image = Image.fromarray(np.uint8(sub_image))
        #
        # sub_image.save("output.jpg")

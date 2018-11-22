import numpy as np
from PIL import Image, ImageDraw

necessary_landmarks = ["left_eye", "right_eye", "nose_bridge"]

# image is face_recognition opened image
def get_eyes_angle(landmark, image_array):
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
    eye_center = np.mean([left_center, right_center], axis=0)
    print(eye_center)
    draw_landmarks(left_center, right_center, eye_center, image_array)
    angle = np.degrees(np.arctan2(dY, dX))
    return angle, eye_center

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

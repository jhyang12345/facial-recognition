import numpy as np

def get_eyes_angle(landmark):
    if "left_eye" not in landmark.keys() or "right_eye" not in landmark.keys():
        return 0
    left_eye = landmark["left_eye"]
    right_eye = landmark["right_eye"]
    left_center = np.mean(left_eye, axis=0)
    right_center = np.mean(right_eye, axis=0)
    dY = right_center[1] - left_center[1]
    dX = right_center[0] - left_center[0]
    print(left_center, right_center)
    angle = np.degrees(np.arctan2(dY, dX))
    print(angle)

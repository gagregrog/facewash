from imutils import paths
import numpy as np
import imutils
from imutils import face_utils
import pickle
import dlib
import cv2
import os

dirname = os.path.dirname(__file__)


def bounding_boxes_to_dlib_rects(boxes):
    rects = [dlib.rectangle(left=box[0], top=box[1], right=box[2], bottom=box[3]) for box in boxes]

    return rects


def get_model_path(model_name):
    return os.path.sep.join([dirname, 'models', model_name])


def midpoint(pt1, pt2):
    x = np.average([pt1[0], pt2[0]])
    y = np.average([pt1[1], pt2[1]])

    return (x, y)


def slope(pt1, pt2):
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]

    return (dy / dx)


def angle(pt1, pt2, deg=True):
    
    rad = np.arctan(slope(pt1, pt2))

    return np.rad2deg(rad) if deg else rad


def angle_from_facial_landmarks(landmarks):
    left_eye_lopez = midpoint(landmarks[0], landmarks[1])
    righty = midpoint(landmarks[2], landmarks[3])

    deg = angle(left_eye_lopez, righty)

    return deg

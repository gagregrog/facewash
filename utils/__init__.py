import numpy as np
import dlib
import os


def bounding_box_to_dlib_rect(box):
    """Accept bounding box in the form (x0, y0, x1, y1) and return a dlib rectangle representation of the box. 
       Used for facial landmark detection."""
    rect = dlib.rectangle(left=box[0], top=box[1], right=box[2], bottom=box[3])

    return rect


def bounding_boxes_to_dlib_rects(boxes):
    """Accept bounding box in the form (x0, y0, x1, y1) and return a dlib rectangle representation of the box. 
       Used for facial landmark detection."""
    rects = [bounding_box_to_dlib_rect(box) for box in boxes]

    return rects


def get_model_path(dirname, model_name):
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
    a, b, c, d = None, None, None, None
    if len(landmarks) == 5:
        a, b, c, d = 0, 1, 2, 3
    else:
        a, b, c, d = 45, 42, 39, 36

    lefty = midpoint(landmarks[a], landmarks[b])
    righty = midpoint(landmarks[c], landmarks[d])

    deg = angle(lefty, righty)

    return deg


def box_to_ellipse(box):
        x0, y0, x1, y1 = box
        center_x = int(((x0 + x1) / 2))
        center_y = int(((y0 + y1) / 2))
        width = int(np.abs(x0 - x1) / 2)
        height = int(np.abs(y0 - y1) / 2)
        center = (center_x, center_y)
        axes = (width, height)

        return center, axes


def pad_box(h, w, box, padding=10):
    x0, y0, x1, y1 = box

    x0 = x0 - padding if x0 - padding > 0 else 0
    y0 = y0 - padding if y0 - padding > 0 else 0
    x1 = x1 + padding if x1 + padding < w else w
    y1 = y1 + padding if y1 + padding < h else h

    return (x0, y0, x1, y1)

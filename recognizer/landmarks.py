from imutils import face_utils
import numpy as np
import dlib
import cv2
import os
import utils as u

dirname = os.path.dirname(__file__)
landmark = 'shape_predictor_5_face_landmarks.dat'
landmarkPath = u.get_model_path(dirname, landmark)


class Landmarker:
    def __init__(self):
        self.predictor = dlib.shape_predictor(landmarkPath)

    def get_facial_landmarks(self, image, boxes):
        """boxes should be an array of (x0, y0, x1, y1)"""

        rects = u.bounding_boxes_to_dlib_rects(boxes)

        facial_landmarks = []

        for rect in rects:
            shapes = self.predictor(image, rect)
            shapes = face_utils.shape_to_np(shapes)
            facial_landmarks.append(shapes)

        return facial_landmarks

    def draw_5_point_landmark(self, image, landmarks, color=(0, 255, 0)):
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 3, color, -1)

    def draw_landmarks_and_boxes(self, image, boxes, colors=None, show_angle=False):
        facial_landmarks = self.get_facial_landmarks(image, boxes)

        if colors is None or len(colors) < len(boxes):
            colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        else:
            colors = colors[:len(boxes)]

        for (box, landmarks, color) in zip(boxes, facial_landmarks, colors):
            x0, y0, x1, y1 = box
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

            if show_angle:
                angle = u.angle_from_facial_landmarks(landmarks)
                cv2.putText(image, 'Angle: {:.2f}'.format(angle), (x0, y0 - 10 if y0 - 10 > 0 else y0 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            self.draw_5_point_landmark(image, landmarks, color)

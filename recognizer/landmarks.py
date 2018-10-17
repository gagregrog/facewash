import numpy as np
import imutils
from imutils import face_utils
import dlib
import cv2
import os
import recognizer.util as u

dirname = os.path.dirname(__file__)
landmark = 'shape_predictor_5_face_landmarks.dat'
landmarkPath = u.get_model_path(dirname, landmark)


class Landmarker:
    def __init__(self, min_conf=0.5, width=600):
        self.predictor = dlib.shape_predictor(landmarkPath)

    def _get_facial_landmarks(self, image, boxes):
        """boxes should be an array of (x0, y0, x1, y1)"""

        rects = u.bounding_boxes_to_dlib_rects(boxes)

        facial_landmarks = []

        for rect in rects:
            shapes = self.predictor(image, rect)
            shapes = face_utils.shape_to_np(shapes)
            facial_landmarks.append(shapes)

        return facial_landmarks

    def draw_5_point_landmarks(self, image, landmarks, color=(0, 255, 0)):
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 3, color, -1)

    @staticmethod
    def get_angles_from_landmarks(landmarks):
        angles = [-1 * u.angle_from_facial_landmarks(landmark) for landmark in landmarks]

        return angles

    def draw_landmarks_and_boxes(self, image, boxes, colors=None, show_angle=False):
        facial_landmarks = self._get_facial_landmarks(image, boxes)
        angles = None

        if colors is None:
            colors = np.random.uniform(0, 255, size=(len(facial_landmarks), 3))

        if show_angle:
            angles = Landmarker.get_angles_from_landmarks(facial_landmarks)

        if len(boxes) == 1:
            colors = [(0, 255, 0)]

        for (box, landmark, angle, color) in zip(boxes, landmarks, angles, colors):
            x0, y0, x1, y1 = box
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

            if show_angle:
                cv2.putText(image, 'Angle: {:.2f}'.format(angle), (x0, y0 - 10 if y0 - 10 > 0 else y0 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            for (x, y) in landmark:
                cv2.circle(image, (x, y), 3, color, -1)


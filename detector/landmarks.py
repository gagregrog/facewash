from imutils import face_utils
import numpy as np
import dlib
import cv2
import os
import utils as u

dirname = os.path.dirname(__file__)
landmark5 = 'shape_predictor_5_face_landmarks.dat'
landmark5Path = u.get_model_path(dirname, landmark5)
landmark68 = 'shape_predictor_68_face_landmarks.dat'
landmark68Path = u.get_model_path(dirname, landmark68)


class Landmarker:
    def __init__(self):
        self.predictor5 = dlib.shape_predictor(landmark5Path)
        self.predictor68 = dlib.shape_predictor(landmark68Path)

    def get_facial_landmarks(self, image, boxes, sixty_eight=False):
        """boxes should be an array of (x0, y0, x1, y1)"""

        rects = u.bounding_boxes_to_dlib_rects(boxes)

        facial_landmarks = []

        predictor = self.predictor5 if sixty_eight is False else self.predictor68
        
        # convert to rhb for dlib
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for rect in rects:
            shapes = predictor(rgb, rect)
            shapes = face_utils.shape_to_np(shapes)
            facial_landmarks.append(shapes)

        return facial_landmarks

    def draw_landmarks(self, image, landmarks, color=(0, 255, 0)):
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 1, color, -1)

    def draw_landmarks_and_boxes(self, image, boxes, colors=None, show_angle=False, sixty_eight=False):
        facial_landmarks = self.get_facial_landmarks(image, boxes, sixty_eight=sixty_eight)

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

            self.draw_landmarks(image, landmarks, color)

    def get_angles_from_boxes(self, image, boxes):
        landmarks = self.get_facial_landmarks(image, boxes)
        angles = [u.angle_from_facial_landmarks(landmark) for landmark in landmarks]

        return angles

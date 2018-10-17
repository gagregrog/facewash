import os
import cv2
import numpy as np
import utils as u
from recognizer.landmarks import Landmarker
from recognizer.transform import Transformer

dirname = os.path.dirname(__file__)

proto = 'deploy.prototxt'
caffe = 'res10_300x300_ssd_iter_140000.caffemodel'

caffePath = u.get_model_path(dirname, caffe)
protoPath = u.get_model_path(dirname, proto)


class Recognizer:
    def __init__(self, min_conf=0.5, with_landmarks=False, with_transformer=False, colors=None):
        self.min_conf = min_conf
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, caffePath)
        self.landmarker = None if with_landmarks is False else Landmarker()
        self.transformer = None if with_transformer is False else Transformer()
        self.colors = colors if colors is not None else np.random.uniform(0, 255, size=(255, 3))

    def _verify_landmarker(self):
        if self.landmarker is None:
            self.landmarker = Landmarker()

    def _verify_transformer(self):
        if self.transformer is None:
            self.transformer = Transformer()

    def _get_colors(self, colors, length):
        if colors is None:
            colors = self.colors

        if len(colors) < length:
            colors = np.random.uniform(0, 255, size=(length, 3))
        else:
            colors = colors[:length]

        return colors

    def detect_faces_raw(self, image):
        """Return the raw facial detections from an image."""

        imageBlob = cv2.dnn.blobFromImage(
          cv2.resize(image, (300, 300)), 1.0, (300, 300),
          (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        return detections

    def get_box_and_conf(self, image, detections, index):
        valid = False
        conf = detections[0, 0, index, 2]

        # ensure the detections meets the min conf
        if conf > self.min_conf:
            h, w = image.shape[:2]

            # compute the bounding box
            box = detections[0, 0, index, 3:7] * np.array([w, h, w, h])
            box = box.astype('int')
            valid = True

        return (box, conf) if valid else (None, None)

    def get_boxes_and_confs(self, image, detections):
        boxes_and_confs = []

        for i in range(0, detections.shape[2]):
            box, conf = self.get_box_and_conf(image, detections, i)

            if box is not None:
                boxes_and_confs.append((box, conf))

        return boxes_and_confs

    def get_boxes_and_confs_from_image(self, image):
        detections = self.detect_faces_raw(image)
        boxes_and_confs = self.get_boxes_and_confs(image, detections)

        return boxes_and_confs

    def get_boxes_from_image(self, image):
        boxes = [a[0] for a in self.get_boxes_and_confs_from_image(image)]

        return boxes

    def draw_boxes(self, image, colors=None, conf_label=False, thickness=2):
        boxes_and_confs = self.get_boxes_and_confs_from_image(image)
        colors = self._get_colors(colors, len(boxes_and_confs))

        for ((box, conf), color) in zip(boxes_and_confs, colors):
            x0, y0, x1, y1 = box

            cv2.rectangle(image, (x0 - 5, y0 - 5), (x1 + 5, y1 + 5), color, 2)

            if conf_label:
                y = (y0 - 10) if ((y0 - 10) > 0) else (y0 + 10)
                cv2.putText(image, 'Conf: {:.2f}'.format(conf), (x0, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), thickness)

    def draw_boxes_angles_and_landmarks(self, image, colors=None, show_angle=False):
        self._verify_landmarker()

        boxes = self.get_boxes_from_image(image)
        colors = self._get_colors(colors, len(boxes))
        self.landmarker.draw_landmarks_and_boxes(image, boxes, colors, show_angle)

    def remove_faces(self, image, background=None, padding=None):
        self._verify_transformer()

        boxes = self.get_boxes_from_image(image)
        self.transformer.remove_faces(image, boxes, background, padding)

    def blur_faces(self, image, kernal_size=50):
        self._verify_transformer()

        boxes = self.get_boxes_from_image(image)
        self.transformer.blur_faces(image, boxes, kernal_size)

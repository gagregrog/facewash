import numpy as np
import imutils
import cv2
import os
import recognizer.util as u

dirname = os.path.dirname(__file__)

proto = 'deploy.prototxt'
caffe = 'res10_300x300_ssd_iter_140000.caffemodel'

caffePath = u.get_model_path(dirname, caffe)
protoPath = u.get_model_path(dirname, proto)


class Recognizer:
    def __init__(self, min_conf=0.5):
        # for detecting faces in a frame
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, caffePath)
        self.min_conf = min_conf

    def detect_faces_raw(self, image):
        imageBlob = cv2.dnn.blobFromImage(
          cv2.resize(image, (300, 300)), 1.0, (300, 300),
          (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        return detections

    def get_rect_and_conf(self, image, detections, index):
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

    def get_rects_and_confs(self, image, detections):
        rects_and_confs = []

        for i in range(0, detections.shape[2]):
            box, conf = self.get_rect_and_conf(image, detections, i)

            if box is not None:
                rects_and_confs.append((box, conf))

        return rects_and_confs

    def get_rects_and_confs_from_image(self, image):
        detections = self.detect_faces_raw(image)
        rects_and_confs = self.get_rects_and_confs(image, detections)

        return rects_and_confs

    def detect_and_draw(self, image, conf_label=False):
        rects_and_confs = self.get_rects_and_confs_from_image(image)

        colors = np.random.uniform(0, 255, size=(len(rects_and_confs), 3))

        for (i, (box, conf)) in enumerate(rects_and_confs):
            x0, y0, x1, y1 = box
            color = colors[i]

            cv2.rectangle(image, (x0 - 5, y0 - 5), (x1 + 5, y1 + 5), color, cv2.FILLED)

            if conf_label:
                y = (y0 - 10) if ((y0 - 10) > 0) else (y0 + 10)
                cv2.putText(image, 'Conf: {:.2f}'.format(conf), (x0, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

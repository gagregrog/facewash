import numpy as np
import imutils
from imutils import paths
import pickle
import cv2
import os
import utils as u
from detector import Detector

minDim = 10
dirname = os.path.dirname(__file__)

embedding_model = 'openface_nn4.small2.v1.t7'

default_input = os.path.sep.join([dirname, 'data', 'img'])
embedding_path = u.get_model_path(dirname, embedding_model)
default_output = os.path.sep.join([dirname, 'data', 'pickle', 'embeddings.pickle'])


class Extractor:
    def __init__(self, width=600, min_conf=0.5, min_dim=minDim):
        self.detector = Detector(min_conf=min_conf)
        self.embedder = cv2.dnn.readNetFromTorch(embedding_path)
        self.width = width
        self.min_dim = min_dim

    def resize(self, image):
        if self.width == 0:
            return image

        h, w = image.shape[:2]

        if w != self.width:
            image = imutils.resize(image, width=self.width)

        return image

    def _get_vec_from_box(self, image, box):
        x0, y0, x1, y1 = box

        # extract the face ROI and dims
        face = image[y0:y1, x0:x1]
        fH, fW = face.shape[:2]

        # ensure the face is large enough
        if fW < self.min_dim or fH < self.min_dim:
            return None

        # construct a blob from the ROI and pass it through the
        # embedding model to get the 128-d face quantification

        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.embedder.setInput(faceBlob)
        vec = self.embedder.forward()

        return vec

    def _get_vec_from_detections(self, image, detections, index):
        image = self.resize(image)

        box = self.detector.get_box(image, detections, index)

        # ensure the detection meets the min conf
        if box is not None:
            vec = self._get_vec_from_box(image, box)
            flattened = vec.flatten()

            return flattened

    def get_boxes_and_embeddings(self, image):
        image = self.resize(image)
        boxes = self.detector.get_boxes_from_image(image)

        vecs = [self._get_vec_from_box(image, box) for box in boxes]

        return boxes, vecs

    def extract_and_write_embeddings(self, input=default_input, output=default_output):
        imagePaths = list(paths.list_images(input))

        embeddings = []
        names = []

        for (i, imagePath) in enumerate(imagePaths):
            # gets "name" assuming input/name/xxx.jpg
            name = imagePath.split(os.path.sep)[-2]

            image = cv2.imread(imagePath)
            image = self.resize(image)

            detections = self.detector.detect_faces_raw(image)

            # at least one face has been found
            if len(detections) > 0:
                # assume each image has only one face
                # grab the one with the largest probability
                i = np.argmax(detections[0, 0, :, 2])

                vec = self._get_vec_from_detections(image, detections, index=i)

                if vec is not None:
                    # add the name of the person and corresponding face
                    # embedding to their respective lists
                    names.append(name)
                    embeddings.append(vec)

        data = {'embeddings': embeddings, 'names': names}

        with open(output, 'wb') as f:
            f.write(pickle.dumps(data))


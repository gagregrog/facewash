import numpy as np
import imutils
from imutils import paths
import pickle
import cv2
import os
import utils as u
from recognizer import Recognizer

minDim = 10
dirname = os.path.dirname(__file__)

embedding_model = 'openface_nn4.small2.v1.t7'

embeddingPath = u.get_model_path(dirname, embedding_model)
defaultOutput = os.path.sep.join([dirname, 'embeddings', 'output', 'embeddings.pickle'])


class Extractor:
    def __init__(self, width=600, min_conf=0.5, min_dim=minDim):
        # for extracting 128-d facial embeddings for training a recognizer
        self.recognizer = Recognizer(min_conf=min_conf)
        self.embedder = cv2.dnn.readNetFromTorch(embeddingPath)
        self.width = width
        self.min_dim = min_dim

    def resize(self, image):
        if self.width == 0:
            return image

        h, w = image.shape[:2]

        if w != self.width:
            image = imutils.resize(image, width=self.width)

        return image

    def _get_vec_from_detections(self, image, detections, index):
        image = self.resize(image)

        box, conf = self.recognizer.get_box_and_conf(image, detections, index)

        # ensure the detection meets the min conf
        if conf is not None:
            x0, y0, x1, y1 = box

            # extract the face ROI and dims
            face = image[y0:y1, x0:x1]
            fH, fW = face.shape[:2]

            # ensure the face is large enough
            if fW < minDim or fH < minDim:
                return None

            # construct a blob from the ROI and pass it through the
            # embedding model to get the 128-d face quantification

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            self.embedder.setInput(faceBlob)
            vec = self.embedder.forward()
            flattened = vec.flatten()

            return flattened

    def extract_embeddings(self, input, output=defaultOutput):
        imagePaths = list(paths.list_images(input))

        embeddings = []
        names = []

        for (i, imagePath) in enumerate(imagePaths):
            # gets "name" assuming input/name/xxx.jpg
            name = imagePath.split(os.path.sep)[-2]

            image = cv2.imread(imagePath)
            image = self.resize(image)

            detections = self.recognizer.detect_faces_raw(image)

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


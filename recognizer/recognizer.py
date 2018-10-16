from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

caffe = 'res10_300x300_ssd_iter_140000.caffemodel'
proto = 'deploy.prototxt'
embedding_model = 'openface_nn4.small2.v1.t7'
dirname = os.path.dirname(__file__)
modelPath = os.path.sep.join([dirname, 'models', caffe])
protoPath = os.path.sep.join([dirname, 'models', proto])
embeddingPath = os.path.sep.join([dirname, 'models', embedding_model])
defaultOutput = os.path.sep.join([dirname, 'embeddings', 'output', 'embeddings.pickle'])
minDim = 20


class Recognizer:
    def __init__(self, min_conf=0.5):        
        # initialize the neural nets
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        self.embedder = cv2.dnn.readNetFromTorch(embeddingPath)
        self.min_conf = min_conf
        self.background = None

    @staticmethod
    def box_to_ellipse(box):
        x0, y0, x1, y1 = box
        center_x = int(((x0 + x1) / 2))
        center_y = int(((y0 + y1) / 2))
        width = int(np.abs(x0 - x1) / 2)
        height = int(np.abs(y0 - y1) / 2)
        center = (center_x, center_y)
        axes = (width, height)

        return center, axes

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

    def remove_faces(self, image, background=None, padding=None):
        rects_and_confs = self.get_rects_and_confs_from_image(image)
        h, w = image.shape[:2]

        # set the replacement background from an empty frame
        # assuming no background was provided
        if len(rects_and_confs) == 0 and background is None:
            self.background = image

        replacement = background if background is not None else self.background

        # replacement is background if background is provided,
        # otherwise replacement is the most recent empty frame
        if replacement is not None:
            for (box, conf) in rects_and_confs:
                x0, y0, x1, y1 = box

                if padding is not None:
                    x0 = x0 - padding if x0 - padding > 0 else 0
                    y0 = y0 - padding if y0 - padding > 0 else 0
                    x1 = x1 + padding if x1 + padding < w else w
                    y1 = y1 + padding if y1 + padding < h else h

                image[y0:y1, x0:x1] = replacement[y0:y1, x0:x1]

    def blur_faces(self, image, kernal_size=50):
        rects_and_confs = self.get_rects_and_confs_from_image(image)

        for r_a_c in rects_and_confs:
            box = r_a_c[0]
            x0, y0, x1, y1 = box
            face = image[y0:y1, x0:x1]
            face = cv2.blur(face, (kernal_size, kernal_size))

            image[y0:y1, x0:x1] = face

            center, dims = Recognizer.get_center_from_box(box)
            cv2.ellipse(image, center, dims, 0, 0, 360, (0, 255, 0), 2)

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
    
    def detect_and_mask(self, image):
        detections = self.detect_faces_raw(image)
        rects_and_confs = self.get_rects_and_confs(image, detections)

        for (box, conf) in rects_and_confs:
            x0, y0, x1, y1 = box

            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)

            if conf_label:
                y = (y0 - 10) if ((y0 - 10) > 0) else (y0 + 10)
                cv2.putText(image, 'Conf: {:.2f}'.format(conf), (x0, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    def extract_embeddings(self, input, output=defaultOutput):
        imagePaths = list(paths.list_images(input))

        embeddings = []
        names = []

        for (i, imagePath) in enumerate(imagePaths):
            # gets "name" assuming input/name/xxx.jpg
            name = imagePath.split(os.path.sep)[-2]

            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)

            detections = self.detect_faces_raw(image)

            # at least one face has been found
            if len(detections) > 0:
                # assume each image has only one face
                # grab the one with the largest probability
                i = np.argmax(detections[0, 0, :, 2])

                box, conf = self.get_rect_and_conf()(image, detections, index=i)

                # ensure the detection meets the min conf
                if conf is not None:
                    x0, y0, x1, y1 = box

                    # extract the face ROI and dims
                    face = image[y0:y1, x0:x1]
                    fH, fW = face.shape[:2]

                    # ensure the face is large enough
                    if fW < minDim or fH < minDim:
                        continue

                    # construct a blob from the ROI and pass it through the
                    # embedding model to get the 128-d face quantification

                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    self.embedder.setInput(faceBlob)
                    vec = self.embedder.forward()

                    # add the name of the person and corresponding face
                    # embedding to their respective lists
                    names.append(name)
                    embeddings.append(vec.flatten())

        data = {'embeddings': embeddings, 'names': names}

        with open(output, 'wb') as f:
            f.write(pickle.dumps(data))


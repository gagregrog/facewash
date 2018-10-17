from imutils import paths
import numpy as np
import imutils
from imutils import face_utils
import pickle
import dlib
import cv2
import os
import recognizer.util as u

minDim = 10
dirname = os.path.dirname(__file__)

proto = 'deploy.prototxt'
embedding_model = 'openface_nn4.small2.v1.t7'
caffe = 'res10_300x300_ssd_iter_140000.caffemodel'
landmark = 'shape_predictor_5_face_landmarks.dat'

caffePath = u.get_model_path(dirname, caffe)
protoPath = u.get_model_path(dirname, proto)
landmarkPath = u.get_model_path(dirname, landmark)
embeddingPath = u.get_model_path(dirname, embedding_model)


defaultOutput = os.path.sep.join([dirname, 'embeddings', 'output', 'embeddings.pickle'])


class Recognizer:
    def __init__(self, min_conf=0.5, width=600):
        # initialize the neural nets and models

        # for detecting faces in a frame
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, caffePath)
        # for extracting 128-d facial embeddings for training a recognizer
        self.embedder = cv2.dnn.readNetFromTorch(embeddingPath)
        # for extracting facial landmarks
        self.landmarker = dlib.shape_predictor(landmarkPath)

        self.min_conf = min_conf
        self.background = None
        self.width = width

    def resize(self, image):
        if self.width == 0:
            return image

        h, w = image.shape[:2]

        if w != self.width:
            image = imutils.resize(image, width=self.width)

        return image

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
        tempImg = image.copy()
        # height, width, 1 channel
        maskShape = (image.shape[0], image.shape[1], 1)

        # fill it with zeros, an empty canvas
        mask = np.full(maskShape, 0, dtype=np.uint8)

        rects_and_confs = self.get_rects_and_confs_from_image(image)

        for r_a_c in rects_and_confs:
            box = r_a_c[0]
            x0, y0, x1, y1 = box

            tempImg[y0:y1, x0:x1] = cv2.blur(tempImg[y0:y1, x0:x1], (kernal_size, kernal_size))

            center, dims = u.box_to_ellipse(box)

            # solid ellipse on mask
            cv2.ellipse(mask, center, dims, 0, 0, 360, (255), -1)

        # get everything but the ellipse
        mask_inv = cv2.bitwise_not(mask)
        no_faces = cv2.bitwise_and(image, image, mask=mask_inv)
        blurred_faces = cv2.bitwise_and(tempImg, tempImg, mask=mask)
        composite = cv2.add(no_faces, blurred_faces)

        image[:, :] = composite[:, :]

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

    def get_facial_landmarks(self, image, boxes=None):
        if boxes is None:
            rs_and_cs = self.get_rects_and_confs_from_image(image)
            boxes = [r_and_c[0] for r_and_c in rs_and_cs]

        rects = u.bounding_boxes_to_dlib_rects(boxes)

        landmarks = []

        for rect in rects:
            shape = self.landmarker(image, rect)
            shape = face_utils.shape_to_np(shape)
            landmarks.append(shape)

        return landmarks

    def draw_5_point_landmarks(self, image, boxes=None):
        landmarks = self.get_facial_landmarks(image, boxes=boxes)

        for face in landmarks:
            for (x, y) in face:
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    def draw_landmarks_and_rect(self, image, show_angle=True):
        rs_a_cs = self.get_rects_and_confs_from_image(image)

        boxes = [a[0] for a in rs_a_cs]
        landmarks = self.get_facial_landmarks(image, boxes=boxes)
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        angles = self.get_angles_from_landmarks(landmarks=landmarks)
        if len(colors) == 1:
            colors = [(0, 255, 0)]

        for (box, landmark, angle, color) in zip(boxes, landmarks, angles, colors):
            x0, y0, x1, y1 = box
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

            if show_angle:
                cv2.putText(image, 'Angle: {:.2f}'.format(angle), (x0, y0 - 10 if y0 - 10 > 0 else y0 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            for (x, y) in landmark:
                cv2.circle(image, (x, y), 3, color, -1)

    def get_angles_from_landmarks(self, image=None, boxes=None, landmarks=None):
        if landmarks is None:
            landmarks = self.get_facial_landmarks(image, boxes)

        angles = [-1 * u.angle_from_facial_landmarks(landmark) for landmark in landmarks]

        return angles

    def get_vec_from_detections(self, image, detections, index):
        image = self.resize(image)

        box, conf = self.get_rect_and_conf(image, detections, index=index)

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

    def get_vecs_from_image(self, image):
        detections = self.detect_faces_raw(image)
        vecs = []

        for i in range(0, detections.shape[2]):
            vec = self.get_vec_from_detections(image, detections, index=i)

            if vec is not None:
                vecs.append(vec)

        return vecs

    def extract_embeddings(self, input, output=defaultOutput):
        imagePaths = list(paths.list_images(input))

        embeddings = []
        names = []

        for (i, imagePath) in enumerate(imagePaths):
            # gets "name" assuming input/name/xxx.jpg
            name = imagePath.split(os.path.sep)[-2]

            image = cv2.imread(imagePath)
            image = self.resize(image)

            detections = self.detect_faces_raw(image)

            # at least one face has been found
            if len(detections) > 0:
                # assume each image has only one face
                # grab the one with the largest probability
                i = np.argmax(detections[0, 0, :, 2])

                vec = self.get_vec_from_detections(image, detections, index=i)

                if vec is not None:
                    # add the name of the person and corresponding face
                    # embedding to their respective lists
                    names.append(name)
                    embeddings.append(vec)

        data = {'embeddings': embeddings, 'names': names}

        with open(output, 'wb') as f:
            f.write(pickle.dumps(data))


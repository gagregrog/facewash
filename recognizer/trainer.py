from sklearn.preprocessing import LabelEncoder
from imutils.video import VideoStream
from sklearn.svm import SVC
import uuid
from time import sleep
import numpy as np
import pickle
import cv2
import os
from detector import Detector

dirname = os.path.dirname(__file__)

default_embedding_path = os.path.sep.join([dirname, 'data', 'pickle', 'embeddings.pickle'])
default_recognizer_path = os.path.sep.join([dirname, 'data', 'pickle', 'recognizer.pickle'])
default_le_path = os.path.sep.join([dirname, 'data', 'pickle', 'le.pickle'])
default_image_path = os.path.sep.join([dirname, 'data', 'img'])


def train_model(embedding_path=default_embedding_path, recognizer_path=default_recognizer_path,
                le_path=default_le_path):
    data = None

    with open(embedding_path, 'rb') as f:
        data = pickle.loads(f.read())

    le = LabelEncoder()
    labels = le.fit_transform(data['names'])

    # train the model that accepts the 128-d embeddings of the face
    # generate the recognizer
    recognizer = SVC(C=1.0, kernel='linear', probability=True)
    recognizer.fit(data['embeddings'], labels)

    # write the recognizer to disk
    with open(recognizer_path, 'wb') as f:
        f.write(pickle.dumps(recognizer))

    # write the label encoder to disk
    with open(le_path, 'wb') as f:
        f.write(pickle.dumps(le))


def generate_training_images(src=0, output=default_image_path, num_pics=10, name=None):
    detector = Detector(colors=[(0, 255, 0)])

    if output == default_image_path:
        if name is None:
            name = str(uuid.uuid4())

        output = os.path.sep.join([output, name])

    if not os.path.exists(output):
        os.makedirs(output)

    imgs = []
    vs = VideoStream(src=src)
    vs.start()
    sleep(2)
    snapped = False
    i = 0
    while len(imgs) < num_pics or i < 3:
        frame = vs.read()
        copy = frame.copy()
        boxes_and_confs = detector.draw_boxes(copy)

        pic_num = len(imgs) + 1
        cv2.putText(copy, '{} / {}'.format(pic_num, num_pics), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        if float(i / 10) == 1.0:
            i = 1
            if len(boxes_and_confs) > 0:
                imgs.append(frame)
                snapped = True

        elif i == 0:
            i += 1
        elif i < 3 and snapped is True:
            white = (copy.shape[0], copy.shape[1], 1)
            copy = np.full(white, 254, dtype=np.uint8)
        elif i == 3 and snapped is True:
            snapped = False

        i += 1

        cv2.imshow('Training Images', copy)
        cv2.waitKey(1) & 0xFF

    vs.stop()
    cv2.destroyAllWindows()

    for (i, img) in enumerate(imgs):
        cv2.imwrite(os.path.sep.join([output, str(i)]) + '.jpg', img)

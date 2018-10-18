import numpy as np
import pickle
import cv2
import os
from recognizer.extractor import Extractor

minDim = 10
dirname = os.path.dirname(__file__)

default_recognizer_path = os.path.sep.join([dirname, 'data', 'pickle', 'recognizer.pickle'])
default_le_path = os.path.sep.join([dirname, 'data', 'pickle', 'le.pickle'])


class Recognizer:
    def __init__(self, recognizer=default_recognizer_path, le=default_le_path, width=600, min_conf=0.5, min_dim=minDim):
        with open(recognizer, 'rb') as f:
            self.recognizer = pickle.loads(f.read())

        with open(le, 'rb') as f:
            self.le = pickle.loads(f.read())

        self.extractor = Extractor(width=width, min_conf=min_conf, min_dim=minDim)
        self.width = width

    def recognize(self, image, draw=False, with_prob=False, colors=None):
        w = image.shape[1]
        r = w / self.width

        boxes, vecs = self.extractor.get_boxes_and_embeddings(image)

        if colors is None or len(colors) < len(boxes):
            colors = self.extractor.detector.colors

        for (i, (box, vec)) in enumerate(zip(boxes, vecs)):
            try:
                predictions = self.recognizer.predict_proba(vec)[0]
                highest = np.argmax(predictions)
                probability = predictions[highest]
                name = self.le.classes_[highest]

                if draw:
                    box = [int(a) for a in (np.array(box) * r)]
                    x0, y0, x1, y1 = box

                    color = colors[i]
                    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

                if with_prob:
                    text = '{}: {:.2f}'.format(name, probability)
                    y = y0 - 10 if y0 - 10 > 0 else y0 + 10
                    cv2.putText(image, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            except Exception as e:
                print('__ERROR__')
                print(e)

    def recognize_and_draw(self, image, colors=None):
        self.recognize(image, draw=True, with_prob=True, colors=colors)

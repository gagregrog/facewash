import numpy as np
import pickle
import cv2
import os
from recognizer.extractor import Extractor
from recognizer.trainer import train_model

minDim = 10
dirname = os.path.dirname(__file__)

default_recognizer_path = os.path.sep.join([dirname, 'data', 'pickle', 'recognizer.pickle'])
default_le_path = os.path.sep.join([dirname, 'data', 'pickle', 'le.pickle'])


class Recognizer:
    def __init__(self, recognizer_path=default_recognizer_path, le_path=default_le_path, width=600, min_conf=0.5, min_dim=minDim, embedding_path=None):
        self.recognizer_path = recognizer_path
        self.le_path = le_path
        self.recognizer = None
        self.le = None
        self.embedding_path = embedding_path
        self.load_models()

        self.extractor = Extractor(width=width, min_conf=min_conf, min_dim=minDim)
        self.width = width

    def load_models(self):
        try:
            with open(self.recognizer_path, 'rb') as f:
                self.recognizer = pickle.loads(f.read())

            with open(self.le_path, 'rb') as f:
                self.le = pickle.loads(f.read())

        except Exception as e:
            print('Failed to read recognizer pickles.')
            print(e)

    def recognize(self, image, draw=False, with_prob=False, colors=None):
        if self.le is None:
            print('Recognizer models not loaded.')
            return

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

    def extract_and_train(self, training_images_path=None):
        extractor_args = {}
        trainer_args = {}

        if self.embedding_path is not None:
            extractor_args['output'] = self.embedding_path
            trainer_args['embedding_path'] = self.embedding_path

        if training_images_path is not None:
            extractor_args['training_images_path'] = training_images_path

        self.extractor.extract_and_write_embeddings(**extractor_args)

        train_model(**trainer_args, recognizer_path=self.recognizer_path, le_path=self.le_path)

        self.load_models()

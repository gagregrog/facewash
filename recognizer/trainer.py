from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os

dirname = os.path.dirname(__file__)

embedding_model = 'openface_nn4.small2.v1.t7'

default_embedding_path = os.path.sep.join([dirname, 'data', 'pickle', 'embeddings.pickle'])
default_recognizer_path = os.path.sep.join([dirname, 'data', 'pickle', 'recognizer.pickle'])
default_le_path = os.path.sep.join([dirname, 'data', 'pickle', 'le.pickle'])


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

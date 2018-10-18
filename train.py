import argparse
from recognizer.trainer import train_model

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--embeddings', help='path to serialized db of facial embeddings')
    ap.add_argument('-r', '--recognizer', help='path to serialized db of facial embeddings')
    ap.add_argument('-l', '--le', help='path to output label encoder')
    args = ap.parse_args()

    passedArgs = {}

    if args.embeddings is not None:
        passedArgs['embedding_path'] = args.embeddings

    if args.recognizer is not None:
        passedArgs['recognizer_path'] = args.recognizer

    if args.le is not None:
        passedArgs['le_path'] = args.le

    train_model(**passedArgs)

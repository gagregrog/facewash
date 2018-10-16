import argparse
from recognizer.recognizer import Recognizer

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
                    help='Path to image dataset for training')
    ap.add_argument('-o', '--output', 
                    help='Path to output serialized facial embeddings')
    ap.add_argument('-c', '--conf', type=float,
                    help='Minimum probability for filtering weak detections')
    args = ap.parse_args()

    init = {'min_conf': args.conf} if args.conf is not None else {}

    recognizer = Recognizer(**init)
    
    passedArgs = {'input': args.input}

    if args.output is not None:
        passedArgs['output'] = args.output

    recognizer.extract_embeddings(**passedArgs)

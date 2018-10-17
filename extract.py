import argparse
from extractor import Extractor

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
                    help='Path to image dataset for training')
    ap.add_argument('-o', '--output', 
                    help='Path to output serialized facial embeddings')
    ap.add_argument('-c', '--conf', type=float, default=0.5,
                    help='Minimum probability for filtering weak detections')
    args = ap.parse_args()

    extractor = Extractor(min_conf=args.conf)
    
    passedArgs = {'input': args.input}

    if args.output is not None:
        passedArgs['output'] = args.output

    extractor.extract_embeddings(**passedArgs)

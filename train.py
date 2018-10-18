import argparse
from recognizer import trainer

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--embeddings', help='path to serialized db of facial embeddings')
    ap.add_argument('-r', '--recognizer', help='path to serialized db of facial embeddings')
    ap.add_argument('-l', '--le', help='path to output label encoder')
    ap.add_argument('-g', '--get-faces', default=False, action='store_true')
    ap.add_argument('-o', '--output-path')
    ap.add_argument('-s', '--src', default=0)
    ap.add_argument('-p', '--num-pics', default=10, type=int)
    ap.add_argument('-n', '--name')
    args = ap.parse_args()

    if args.get_faces:
        passed_args = {'src': args.src, 'num_pics': args.num_pics}

        if args.output_path is not None:
            passed_args['output'] = args.output_path

        if args.name is not None:
            passed_args['name'] = args.name

        trainer.generate_training_images(**passed_args)
    else:
        passedArgs = {}

        if args.embeddings is not None:
            passedArgs['embedding_path'] = args.embeddings

        if args.recognizer is not None:
            passedArgs['recognizer_path'] = args.recognizer

        if args.le is not None:
            passedArgs['le_path'] = args.le

        trainer.train_model(**passedArgs)

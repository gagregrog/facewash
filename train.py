import argparse
from recognizer import trainer
from recognizer import Recognizer

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-e', '--embedding-path', help='path to serialized db of facial embeddings')
    ap.add_argument('-r', '--recognizer-path', help='path to serialized db of recognized embeddings')
    ap.add_argument('-l', '--le-path', help='path to output label encoder')
    ap.add_argument('-g', '--get-faces', default=False, action='store_true')
    ap.add_argument('-o', '--image-output-path')
    ap.add_argument('-i', '--training-images-path')
    ap.add_argument('-s', '--src', default=0)
    ap.add_argument('-p', '--num-pics', default=10, type=int)
    ap.add_argument('-n', '--name')
    ap.add_argument('-t', '--train', default=False, action='store_true')
    ap.add_argument('-x', '--extract', default=False, action='store_true')
    ap.add_argument('-w', '--width', default=600, type=int)
    ap.add_argument('-c', '--conf', default=0.5, type=float)
    args = ap.parse_args()

    if args.get_faces:
        passed_args = {'src': args.src, 'num_pics': args.num_pics}

        if args.image_output_path is not None:
            passed_args['output'] = args.image_output_path

        if args.name is not None:
            passed_args['name'] = args.name

        trainer.generate_training_images(**passed_args)
    else:
        passedArgs = {}
        extractArgs = {}

        if args.embedding_path is not None:
            passedArgs['embedding_path'] = args.embedding_path
            extractArgs['embedding_path'] = args.embedding_path

        if args.recognizer_path is not None:
            passedArgs['recognizer_path'] = args.recognizer_path

        if args.le_path is not None:
            passedArgs['le_path'] = args.le_path

        if args.train:
            trainer.train_model(**passedArgs)
        else:
            passedArgs['width'] = args.width
            passedArgs['min_conf'] = args.conf
            recognizer = Recognizer(**passedArgs)

            if args.training_images_path is not None:
                extractArgs['training_images_path'] = args.training_images_path

            if args.extract:
                recognizer.extractor.extract_and_write_embeddings(**extractArgs)
            else:
                if 'embedding_path' in extractArgs:
                    del extractArgs['embedding_path']
                recognizer.extract_and_train(**extractArgs)

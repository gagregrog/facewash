import cv2
import imutils
import argparse
from time import sleep
from imutils.video import VideoStream
from detector import Detector
from recognizer import Recognizer

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--src', default=0, type=int)
ap.add_argument('-c', '--conf', default=0.5, type=float)
ap.add_argument('-b', '--blur', default=False, action='store_true')
ap.add_argument('-l', '--landmarks', default=False, action='store_true')
ap.add_argument('-x', '--remove', default=False, action='store_true')
ap.add_argument('-ff', '--first-frame', default=False, action='store_true')
ap.add_argument('-bg', '--background')
ap.add_argument('-p', '--padding', type=int)
ap.add_argument('-w', '--width', default=600, type=int)
ap.add_argument('-r', '--recognize', default=False, action='store_true')
ap.add_argument('-rp', '--recognizer-path')
ap.add_argument('-lp', '--label-path')
args = ap.parse_args()

detector = None
recognizer = None

if args.recognize:
    passed_args = {'min_conf': args.conf}

    if args.recognizer_path is not None:
            passed_args['recognizer'] = args.recognizer_path

    if args.label_path is not None:
            passed_args['le'] = args.label_path

    recognizer = Recognizer(**passed_args)
else:
    detector = Detector(min_conf=args.conf)

vs = VideoStream(src=args.src)

vs.start()
sleep(2)
background = None
if args.background is not None:
    background = args.background

while True:
    image = vs.read()

    if args.width != 0:
        image = imutils.resize(image, width=args.width)

    if args.recognize:
        recognizer.recognize_and_draw(image)
    elif args.blur:
        detector.blur_faces(image, padding=args.padding)
    elif args.landmarks:
        detector.draw_boxes_angles_and_landmarks(image, show_angle=True)
    elif args.remove:
        passed_args = {'image': image}

        if background is None and args.first_frame:
            background = image
            continue

        if background is not None:
            passed_args['background'] = background

        if args.padding is not None:
            passed_args['padding'] = args.padding

        detector.remove_faces(**passed_args)
    else:
        detector.draw_boxes(image, conf_label=True)

    cv2.imshow('Faces', image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()

import cv2
import imutils
import argparse
from time import sleep
from imutils.video import VideoStream
from recognizer import Recognizer

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--padding', type=int)
ap.add_argument('-b', '--background', default=None)
ap.add_argument('-s', '--src', default=0, type=int)
ap.add_argument('-c', '--conf', default=0.3, type=float)
ap.add_argument('-f', '--first-frame', action='store_true', default=False)
args = ap.parse_args()

recognizer = Recognizer(min_conf=args.conf)

vs = VideoStream(src=args.src)

vs.start()
sleep(2)
background = cv2.imread(args.background) if args.background is not None else None

while True:
    image = vs.read()
    image = imutils.resize(image, width=600)

    if background is None and args.first_frame:
        background = image
        continue

    arguments = {
        'image': image,
        'padding': args.padding
    }

    if background is not None:
        arguments['background'] = background

    recognizer.remove_faces(**arguments)
    cv2.imshow('Removed Faces', image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()

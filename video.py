import cv2
import imutils
import argparse
from time import sleep
from imutils.video import VideoStream
from recognizer import Recognizer

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--src', default=0, type=int)
ap.add_argument('-c', '--conf', default=0.5, type=float)
ap.add_argument('-b', '--blur', default=False, action='store_true')
args = ap.parse_args()

recognizer = Recognizer(min_conf=args.conf)

vs = VideoStream(src=args.src)

vs.start()
sleep(2)

while True:
    image = vs.read()
    image = imutils.resize(image, width=600)

    if args.blur:
        recognizer.blur_faces(image)
    else:
        recognizer.draw_boxes(image, conf_label=True)

    cv2.imshow('Faces', image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()

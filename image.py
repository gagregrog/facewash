import cv2
import imutils
import argparse
from recognizer.recognizer import Recognizer

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True)
ap.add_argument('-c', '--conf', default=0.5, type=float)
args = ap.parse_args()

recognizer = Recognizer(min_conf=args.conf)

image = cv2.imread(args.image)
image = imutils.resize(image, width=600)
recognizer.detect_and_draw(image)
cv2.imshow('Faces', image)
key = cv2.waitKey(0)

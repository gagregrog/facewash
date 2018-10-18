import cv2
import imutils
import argparse
from detector import Detector

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True)
ap.add_argument('-c', '--conf', default=0.5, type=float)
args = ap.parse_args()

detector = Detector(min_conf=args.conf)

image = cv2.imread(args.image)
image = imutils.resize(image, width=600)
detector.draw_boxes(image)
cv2.imshow('Faces', image)
key = cv2.waitKey(0)

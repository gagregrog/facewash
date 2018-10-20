# Facewash

## Install

1. [Setup and activate a virtualenv](https://docs.python-guide.org/dev/virtualenvs/)
2. Install project requirements: `pip install -r requirements.txt`
3. [Install OpenCV 4 from source](https://www.pyimagesearch.com/category/opencv-4/) or, install OpenCV 3.4 using pip `pip install opencv-contrib-python`

## CLI

### `video.py`

Use a camera to detect faces and perform various operations in a live feed. 

Only one operation can run at a time. 

Operations take precedence in the following order:

  1.  `-r` Recognizing faces
  2.  `-b` Blurring faces
  3.  `-l` Drawing landmarks
  4.  `-x` Removing faces
  5.  `default` Drawing faces

#### Flags

- `-s`, `--src`

  Video input source. Defaults to `0` for built in webcam. Provide `1` for a usb camera source.

- `-c`, `--conf`

  Minimum confidence for detecting a face. Defaults to `0.5`. Accepts values `(0, 1)`.

- `-w`, `--width`
  
  Width to resize stream to. Defaults to 600px. Pass `0` to not resize.

##### Operations

- `-r`, `--recognize`
  
  Recognize faces in the video stream. Requires a pre-trained model. Use with the following two flags.
  
  - `-rp`, `--recognizer-path`

  Path to the trained recognizer. Defaults to `facewash/recognizer/pickle/recognizer.pickle`
    
  - `-lp`, `--label-path`

  Path to the corresponding labels. Defaults to `facewash/recognizer/pickle/le.pickle`


- `-b`, `--blur`
  
  Blur detected faces in the video stream.

- `-l`, `--landmarks`
  
  Draw 5 point facial landmarks and display angle of the head.

- `-x`, `--remove`
  
  Remove faces found in the video stream. Each frame without detected faces gets set as the background image.
  
  Use with the following 3 flags to refine.

  - `-ff`, `--first-frame`
  
    When removing faces, use the first frame as the replacement image.

  - `-bg`, `--background`
  
    When removing faces, provide the path to an image to use as the replacement image.

  - `-p`, `--padding`
  
    When removing faces, add extra padding around the ROI for better coverage.

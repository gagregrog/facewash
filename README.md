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

### `train.py`

Train a `Recognizer` to recognize faces in a video or image.

By default, the script will extract facial embeddings and and train a model. You can override this behavior and do just one or the other by providing the appropriate flag. You can also use the webcam to take pictures for training.

Operations take precedence in the following order:

  1.  `-g` Use the webcam to get face screen shots
  2.  `-t` Train the model on a preexisting set of embeddings
  3.  `-x` Extract facial embeddings from a set of images
  4.  `default` Extract facial embeddings _and_ train a model

#### Flags

##### Operations

- `-g`, `--get-faces`
  
  Use a camera to take pictures for training a model.

  Use with the following 4 flags to refine.

  - `-s`, `--src`

  Video input source. Defaults to `0` for built in webcam. Provide `1` for a usb camera source.
  
  - `-p`, `--num-pics`

  The number of images to capture. Defaults to `10`.
    
  - `-o`, `--image-output-path`

  Directory to save the files. The directory will be created if it does not exist. The directory should end in the name of the person in the photos, as this is used for labeling.

  If omitted it will default to `facewash/recognizer/data/img/<uuid>`.

  - `-n`, `--name`

  Name of the person in the photos. Ignored if an `image-output-path` is provided. Otherwise, will save photos to the directory `facewash/recognizer/data/img/<name>`

- `-x`, `--extract`

  Extract facial embeddings from a directory containing directories containing images of people's faces.
  
  Use with the following 5 flags to refine.

  - `-i`, `--training-images-path`
  
    Path to a directory of directories containing training images. Each directory in the path should be the name of the person in the photos inside the directory. The photos should only contain the person of interest.

    You may train on as many different people as you want. Each person will be trained on the minimum number of photos across the directories.

    A category of `unknown` will automatically be trained from a random subset of person images. You can see these images in `facewash/recognizer/data/unknown`.

    Defaults to `facewash/recognizer/data/img/`

  - `-e`, `--embedding-path`
  
    Location and name for saving the `pickle` file containing the facial embedding information. This is one of the two files needed for training a model. Filename should end in `.pickle`.

    Defaults to `facewash/recognizer/data/pickle/embeddings.pickle`
  
  - `-l`, `--le-path`
  
    Location and name for saving the `pickle` file containing the label information. This is one of the two files needed for training a model. Filename should end in `.pickle`.

    Defaults to `facewash/recognizer/data/pickle/le.pickle`

  - `-c`, `--conf`

    Minimum confidence for detecting a face in the training images. Defaults to `0.5`. Accepts values `(0, 1)`.

  - `-w`, `--width`
    
    Width to train the model on. Defaults to 600px. Best to leave this as is.

- `-t`, `--train`

  Use a set of extracted features and labels to train a SVM.
  
  Use with the following 3 flags to refine.

  - `-r`, `--recognizer-path`
  
    Location and name for saving the `pickle` file containing the recognizer model. Filename should end in `.pickle`. This is the file that will ultimately be passed to the `Recognizer` and used for identifying persons if interest.

    Defaults to `facewash/recognizer/data/pickle/recognizer.pickle`

  - `-e`, `--embedding-path`
  
    Path to the `pickle` file containing the facial embedding information obtained during extraction. 

    Defaults to `facewash/recognizer/data/pickle/embeddings.pickle`
  
  - `-l`, `--le-path`
  
    Path to the `pickle` file containing the label information obtained during extraction.

    Defaults to `facewash/recognizer/data/pickle/le.pickle`

- Extract and Train

    By omitting `-g`, `-x`, and `-t` flags it is assumed that you are extracting facial embeddings from an existing image set. Provide all of the necessary flags required for both extracting and training.

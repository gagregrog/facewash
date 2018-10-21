# Facewash

A set of facial detection, recognition, and transformation utilities for images and video streams. Use the CLI to get started, or import the modules into a project.

## Install

Requires python3.x and OpenCV >= 3.4

1. [Setup and activate a virtualenv](https://docs.python-guide.org/dev/virtualenvs/).
2. Install project requirements: `pip install -r requirements.txt`
3. [Install OpenCV 4 from source](https://www.pyimagesearch.com/category/opencv-4/) or, install OpenCV 3.4 using `pip install opencv-contrib-python`.

## Contents

- [CLI](#cli)
  - [video.py](#video.py)
  - [train.py](#train.py)
- [API](#api)
  - [Detector](#detector())
  - [Landmarker](#landmarker())
  - [Transformer](#transformer())
  - [Recognizer](#recognizer())
  - [Extractor](#extractor())
  - [Trainer](#trainer)

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

    [[top]](#facewash)

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

    [[top]](#facewash)

## API

### `Detector()`

Basic facial detection and transformations. Uses the `res10_300x300_ssd_iter_140000.caffemodel` for detection.

#### Optional Arguments

- `min_conf=0.5`

  Confidence threshold for classifying a region as a face.

- `with_landmarks=False`

  Initialize the facial landmark model. Automatically initialized if a landmark transformation is applied.

- `with_transformer=False`

  Initialize the facial transformation model. Automatically initialized if a facial transformation is applied.

- `colors=None`

  An array of colors of the form `[(0, 255, 0), (145, 145, 145), ...]` to be used for the first `n` detections. Randomly generated if omitted.

#### Instantiation

  ```
  from detector import Detector

  detector = Detector(min_conf=0.3)
  ```

#### Methods

##### `detector.detect_faces_raw(image)`

Requires a `cv2` image and returns the raw facial detections  including the confidences and localizations.

```
img = cv2.imread('people.jpg')

detections = detector.detect_faces_raw(img)
```

##### `detector.get_box_and_conf(image, detections, index)`

Requires a `cv2` image, raw facial detections, and an index of interest, returning a tuple including the coordinates of the facial box and the confidence. Returns (None, None) if the face is below the `min_conf` provided at initialization.

```
box, conf = detector.get_box_and_conf(img, detections, 0)

# returns the info for the first face detected in the image
# box => (x0, y0, x1, y1)
# conf => float (0, 1)
```

##### `detector.get_box(image, detections, index)`

As above, but only returns the bounding box coordinates.

##### `detector.get_boxes_and_confs(image, detections)`

Calls `get_box_and_conf` for all detections. Returns an array of tuples in the form (box, conf)  where box is a tuple containing (x0, y0, x1, y1).

```
boxes_and_confs = detector.get_boxes_and_confs(img, detections)
```

##### `detector.get_boxes_and_confs_from_image(image)`

A convenience method if you don't need the raw detections.

```
boxes_and_confs = detector.get_boxes_and_confs_from_image(img)
```

##### `detector.get_boxes_from_image(image)`

As above, but only returns the bounding boxes.

##### `detector.get_all_from_image(image)`

Returns an array of tuples containing `(boxes, confs, landmarks, angles)`.

##### `detector.draw_boxes(image, colors=None, conf_label=False, thickness=2)`

Detect faces and draw the bounding boxes.

This mutates the original image. If this is not desired, make a copy first with `img.copy()`.

If `colors` are provided to the method they will be used in lieu of `colors` passed when instantiating the `Detector`. If there are not enough colors provided, a random set of colors will be generated.

If `conf_label` is `True` the confidence values will be written to the screen above the ROI.

`thickness` is the thickness of the font and should be an `int`.

As a convenience, the boxes_and_confs will be returned.

```
detector.draw_boxes(img)

cv2.imshow('Faces', img)
cv2.waitKey(0)
```

##### `detector.draw_boxes_angles_and_landmarks(image, colors=None, show_angle=False)`

Draws the 5 point facial landmarks to the image.

`colors` as described above.

If `show_angle` is `True` the angle of the face will be printed to the screen.

This is a convenience method that accesses the `Landmarker` class.

```
detector.draw_boxes_angles_and_landmarks(img, show_angle=True)

cv2.imshow('Landmarks', img)
cv2.waitKey(0)
```

##### `detector.remove_faces(image, background=None, padding=None)`

Remove the detected faces from the image, replacing them with a splice from the `background` image supplied.

`padding` increases the size of the rectangular splice in both dimensions.

This is a convenience method that accesses the `Transformer` class.

```
detector.remove_faces(img, background='bkg.jpg', padding=15)

cv2.imshow('Headless', img)
cv2.waitKey(0)
```

##### `detector.blur_faces(image, kernal_size=50, padding=None)`

Blur the oval region over the detected faces in the image.

`kernal_size` affects the blurriness. Higher values are more blurred.

`padding` increases the area of the blurred region in each direction.

This is a convenience method that accesses the `Transformer` class.

```
detector.blur_faces(img)

cv2.imshow('Blurry', img)
cv2.waitKey(0)
```

[[top]](#facewash)

### `Landmarker()`

Extract the 5 point facial landmarks from a facial ROI. Uses `shape_predictor_5_face_landmarks.dat` for landmark detection.

The methods require the `boxes` returned from the `Detector` class, so while you *can* use this class directly, it makes more sense to access it through a `Detector` instance.

#### Instantiation

  Accessing directly:

  ```
  from detector import Detector
  from detector.landmarks import Landmarker

  detector = Detector()
  landmarker = Landmarker()
  ```

  Accessing through a detector:

  ```
  from detector import Detector

  detector = Detector(with_landmarks=True)
  landmarker = detector.landmarker
  ```

  **Note** If you do not supply `with_landmarks=True` you will not be able to directly access the `Landmarker` inside of the `Detector`.

#### Methods

##### `landmarker.get_facial_landmarks(image, boxes)`

Requires a `cv2` image and array of bounding boxes. Returns an array including facial landmarks for each face. Each set of facial landmarks is an array of xy-coordinates marking a point on the 2 corners of each eye and a point below the nose.

```
img = cv2.imread('people.jpg')

boxes = detector.get_boxes_from_image(img)
facial_landmarks = landmarker.get_facial_landmarks(img, boxes)
# => [
  [
    (x, y),
    (x, y),
    (x, y),
    (x, y),
    (x, y),
  ],
  ...
]
```

##### `landmarker.draw_5_point_landmark(image, landmarks, color=(0, 255, 0))`

Draw a single set of landmarks on an image.

```
landmarker.draw_5_point_landmark(img, facial_landmarks[0], (0, 0, 0))
```

##### `landmarker.draw_landmarks_and_boxes(image, boxes, colors=None, show_angle=False)`

Draw all landmarks and bounding boxes. If using the detector, it is a good idea to pass `detector.colors` to maintain consistent colors.

If `show_angle` is true the angle of the head is crudely calculated and displayed on screen.

```
landmarker.draw_landmarks_and_boxes(img, boxes, detector.colors, True)
```


##### `landmarker.get_angles_from_boxes(image, boxes)`

Returns an array of face angles for a given image.

```
angles = landmarker.get_angles_from_boxes(img, boxes)
```

[[top]](#facewash)

### `Transformer()`

Apply transformations to the faces in an image. The `Transofrmer` class relies on a `Detector`, and can be instantiated at the same time. As with the `Landmarker`, it makes the most sense to use this through the detector.

In fact, both transformations can be accessed in a simpler fashion directly off of the detector. See `Detector.remove_faces()` and `Detector.blur_faces()` above.

#### Optional Arguments

- `background=None`

  A background image to use when removing faces. Superceded by passing a background image directly to the method.

#### Instantiation

  Accessing directly:

  ```
  from detector import Detector
  from detector.transformer import Transformer

  detector = Detector()
  transformer = Transformer()
  ```

  Accessing through a detector:

  ```
  from detector import Detector

  detector = Detector(with_transformer=True)
  transformer = detector.transformer
  ```

  **Note** If you do not supply `with_transformer=True` you will not be able to directly access the `Transformer` inside of the `Detector`.

#### Methods

##### `transformer.remove_faces(image, boxes, background=None, padding=None)`

Requires a `cv2` image, array of bounding boxes, background image, and optionally padding. For each bounding box, the image inside the bounding box is replaced with the corresponding image contained in the background image.

The background should be the same shape as the original image. If boxes has a length of zero, `background` gets set as `self.background`.

`background` is *mostly* required. The exception being if you are using this method in a video loop.

`padding` allows you to increase the size of the rectangle around the face in all directions.

```
img = cv2.imread('people.jpg')
bkg = cv2.imread('people-bkg.jpg')

boxes = detector.get_boxes_from_image(img)
transformer.remove_faces(img, boxes, bkg, 10)

cv2.imshow('Faceless', img)
cv2.waitKey(0)
```

Or, more succinctly accessed through the `detector` instance:

```
img = cv2.imread('people.jpg')
bkg = cv2.imread('people-bkg.jpg')

detector.remove_faces(img, bkg, 10):

cv2.imshow('Faceless', img)
cv2.waitKey(0)
```

##### `transformer.blur_faces(image, boxes, angles, kernal_size=50, padding=None)`

For each box, blur the image with an ellipse that fills the box. Bigger `kernal_size` means blurrier. `padding` increases the area of the blurred image.

```
from detector.landmarks import Landmarker

landmarker = Landmarker()
angles = landmarker.get_angles_from_boxes(img, boxes)
transformer.blur_faces(img, boxes, angles)

cv2.imshow('Blurry', img)
cv2.waitKey(0)
```

Or, more succinctly accessed through the `detector` instance:

```
detector.blur_faces(img):

cv2.imshow('Blurry', img)
cv2.waitKey(0)
```

[[top]](#facewash)

### `Recognizer()`

Recognize faces in images after extracting facial embeddings and training models.


#### Optional Arguments

- `recognizer_path`

  Path to the trained model for recognizing faces. Defaults to `facewash/recognizer/data/pickle/recognizer.pickle`

- `le_path`

  Path to the labels corresponding to the loaded recognizer. Defaults to `facewash/recognizer/data/pickle/le.pickle`

- `width=600`

  The width for training the models and for adjusting the coordinates after recognizing. Best to leave this at the default.

- `min_conf=0.5`

  Confidence used to reject bad training images.

- `min_dim=10`

  Minimum side length for a detected face. Rejected if smaller.

- `embedding_path=None`

  Embedding path used for extracting and training. Defaults to `facewash/recognizer/data/pickle/embeddings.pickle`

#### Instantiation

  ```
  from recognizer import Recognizer

  recognizer = Recognizer()
  ```

#### Methods

##### `recognizer.load_models()`

You would really only need to use this if you trained a model and wanted to recognize in the same script. Otherwise, this method is called on instantiation. 

It loads the recognizer and labels into the neural net.

##### `recognizer.recognize(image, draw=False, with_prob=False, colors=None)`

Models must be loaded in memory for this to work. Returns None otherwise.

If `draw` is `True` then boxes and names are drawn on screen.

`with_prob` will draw the probabilities too.

It is typically ok to leave `colors` as `None`, as they will be pulled internally from the colors generated by the `Detector`.

The image is scanned and faces are detected. For each face, the `Extractor` is used to extract 128-d facial embeddings, which are used to detect the most likely face from the recognizer model.

The highest probability match is used to determine the appropriate label.

Detections are scaled to the image size based off of `self.width`, so it is crucial that `self.width` matches the `width` that the images were trained on.

```
img = cv2.imread('people.jpg')
recognizer.recognize(img, True, True)

cv2.imshow('Detected', img)
cv2.waitKey(0)
```

##### `recognizer.recognize_and_draw(image, colors=None)

Same as above, but draw boxes and confidences by default.

##### `recognizer.extract_and_train(training_images_path):

Convenience method for accessing `Extractor` and `Trainer` methods.

If an `embedding_path` was passed during instantiation, it will be used for extracting and training. Otherwise the default path of `facewash/recognizer/data/pickle/embeddings.pickle` will be used.

If `training_images_path` is provided, it will be passed to the extractor. Otherwise, the default path of `facewash/recognizer/data/img` will be used. This path needs to contain directories named by person including training images &mdash; photos containing only that person. The more the better.

As this is a convenience method, it calls `extractor.extract_and_write_embeddings()` and then `train_model()` for you.

The result is a recognizer file and label encoder file saved to the paths passed during instantiation, or the defaults otherwise.

After the model has been trained, `load_models` is called, so you could theoretically start recognizing faces immediately.

[[top]](#facewash)

### `Extractor()`

Extract 128-dimensional facial embeddings from a set of training images.

#### Optional Arguments

- `width=600`

  The width to resize training images to. This value needs to match the value of `recognizer.width` when recognizing faces. Best to leave this at the default.

- `min_conf=0.5`

  Confidence used to reject bad training images.

- `min_dim=10`

  Minimum side length for a detected face. Rejected if smaller.


#### Instantiation

  ```
  from recognizer.extractor import Extractor

  extractor = Extractor()
  ```

#### Methods

##### `extractor.get_boxes_and_embeddings(image)`

Given a `cv2` image, detect faces, returning the bounding boxes and extracted embeddings for each face. Returned as two arrays.

##### `extractor.extract_and_write_embeddings(training_images_path, embedding_path)`

Take the training images and extract the facial embeddings, writing the results to `embedding_path`. If no path is given,
defaults to `facewash/recognizer/data/pickle/embeddings.pickle`.

If `training_images_path` is not provided, it will default to `facewash/recognizer/data/img`. This path needs to contain directories named by person including training images &mdash; photos containing only that person. The more the better.

Don't forget that after you extract the facial embeddings, you still need to train a model before you can recognize images. Consider using `recognizer.extract_and_train(...)` instead.

[[top]](#facewash)

### Trainer

Not a class, merely a library providing two training functions.

#### Import

  ```
  from recognizer import trainer
  ```

#### Methods

##### `trainer.train_model(embedding_path, recognizer_path, le_path)`

Requires a `pickle` file (`embedding_path`) containing `embeddings` and `names`, extracted using `extractor.extract_and_write_embeddings(...)`.

`embedding_path` defaults to `facewash/recognizer/data/pickle/embeddings.pickle`.

`recognizer_path` is the output for the trained recognizer. It defaults to `facewash/recognizer/data/pickle/recognizer.pickle`. This is used in the `Recognizer` class to recognize faces.

`le_path` is the output for the label encoder. It defaults to `facewash/recognizer/data/pickle/le.pickle`. This is used in the `Recognizer` class to identify the recognized faces.


##### `trainer.generate_training_images(src=0, output, num_pics=10, name=None)`

Starts the webcam and grabs `num_pics` images from the camera. Every 10 frames a picture is taken (as long as a single face has been detected in the frame).

The default `output` is `facewash/recognizer/data/img/<uuid>`, or if you provide a `name` it will be `facewash/recognizer/data/img/<name>`. Or, provide your own path to a folder to dump images by providing `output`.

Since these images will be used to train the model, it is important to provide a variety of different facial expressions and angles. The more pictures you use, the better the model will be (theoretically).

[[top]](#facewash)

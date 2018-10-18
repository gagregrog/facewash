import numpy as np
import cv2
import utils as u


class Transformer:
    def __init__(self, background=None):
        self.background = background

    def remove_faces(self, image, boxes, background=None, padding=None):
        h, w = image.shape[:2]

        # set the replacement background from an empty frame
        # assuming no background was provided
        if len(boxes) == 0 and background is None:
            self.background = image

        replacement = background if background is not None else self.background

        # replacement is background if background is provided,
        # otherwise replacement is the most recent empty frame
        if replacement is not None:
            for box in boxes:
                x0, y0, x1, y1 = box

                if padding is not None:
                    x0 = x0 - padding if x0 - padding > 0 else 0
                    y0 = y0 - padding if y0 - padding > 0 else 0
                    x1 = x1 + padding if x1 + padding < w else w
                    y1 = y1 + padding if y1 + padding < h else h

                image[y0:y1, x0:x1] = replacement[y0:y1, x0:x1]

    def blur_faces(self, image, boxes, kernal_size=50):
        tempImg = image.copy()
        # height, width, 1 channel
        maskShape = (image.shape[0], image.shape[1], 1)

        # fill it with zeros, an empty canvas
        mask = np.full(maskShape, 0, dtype=np.uint8)

        for box in boxes:
            x0, y0, x1, y1 = box

            tempImg[y0:y1, x0:x1] = cv2.blur(tempImg[y0:y1, x0:x1], (kernal_size, kernal_size))

            center, dims = u.box_to_ellipse(box)

            # solid ellipse on mask
            cv2.ellipse(mask, center, dims, 0, 0, 360, (255), -1)

        # get everything but the ellipse
        mask_inv = cv2.bitwise_not(mask)
        no_faces = cv2.bitwise_and(image, image, mask=mask_inv)
        blurred_faces = cv2.bitwise_and(tempImg, tempImg, mask=mask)
        composite = cv2.add(no_faces, blurred_faces)

        image[:, :] = composite[:, :]

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
                x0, y0, x1, y1 = box if padding is None else u.pad_box(h, w, box, padding)

                image[y0:y1, x0:x1] = replacement[y0:y1, x0:x1]

    def blur_faces(self, image, boxes, angles=None, kernal_size=50, padding=None):
        tempImg = image.copy()
        tempImg[:, :] = cv2.blur(tempImg[:, :], (kernal_size, kernal_size))
        # height, width, 1 channel
        maskShape = (image.shape[0], image.shape[1], 1)

        # fill it with zeros, an empty canvas
        mask = np.full(maskShape, 0, dtype=np.uint8)
        h, w = image.shape[:2]

        for (i, box) in enumerate(boxes):
            if padding is not None:
                box = u.pad_box(h, w, box, padding)
                
            center, dims = u.box_to_ellipse(box)

            angle = 0 if angles is None else angles[i]

            # solid ellipse on mask
            cv2.ellipse(mask, center, dims, angle, 0, 360, (255), -1)

        # get everything but the ellipse
        mask_inv = cv2.bitwise_not(mask)
        no_faces = cv2.bitwise_and(image, image, mask=mask_inv)
        blurred_faces = cv2.bitwise_and(tempImg, tempImg, mask=mask)
        composite = cv2.add(no_faces, blurred_faces)

        image[:, :] = composite[:, :]

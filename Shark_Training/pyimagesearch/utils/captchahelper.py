#=============================================================================#
#                                                                             #
# MODIFIED: 27-Jun-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

import imutils
import cv2

#-----------------------------------------------------------------------------#
def preprocess(image, width, height):

    # Measure the spatial dimensions of the image
    (h, w) = image.shape[:2]

    # Resize to be square
    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)

    # Determine the padding values for the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # Pad the image  and force size to handle any rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
                               cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image

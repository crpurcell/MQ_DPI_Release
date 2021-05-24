#=============================================================================#
#                                                                             #
# MODIFIED: 12-Sep-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

import imutils
import cv2

#-----------------------------------------------------------------------------#
class AspectAwarePreprocessor:
    """
    Class to resize an image prior to using with  machine learning algorithms.
    This version preserves the aspect ratio of the input image.
    """

    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width    # target image width
        self.height = height  # target image height
        self.inter = inter    # interpolarion, INTER_AREA=3

    def preprocess(self, image):
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        # Resize using smaller dimension and record the change in the other
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        # Crop the image
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        # Force resize to correct any rounding errors
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.inter)

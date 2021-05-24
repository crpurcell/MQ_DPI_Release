#=============================================================================#
#                                                                             #
# MODIFIED: 15-Jan-2019 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

import cv2

#-----------------------------------------------------------------------------#
class MeanPreprocessor:

    def __init__(self, rMean, gMean, bMean, rgbOrder=True):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean
        self.rgbOrder = rgbOrder

    def preprocess(self, image):
        # Split the image into its respective RGB channels
        if self.rgbOrder:
            (R, G, B) = cv2.split(image.astype("float32"))
        else:
            (B, G, R) = cv2.split(image.astype("float32"))

        # Subtract the means for each channel
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        # Merge the channels back together and return the image
        if self.rgbOrder:
            return cv2.merge([R, G, B])
        else:
            return cv2.merge([B, G, R])

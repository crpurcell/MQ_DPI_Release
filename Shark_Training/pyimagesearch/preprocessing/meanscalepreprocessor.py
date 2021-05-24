#=============================================================================#
#                                                                             #
# MODIFIED: 16-Jan-2019 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

import cv2

#-----------------------------------------------------------------------------#
class MeanScalePreprocessor:
    """
    Class used to scale the pixel values of an image before processing. The
    default behaviour is to simply pass through the images without alteration.
    Can also be used to subtract mean RGB values and scale 256-colour images
    to different ranges. Examples:

        # Scale 256 colour image to exactly [0, 1] range:
        (scale=255.0, offset=0.0)

        # Scale 256 colour image to exactly [-1, 1] range:
        (scale=127.5, offset=-1.0)

        # Subtract ImageNet means and scale to a range spanning ~1.0:
        (rMean=123.68, gMean=116.779, bMean=103.939, scale=255.0, offset=0.0)

    """

    def __init__(self, rMean=0.0, gMean=0.0, bMean=0.0, scale=1.0,
                 offset=0.0, rgbOrder=True):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean
        self.scale = scale
        self.offset = offset
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

        # Scale data (default of 255 scales RGB to [0, 1])
        R /= self.scale
        G /= self.scale
        B /= self.scale

        # Apply offset shift (default no offset)
        R += self.offset
        G += self.offset
        B += self.offset

        # Merge the channels back together and return the image
        if self.rgbOrder:
            return cv2.merge([R, G, B])
        else:
            return cv2.merge([B, G, R])

#=============================================================================#
#                                                                             #
# MODIFIED: 25-Jun-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

from keras.preprocessing.image import img_to_array

#-----------------------------------------------------------------------------#
class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        """Apply Keras function to correctly orders image channels."""
        return img_to_array(image, data_format=self.dataFormat)
